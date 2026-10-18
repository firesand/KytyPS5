#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>
#include <cstdlib>
#include <functional>

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// IsRegionCpuModified(...) && !IsRegionGpuModified(...) with one lock acquisition per region
	// instead of two (BufferCache::ObtainBuffer's stream-path test, hot on the GPU thread).
	[[nodiscard]] bool IsRegionCpuModifiedNotGpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	// Marks the range GPU-modified unless a page in it is CPU-dirty; returns false and leaves the
	// range untouched on such a conflict. Callable from any thread (fault-processing operations).
	[[nodiscard]] bool TryMarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);
	// Kernel write-watch (writeWatch.h; no-op without it). Collects, re-protecting them in the
	// kernel, the pages of [vaddr, vaddr + size) written since their last collection that carry
	// write watchers, the pages a CPU write would have faulted on: func(vaddr, size). size == 0
	// collects every region span holding write-watched pages. Any thread; func runs with no
	// tracker or PageManager lock held, and must apply every run (WriteWatch's invariant). The
	// scan also re-protects written pages the filter drops; that is safe: a page without write
	// watchers is CPU-dirty already (its next upload copies the current bytes) or watched by no
	// one, and gaining its first watcher re-protects it anyway (PageManager).
	void CollectWrittenPages(uint64_t vaddr, uint64_t size,
	                         const std::function<void(uint64_t, uint64_t)>& func);
	// Applies collected CPU writes without a download (safe inside cache operations and GPU
	// completions): pages that are not GPU-dirty become CPU-dirty; GPU-dirty runs go to
	// gpu_part(vaddr, size) and need a full write-fault replay at a GPU-thread safe point.
	void MarkCpuWrittenUnlessGpuModified(uint64_t vaddr, uint64_t size,
	                                     const std::function<void(uint64_t, uint64_t)>& gpu_part);
	// A buffer was created or destroyed over the range: hot-page hashes no longer describe its
	// GPU copy. Called from the GPU thread.
	void InvalidatePageHashes(uint64_t vaddr, uint64_t size);
	static HotPageStats TakeHotStats() { return RegionManager::TakeHotStats(); }
	[[nodiscard]] uint64_t CpuWriteFaultCount() const noexcept {
		return m_cpu_write_fault_count.load(std::memory_order_relaxed);
	}
	void NoteCpuWriteFault() noexcept {
		m_cpu_write_fault_count.fetch_add(1, std::memory_order_relaxed);
	}
	// Regions whose CPU-dirty state changed since the last call (or that were flagged through
	// MarkBdaSyncPending): func(region_address, region_size). GPU thread.
	template <typename Func>
	void ForEachBdaSyncPendingRegion(bool include_hot, Func&& func) {
		if (m_bda_pending_count.load(std::memory_order_acquire) == 0 &&
		    (!include_hot || m_hot_region_count.load(std::memory_order_acquire) == 0)) {
			return;
		}
		std::vector<RegionManager*> pending;
		{
			std::lock_guard lock(m_region_mutex);
			for (const auto& manager: m_region_storage) {
				const bool flagged = manager->TakeBdaSyncPending();
				if (flagged || (include_hot && manager->HasHotPages())) {
					pending.push_back(manager.get());
				}
			}
		}
		for (auto* manager: pending) {
			func(manager->GetCpuAddr(), TRACKER_REGION_SIZE);
		}
	}
	template <typename Func>
	void ForEachBdaSyncPendingRegion(Func&& func) {
		ForEachBdaSyncPendingRegion(true, std::forward<Func>(func));
	}
	void MarkBdaSyncPending(uint64_t vaddr, uint64_t size);
	// KYTY_FAULT_AHEAD_PAGES (default 0 = off): pages a CPU write fault unprotects ahead. The
	// window gave no measurable frame-rate gain and marks pages CPU-dirty that a pending GPU
	// write-fault record may still own (audit 2026-10-12), so it is opt-in.
	static size_t FaultAheadPages() {
		static const size_t pages = [] {
			const char* value = std::getenv("KYTY_FAULT_AHEAD_PAGES");
			return value != nullptr ? static_cast<size_t>(std::atoi(value)) : size_t {0};
		}();
		return pages;
	}
	// CPU write fault: like InvalidateRegion, but widens the unprotected range forward over
	// clean pages (RegionManager::ExtendWriteFault) so sequential fills fault once per window.
	template <typename Flush>
	void InvalidateWriteFault(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		const auto ahead = FaultAheadPages();
		if (ahead <= 1) {
			InvalidateRegion(vaddr, size, std::forward<Flush>(on_flush));
			return;
		}
		CheckNotInUploadCallback();
		NoteCpuWriteFault();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				const auto address  = manager->GetCpuAddr() + offset;
				const auto extended = manager->ExtendWriteFault(address, bytes, ahead);
				manager->ChangeState<DirtySource::Cpu, true>(address, extended);
				return false;
			}();
			if (should_flush) {
				on_flush();
				InvalidateRegion(vaddr, size, std::forward<Flush>(on_flush));
			}
		});
	}
	// Removes protection from a range and flushes GPU-owned data when required.
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		CheckNotInUploadCallback();
		NoteCpuWriteFault();

		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
	}
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	void ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                           const char* operation) const noexcept;
	void ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                               const char* operation);
#else
	void ValidateGpuDirtyPages(const RangeSet&, uint64_t, uint64_t, const char*) const noexcept {}
	void ValidateGpuDirtyOwnership(const RangeSet&, uint64_t, uint64_t, const char*) {}
#endif

	// Runs of pages a CPU-write synchronization must examine (see RegionManager). GPU thread.
	template <typename Func>
	void ForEachSyncCandidateRun(uint64_t vaddr, uint64_t size, Func&& func) {
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			manager->ForEachSyncCandidateRun(manager->GetCpuAddr() + offset, bytes, func);
		});
	}

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			const auto       address = manager->GetCpuAddr() + offset;
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(address, bytes, func);
			if constexpr (clear) {
				manager->template ChangeState<DirtySource::Gpu, false>(address, bytes);
			}
		});
	}

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t, const void*>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		CheckNotInUploadCallback();
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		RegionManager::ResetSnapshots();
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			manager->lock.lock();
			manager->ForEachModifiedRange<DirtySource::Cpu, true, /*filter_hot=*/true>(
			    manager->GetCpuAddr() + offset, bytes, range_func, /*keep_hot=*/!is_written);
			if (!is_written) {
				manager->lock.unlock();
			}
		});
		upload_func();
		if (is_written) {
			Iterate<false>(vaddr, size,
			               [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				               manager->template ChangeState<DirtySource::Gpu, true>(
				                   manager->GetCpuAddr() + offset, bytes);
				               manager->lock.unlock();
			               });
		}
		s_upload_owner = previous_upload_owner;
	}

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	inline static thread_local const MemoryTracker* s_upload_owner = nullptr;

	void CheckNotInUploadCallback() const noexcept {
		if (s_upload_owner == this) {
			EXIT("memory tracker re-entered from upload callback\n");
		}
	}

	template <bool create, typename Func>
	bool Iterate(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		using Result = std::invoke_result_t<Func, RegionManager*, uint64_t, uint64_t>;
		constexpr bool returns_bool = std::is_same_v<Result, bool>;
		uint64_t       remaining    = size;
		uint64_t       index        = vaddr / TRACKER_REGION_SIZE;
		uint64_t       offset       = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr && create) {
				manager = GetOrCreateRegion(index);
			}
			if (manager != nullptr) {
				if constexpr (returns_bool) {
					if (func(manager, offset, bytes)) {
						return true;
					}
				} else {
					func(manager, offset, bytes);
				}
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return false;
	}

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	PageManager&                                   m_page_manager;
	std::atomic<uint32_t>                          m_bda_pending_count {0};
	std::atomic<uint32_t>                          m_hot_region_count {0};
	std::atomic<uint64_t>                          m_cpu_write_fault_count {1};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
