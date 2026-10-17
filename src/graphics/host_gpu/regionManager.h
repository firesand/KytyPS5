#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <cstdlib>
#include <array>
#include <vector>
#include <mutex>
#include <cstring>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <memory>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
#if defined(__x86_64__) || defined(_M_X64)
			_mm_pause();
#elif defined(_M_ARM64)
			__yield();
#elif defined(__aarch64__)
			asm volatile("yield");
#endif
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		static thread_local const uint32_t tid =
		    static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
		return tid;
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

// XXH3 of one tracker page (defined in memoryTracker.cpp so this header stays light).
uint64_t HashTrackerPage(const void* page) noexcept;

// Hot-page content tracking counters (GPU thread), reported by the fault histogram.
struct HotPageStats {
	uint64_t hashed      = 0;
	uint64_t skipped     = 0;
	uint64_t demoted     = 0;
	uint64_t invalidated = 0; // hash no longer matched the page after the upload copied it
};

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr,
	              std::atomic<uint32_t>* bda_pending_count = nullptr,
	              std::atomic<uint32_t>* hot_region_count  = nullptr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr),
	      m_bda_pending_count(bda_pending_count), m_hot_region_count(hot_region_count) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_maybe.Clear();
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }
	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().FirstRangeFrom(start).first < end;
	}

	// Hot pages: the CPU writes them again and again (one write fault per page per sync point).
	// After KYTY_HOT_PAGE_FAULTS faults (default 8, 0 disables) a page stays CPU-dirty and
	// writable; every synchronization re-uploads it instead of re-protecting it.
	static uint32_t HotPageFaultThreshold() {
		static const uint32_t threshold = [] {
			const char* value = std::getenv("KYTY_HOT_PAGE_FAULTS");
			return value != nullptr ? static_cast<uint32_t>(std::atoi(value)) : 8u;
		}();
		return threshold;
	}
	[[nodiscard]] bool HasHotPages() const { return m_has_hot.load(std::memory_order_relaxed); }
	// Pages a CPU-write synchronization has to examine in [vaddr, vaddr + size): CPU-dirty pages
	// (new writes; hot pages stay CPU-dirty) and, for safety, the hot set. The bits are copied under
	// the region lock and emitted as runs after it is released, so the caller can synchronize the
	// owning buffers (which take this lock again) without nesting.
	template <typename Func>
	void ForEachSyncCandidateRun(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		RegionBits candidates;
		{
			std::scoped_lock guard(lock);
			candidates = RegionBits(m_cpu_dirty | m_maybe, start, end);
			if (HasHotPages()) {
				for (const auto [first, last]: RegionBits(m_hot, start, end)) {
					candidates.SetRange(first, last);
				}
			}
		}
		for (const auto [first, last]: candidates) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	static size_t FaultAheadCap() {
		static const size_t cap = [] {
			const char* value = std::getenv("KYTY_FAULT_AHEAD_PAGES");
			return value != nullptr ? static_cast<size_t>(std::atoi(value)) : size_t {256};
		}();
		return cap;
	}

	[[nodiscard]] std::pair<size_t, size_t> ExtendWriteFaultMaybe(size_t start, size_t end) {
		const auto cap = FaultAheadCap();
		if (cap <= 1) {
			m_last_fault_end = end;
			return {end, end};
		}
		if (start == m_last_fault_end) {
			m_current_ahead = (m_current_ahead == 0) ? 4 : std::min(cap, m_current_ahead * 2);
		} else {
			m_current_ahead = 0;
		}
		if (m_current_ahead == 0) {
			m_last_fault_end = end;
			return {end, end};
		}
		auto limit = std::min<size_t>(TRACKER_REGION_PAGES, end + m_current_ahead);
		limit      = std::min<size_t>(limit, m_cpu_dirty.FirstRangeFrom(end).first);
		limit      = std::min<size_t>(limit, m_gpu_dirty.FirstRangeFrom(end).first);
		limit      = std::min<size_t>(limit, m_maybe.FirstRangeFrom(end).first);
		const auto maybe_end = std::max(end, limit);
		m_last_fault_end     = maybe_end;
		return {end, maybe_end};
	}

	void ApplyWriteFault(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		ChangeState<DirtySource::Cpu, true>(vaddr, size);
		const auto [maybe_start, maybe_end] = ExtendWriteFaultMaybe(start, end);
		if (maybe_start < maybe_end) {
			m_maybe.SetRange(maybe_start, maybe_end);
			UpdateProtection<false, false>();
		}
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (RegionBits(m_gpu_dirty, start, end).Any()) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
			m_maybe.UnsetRange(start, end);
			if (const auto threshold = HotPageFaultThreshold(); threshold != 0 && end - start <= 16) {
				for (auto page = start; page < end; ++page) {
					if (m_fault_counts[page] < 255 && ++m_fault_counts[page] == threshold) {
						m_hot.Set(page);
						if (!m_has_hot.exchange(true, std::memory_order_acq_rel) &&
						    m_hot_region_count != nullptr) {
							m_hot_region_count->fetch_add(1, std::memory_order_acq_rel);
						}
					}
				}
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (RegionBits(m_cpu_dirty, start, end).Any()) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
			m_maybe.UnsetRange(start, end);
			m_hash_valid.UnsetRange(start, end); // the GPU copy no longer matches any CPU hash
		}
		auto& bits = GetBits<source>();
		if constexpr (enable) {
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu && enable) {
			MarkBdaSyncPending();
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>();
		} else {
			UpdateProtection<enable, true>();
		}
	}

	// filter_hot (upload path only): hot pages whose bytes are unchanged since their last upload
	// are left out of the emitted ranges; see EmitUploadRun.
	template <DirtySource source, bool clear, bool filter_hot = false, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func, bool keep_hot = true) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		RegionBits mask;
		if constexpr (source == DirtySource::Cpu) {
			mask = RegionBits(m_cpu_dirty | m_maybe, start, end);
		} else {
			mask = RegionBits(m_gpu_dirty, start, end);
		}
		if (mask.None()) {
			return;
		}
		if constexpr (clear) {
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) {
				m_maybe.UnsetRange(start, end);
				if (keep_hot && HasHotPages()) {
					// Hot pages stay dirty (and writable) so the next sync uploads them again,
					// unless their content stayed identical for HotDemoteSyncs syncs: those leave
					// the hot set, stay clean here and get re-protected below.
					const auto demote = HotDemoteSyncs();
					for (const auto [first, last]: RegionBits(m_hot, start, end)) {
						for (auto page = first; page < last; ++page) {
							if (!mask.Get(page)) continue;
							if (demote != 0 && m_unchanged[page] >= demote) {
								m_hot.Unset(page);
								m_fault_counts[page] = 0;
								m_unchanged[page]    = 0;
								m_hash_valid.Unset(page); // non-hot uploads bypass the hash: forget it
								s_hot_stats.demoted++;
								continue;
							}
							bits.Set(page);
						}
					}
					if (m_hot.None() && m_has_hot.exchange(false, std::memory_order_acq_rel) &&
					    m_hot_region_count != nullptr) {
						m_hot_region_count->fetch_sub(1, std::memory_order_acq_rel);
					}
				} else if (!keep_hot && HasHotPages()) {
					// The GPU takes ownership (written binding): the pages leave the hot set.
					m_hot.UnsetRange(start, end);
					for (auto page = start; page < end; ++page) m_fault_counts[page] = 0;
					if (m_hot.None() && m_has_hot.exchange(false, std::memory_order_acq_rel) &&
					    m_hot_region_count != nullptr) {
						m_hot_region_count->fetch_sub(1, std::memory_order_acq_rel);
					}
				}
			}
			if constexpr (source == DirtySource::Cpu) {
				UpdateProtection<true, false>();
			} else {
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			if constexpr (filter_hot && source == DirtySource::Cpu) {
				EmitUploadRun(first, last, func);
			} else {
				if constexpr (source == DirtySource::Cpu) {
					m_hash_valid.UnsetRange(first, last); // uploaded without a hash
				}
				func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
			}
		}
	}

	// KYTY_HOT_PAGE_HASH=0 disables the content check; KYTY_HOT_DEMOTE_SYNCS (default 32, 0 never)
	// is how many consecutive unchanged syncs make a hot page leave the hot set. Zero Company's
	// menu scene: 4 → 6.0 fps, 32 → 9.9 fps, never → 9.1 fps (baseline 5.2).
	static bool HotHashEnabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_HOT_PAGE_HASH");
			return value == nullptr || value[0] != '0';
		}();
		return enabled;
	}
	static uint8_t HotDemoteSyncs() {
		static const uint8_t syncs = [] {
			const char* value = std::getenv("KYTY_HOT_DEMOTE_SYNCS");
			const int   parsed = value != nullptr ? std::atoi(value) : 32;
			return static_cast<uint8_t>(std::clamp(parsed, 0, 255));
		}();
		return syncs;
	}
	static HotPageStats TakeHotStats() { return std::exchange(s_hot_stats, {}); }

	// Upload snapshots (GPU thread): immutable copies of the changed hot pages emitted by one
	// MemoryTracker::ForEachUploadRange call. The remembered hash is the snapshot's hash and the
	// upload copies the snapshot, so the hash always describes the bytes the GPU holds; a guest
	// write between hashing and copying (A -> B -> A, review 2026-10-12) can no longer leave a
	// stale "unchanged" verdict. Reset at the start of every upload call.
	static void ResetSnapshots() { s_snapshot_used = 0; }

	// Buffer created or destroyed over this range: its GPU copy is new, so no hash is current.
	void InvalidatePageHashes(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		m_hash_valid.UnsetRange(start, end);
	}

private:
	static uint8_t* TakeSnapshotPage() {
		if (s_snapshot_used == s_snapshot_pool.size()) {
			s_snapshot_pool.push_back(std::make_unique<std::array<uint8_t, TRACKER_PAGE_SIZE>>());
		}
		return s_snapshot_pool[s_snapshot_used++]->data();
	}
	static void ReleaseSnapshotPage() { --s_snapshot_used; }

	// Emits the upload ranges of one dirty run as func(address, bytes, snapshot): snapshot is
	// null for plain guest memory, or the immutable copy to upload for a changed hot page. Hot
	// pages whose snapshot hash equals the hash remembered at their previous upload are skipped.
	// Caller holds `lock`; hot pages are writable, so reading them cannot fault.
	template <typename Func>
	void EmitUploadRun(size_t first, size_t last, Func&& func) {
		if (!HotHashEnabled() || !HasHotPages()) {
			m_hash_valid.UnsetRange(first, last); // uploaded without a hash
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE,
			     nullptr);
			return;
		}
		size_t     run_begin = first;
		const auto flush_run = [&](size_t end) {
			if (run_begin < end) {
				func(m_cpu_addr + run_begin * TRACKER_PAGE_SIZE,
				     (end - run_begin) * TRACKER_PAGE_SIZE, nullptr);
			}
		};
		for (size_t page = first; page < last; ++page) {
			if (!m_hot.Get(page)) {
				m_hash_valid.Unset(page); // uploaded below without a hash: the old one is stale
				continue;
			}
			auto* snapshot = TakeSnapshotPage();
			std::memcpy(snapshot, reinterpret_cast<const void*>(m_cpu_addr + page * TRACKER_PAGE_SIZE),
			            TRACKER_PAGE_SIZE);
			const auto hash = HashTrackerPage(snapshot);
			s_hot_stats.hashed++;
			if (m_hash_valid.Get(page) && m_page_hash != nullptr && (*m_page_hash)[page] == hash) {
				if (m_unchanged[page] < 255) ++m_unchanged[page];
				s_hot_stats.skipped++;
				ReleaseSnapshotPage();
				flush_run(page);
				run_begin = page + 1;
				continue;
			}
			if (m_page_hash == nullptr) {
				m_page_hash = std::make_unique<std::array<uint64_t, TRACKER_REGION_PAGES>>();
			}
			(*m_page_hash)[page] = hash;
			m_hash_valid.Set(page);
			m_unchanged[page] = 0;
			flush_run(page);
			func(m_cpu_addr + page * TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE, snapshot);
			run_begin = page + 1;
		}
		flush_run(last);
	}

public:

	TrackingSpinLock lock;

	// BDA (pointer) shaders must see CPU writes; instead of re-walking every cached buffer per
	// draw, PrepareBda only synchronizes regions that gained CPU-dirty pages (or new buffers).
	void MarkBdaSyncPending() {
		if (!bda_sync_pending.exchange(true, std::memory_order_acq_rel) &&
		    m_bda_pending_count != nullptr) {
			m_bda_pending_count->fetch_add(1, std::memory_order_acq_rel);
		}
	}
	[[nodiscard]] bool TakeBdaSyncPending() {
		if (!bda_sync_pending.exchange(false, std::memory_order_acq_rel)) {
			return false;
		}
		if (m_bda_pending_count != nullptr) {
			m_bda_pending_count->fetch_sub(1, std::memory_order_acq_rel);
		}
		return true;
	}
	std::atomic<bool> bda_sync_pending {false};

private:
	PageManager&           m_page_manager;
	uint64_t               m_cpu_addr          = 0;
	std::atomic<uint32_t>* m_bda_pending_count = nullptr;
	std::atomic<uint32_t>* m_hot_region_count  = nullptr;
	std::atomic<bool>      m_has_hot {false};
	RegionBits             m_hot;
	std::array<uint8_t, TRACKER_REGION_PAGES> m_fault_counts {};
	// Hot-page content tracking (GPU thread, under `lock`).
	std::unique_ptr<std::array<uint64_t, TRACKER_REGION_PAGES>> m_page_hash;
	RegionBits                                                  m_hash_valid;
	std::array<uint8_t, TRACKER_REGION_PAGES>                   m_unchanged {};
	inline static HotPageStats                                  s_hot_stats {};
	inline static thread_local std::vector<std::unique_ptr<std::array<uint8_t, TRACKER_PAGE_SIZE>>>
	                                        s_snapshot_pool;
	inline static thread_local size_t      s_snapshot_used = 0;
	template <bool track, bool is_read>
	void UpdateProtection() {
		const auto protection = is_read ? ~m_gpu_dirty : (m_cpu_dirty | m_maybe);
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	RegionBits   m_maybe;
	size_t       m_last_fault_end = SIZE_MAX;
	size_t       m_current_ahead  = 0;
	RegionBits   m_writable;
	RegionBits   m_readable;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
