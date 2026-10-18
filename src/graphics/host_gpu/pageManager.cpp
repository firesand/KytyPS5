#include "graphics/host_gpu/pageManager.h"

#include "common/alignment.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostReadStats.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/writeWatch.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

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
#else
#include <unistd.h>

namespace {
std::atomic<uint64_t> g_protect_calls {0};
std::atomic<uint64_t> g_protect_bytes {0};
} // namespace
#endif

namespace Libs::Graphics {
namespace {

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;

[[noreturn]] void FailFast(const char* reason = nullptr) noexcept {
	std::fputs("PageManager fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "invalid page state", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(322);
}

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(322);
}

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		while (m_lock.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
			_mm_pause();
#elif defined(_M_ARM64)
			__yield();
#elif defined(__aarch64__)
			asm volatile("yield");
#endif
		}
	}
	~SpinGuard() { m_lock.clear(std::memory_order_release); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	std::atomic_flag& m_lock;
};

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  : 7 = 0;
		uint8_t access_watchers : 1 = 0;

		// covered: the page lies in a kernel write-watched view (WriteWatch), which records CPU
		// writes itself, so write watchers need no write protection there.
		[[nodiscard]] Common::VirtualMemory::Mode Perms(bool covered) const noexcept {
			if (access_watchers != 0) {
				return Common::VirtualMemory::Mode::NoAccess;
			}
			if (write_watchers != 0 && !covered) {
				return Common::VirtualMemory::Mode::Read;
			}
			return Common::VirtualMemory::Mode::ReadWrite;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers != 0) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == 0x7f) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 1);

	struct Region {
		std::atomic_flag                    lock = ATOMIC_FLAG_INIT;
		std::array<PageState, REGION_PAGES> pages;
		uint64_t                            base = 0;
		// Pages with write watchers (lock-free reads by write-watch span scans).
		std::atomic<uint32_t> write_watched {0};
		// Pages inside kernel write-watched views, as of WriteWatch::ViewsEpoch() covered_epoch.
		RegionBits covered;
		uint64_t   covered_epoch = 0;
	};

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
	}

	~Impl() {
		for (const auto& region: region_storage) {
			SpinGuard lock(region->lock);
			for (auto& page: region->pages) {
				if (page.write_watchers != 0 || page.access_watchers != 0) {
					FailFast("PageManager destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return vaddr < ADDRESS_SIZE ? regions[vaddr / REGION_SIZE].load(std::memory_order_acquire)
		                            : nullptr;
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		ptr->base    = index * REGION_SIZE;
		region_storage.push_back(std::move(region));
		sorted_regions.insert(std::upper_bound(sorted_regions.begin(), sorted_regions.end(), ptr,
		                                       [](const Region* a, const Region* b) { return a->base < b->base; }),
		                      ptr);
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	// Pages of the region inside kernel write-watched views (empty without write-watch). Caller
	// holds region.lock.
	static const RegionBits& CoveredPages(Region& region) noexcept {
		auto* write_watch = WriteWatch::Get();
		if (write_watch == nullptr) {
			return region.covered;
		}
		const auto epoch = write_watch->ViewsEpoch();
		if (region.covered_epoch != epoch) {
			write_watch->CoveredPages(region.base, &region.covered);
			region.covered_epoch = epoch;
		}
		return region.covered;
	}

	void ForEachWriteWatchedSpan(const std::function<void(uint64_t, uint64_t)>& func) {
		std::vector<std::pair<uint64_t, uint64_t>> spans;
		{
			std::lock_guard lock(region_mutex);
			for (const auto* region: sorted_regions) {
				if (region->write_watched.load(std::memory_order_acquire) == 0) {
					continue;
				}
				if (!spans.empty() && spans.back().second == region->base) {
					spans.back().second += REGION_SIZE;
				} else {
					spans.emplace_back(region->base, region->base + REGION_SIZE);
				}
			}
		}
		for (const auto& [start, end]: spans) {
			func(start, end - start);
		}
	}

	void ForEachWriteWatchedRun(uint64_t vaddr, uint64_t size,
	                            const std::function<void(uint64_t, uint64_t)>& func) {
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		std::vector<std::pair<uint64_t, uint64_t>> runs;
		for (auto chunk = begin; chunk < end;) {
			const auto region_base = Common::AlignDown(chunk, REGION_SIZE);
			const auto chunk_end   = std::min(end, region_base + REGION_SIZE);
			auto*      region      = FindRegion(chunk);
			if (region != nullptr && region->write_watched.load(std::memory_order_acquire) != 0) {
				SpinGuard  lock(region->lock);
				const auto first = static_cast<size_t>((chunk - region_base) / PAGE_SIZE);
				const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				for (auto page = first; page < last; ++page) {
					if (region->pages[page].write_watchers == 0) {
						continue;
					}
					const auto address = region_base + page * PAGE_SIZE;
					if (!runs.empty() && runs.back().second == address) {
						runs.back().second += PAGE_SIZE;
					} else {
						runs.emplace_back(address, address + PAGE_SIZE);
					}
				}
			}
			chunk = chunk_end;
		}
		for (const auto& [start, stop]: runs) {
			func(start, stop - start);
		}
	}

	void Protect(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
		g_protect_calls.fetch_add(1, std::memory_order_relaxed);
		g_protect_bytes.fetch_add(size, std::memory_order_relaxed);
		if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size, mode)) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32, vaddr,
			      static_cast<uint32_t>(mode));
		}
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		SpinGuard   lock(region.lock);
		const auto& covered               = CoveredPages(region);
		auto        perms                 = region.pages[first].Perms(covered.Get(first));
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms);
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};
		// Kernel write-watch: covered pages gaining their first write watcher are re-protected in
		// the kernel, as mprotect(Read) would have protected them: earlier writes to them were
		// nobody's concern and must not be reported as new ones.
		size_t     rearm_begin = 0;
		size_t     rearm_end   = 0;
		const auto flush_rearm = [&] {
			if (rearm_end != rearm_begin) {
				WriteWatch::Get()->Rearm(base_addr + rearm_begin * PAGE_SIZE,
				                         (rearm_end - rearm_begin) * PAGE_SIZE);
			}
			rearm_begin = rearm_end = 0;
		};
		(void)flush_rearm; // read-watcher and untrack instantiations never re-arm

		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const bool page_covered = covered.Get(page_index);
			const auto old_perms    = page.Perms(page_covered);
			const auto new_count    = update ? page.AddDelta<track ? 1 : -1, is_read>(address)
			                                 : page.AddDelta<0, is_read>(address);
			const auto new_perms    = page.Perms(page_covered);

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if constexpr (!is_read) {
				if (watcher_edge) {
					if constexpr (track) {
						region.write_watched.fetch_add(1, std::memory_order_acq_rel);
						if (page_covered) {
							if (rearm_end != page_index) {
								flush_rearm();
								rearm_begin = page_index;
							}
							rearm_end = page_index + 1;
						}
					} else {
						region.write_watched.fetch_sub(1, std::memory_order_acq_rel);
					}
				}
			}
			if (watcher_edge && old_perms != new_perms) {
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
		if constexpr (track && !is_read) {
			flush_rearm();
		}
	}

	template <bool track, bool is_read>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false>(*region, region_base, first, last);
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::vector<Region*>                    sorted_regions; // by base, under region_mutex
	std::mutex                              region_mutex;
};

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {}

PageManager::~PageManager() = default;

std::pair<uint64_t, uint64_t> PageManager::TakeProtectStats() {
	return {g_protect_calls.exchange(0, std::memory_order_relaxed),
	        g_protect_bytes.exchange(0, std::memory_order_relaxed)};
}

uint64_t PageManager::GetPageSize() const {
	return PAGE_SIZE;
}

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	BudgetScope budget(BudgetProbe::Protect, GpuThreadMark::on);
	if (base_addr % REGION_SIZE != 0 || base_addr >= ADDRESS_SIZE ||
	    REGION_SIZE > ADDRESS_SIZE - base_addr) {
		Fatal("invalid tracking region base 0x%016" PRIx64, base_addr);
	}

	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	if (start_range.first == REGION_PAGES) {
		FailFast("empty region watcher mask");
	}
	const auto first = start_range.first;
	const auto last  = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask);
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

void PageManager::ForEachWriteWatchedSpan(
    const std::function<void(uint64_t, uint64_t)>& func) const {
	m_impl->ForEachWriteWatchedSpan(func);
}

void PageManager::ForEachWriteWatchedRun(
    uint64_t vaddr, uint64_t size, const std::function<void(uint64_t, uint64_t)>& func) const {
	m_impl->ForEachWriteWatchedRun(vaddr, size, func);
}

} // namespace Libs::Graphics
