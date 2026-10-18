#include "graphics/host_gpu/writeWatch.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <string>
#include <utility>

#if defined(__linux__)
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/fs.h>
#include <linux/userfaultfd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#if defined(PAGEMAP_SCAN) && defined(UFFD_FEATURE_WP_ASYNC) && defined(UFFD_USER_MODE_ONLY)
#define KYTY_WRITE_WATCH_SUPPORTED 1
#endif
#endif

namespace Libs::Graphics {

namespace {

constexpr uint64_t PAGE_SIZE  = TRACKER_PAGE_SIZE;
constexpr uint64_t SCAN_CHUNK = 64ull * 1024ull * 1024ull; // bounds a run-count early return's re-walk
constexpr size_t   MAX_LABELS = 4096;

std::atomic<uint64_t> g_collects {0};
std::atomic<uint64_t> g_ioctls {0};
std::atomic<uint64_t> g_pages {0};
std::atomic<uint64_t> g_errors {0};
std::atomic<uint64_t> g_ns {0};
std::atomic<uint64_t> g_label_checks {0};
std::atomic<uint64_t> g_label_written {0};

// KYTY_WRITE_WATCH_LOG=1: one line per collection that reported pages (first 4000).
bool LogEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_WRITE_WATCH_LOG");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void LogRuns(const char* what, uint64_t vaddr, uint64_t size,
             const std::vector<std::pair<uint64_t, uint64_t>>& runs) {
	static std::atomic<uint32_t> lines {0};
	if (!LogEnabled() || runs.empty() || lines.fetch_add(1, std::memory_order_relaxed) >= 4000) {
		return;
	}
	std::string text;
	for (size_t i = 0; i < runs.size() && i < 8; i++) {
		text += fmt::format(" 0x{:x}+0x{:x}", runs[i].first, runs[i].second - runs[i].first);
	}
	LOGF("WriteWatch: %s 0x%016" PRIx64 "+0x%" PRIx64 " reported %zu run(s):%s%s\n", what, vaddr,
	     size, runs.size(), text.c_str(), runs.size() > 8 ? " ..." : "");
}

void AppendRun(std::vector<std::pair<uint64_t, uint64_t>>* runs, uint64_t start, uint64_t end) {
	if (!runs->empty() && runs->back().second == start) {
		runs->back().second = end;
	} else {
		runs->emplace_back(start, end);
	}
}

} // namespace

WriteWatch* WriteWatch::Get() noexcept {
	static WriteWatch* const instance = Create();
	return instance;
}

WriteWatch* WriteWatch::Create() noexcept {
	const char* value = std::getenv("KYTY_WRITE_WATCH");
	if (value == nullptr || value[0] != '1') {
		return nullptr;
	}
#if defined(KYTY_WRITE_WATCH_SUPPORTED)
	// vm.unprivileged_userfaultfd=0 still allows a user-mode-only context. Async write-protect
	// never delivers a fault message, so kernel-mode writes (read(2) into guest memory) are
	// recorded too (tools/writewatch-proto, views test).
	const int uffd =
	    static_cast<int>(::syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY));
	if (uffd < 0) {
		Log::WriteToConsoleAndLog(fmt::format("WriteWatch: off (userfaultfd: {})\n", std::strerror(errno)));
		return nullptr;
	}
	constexpr uint64_t features =
	    UFFD_FEATURE_WP_ASYNC | UFFD_FEATURE_WP_HUGETLBFS_SHMEM | UFFD_FEATURE_WP_UNPOPULATED;
	uffdio_api api {};
	api.api      = UFFD_API;
	api.features = features;
	if (::ioctl(uffd, UFFDIO_API, &api) != 0 || (api.features & features) != features) {
		Log::WriteToConsoleAndLog(fmt::format("WriteWatch: off (UFFDIO_API: {}, features 0x{:x})\n",
		                                      std::strerror(errno), api.features));
		::close(uffd);
		return nullptr;
	}
	const int pagemap = ::open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
	if (pagemap < 0) {
		Log::WriteToConsoleAndLog(fmt::format("WriteWatch: off (/proc/self/pagemap: {})\n", std::strerror(errno)));
		::close(uffd);
		return nullptr;
	}
	// PAGEMAP_SCAN exists (ENOTTY otherwise): a report-only scan of one page of our own memory.
	alignas(PAGE_SIZE) static uint8_t probe_page[PAGE_SIZE];
	probe_page[0] = 1;
	pm_scan_arg probe {};
	probe.size          = sizeof(probe);
	probe.start         = reinterpret_cast<uint64_t>(probe_page);
	probe.end           = probe.start + PAGE_SIZE;
	probe.category_mask = PAGE_IS_PRESENT;
	probe.return_mask   = PAGE_IS_PRESENT;
	if (::ioctl(pagemap, PAGEMAP_SCAN, &probe) < 0) {
		Log::WriteToConsoleAndLog(fmt::format("WriteWatch: off (PAGEMAP_SCAN: {})\n", std::strerror(errno)));
		::close(pagemap);
		::close(uffd);
		return nullptr;
	}
	Log::WriteToConsoleAndLog("WriteWatch: on (userfaultfd async write-protect + PAGEMAP_SCAN)\n");
	return new WriteWatch(uffd, pagemap); // process lifetime
#else
	Log::WriteToConsoleAndLog("WriteWatch: off (needs Linux with userfaultfd WP_ASYNC and PAGEMAP_SCAN)\n");
	return nullptr;
#endif
}

WriteWatch::WriteWatch(int uffd, int pagemap) noexcept: m_uffd(uffd), m_pagemap(pagemap) {}

void WriteWatch::RegisterView(uint64_t vaddr, uint64_t size) noexcept {
	if (size == 0) {
		return;
	}
#if defined(KYTY_WRITE_WATCH_SUPPORTED)
	std::unique_lock lock(m_views_mutex);
	uffdio_register reg {};
	reg.range.start = vaddr;
	reg.range.len   = size;
	reg.mode        = UFFDIO_REGISTER_MODE_WP;
	if (::ioctl(m_uffd, UFFDIO_REGISTER, &reg) != 0) {
		// Its pages may already count as covered (a re-map over a live view) and carry write
		// watchers without write protection: CPU writes there would go unseen.
		EXIT("WriteWatch: view 0x%016" PRIx64 "+0x%" PRIx64 " could not be registered (%s); run "
		     "without KYTY_WRITE_WATCH\n",
		     vaddr, size, std::strerror(errno));
	}
	// Arm only the part that was not registered before: re-arming a live view would discard
	// writes nobody collected. A re-mapped part that stays unarmed reports its populated page
	// tables as written once (conservative).
	RangeSet fresh;
	fresh.Add(vaddr, size);
	m_views.ForEachInRange(vaddr, size,
	                       [&](uint64_t start, uint64_t end) { fresh.Subtract(start, end - start); });
	fresh.ForEach([&](uint64_t start, uint64_t end) {
		uffdio_writeprotect wp {};
		wp.range.start = start;
		wp.range.len   = end - start;
		wp.mode        = UFFDIO_WRITEPROTECT_MODE_WP;
		if (::ioctl(m_uffd, UFFDIO_WRITEPROTECT, &wp) != 0) {
			LOGF("WriteWatch: arming 0x%016" PRIx64 "+0x%" PRIx64 " failed (%s)\n", start,
			     end - start, std::strerror(errno));
		}
	});
	m_views.Add(vaddr, size);
	m_views_epoch.fetch_add(1, std::memory_order_acq_rel);
#else
	(void)vaddr;
#endif
}

void WriteWatch::UnregisterView(uint64_t vaddr, uint64_t size) noexcept {
	if (size == 0) {
		return;
	}
	std::unique_lock lock(m_views_mutex);
	// The placeholder mapping that replaces the view drops the kernel registration itself.
	m_views.Subtract(vaddr, size);
	m_views_epoch.fetch_add(1, std::memory_order_acq_rel);
}

void WriteWatch::CoveredPages(uint64_t region_base, RegionBits* covered) const noexcept {
	*covered = RegionBits {};
	std::shared_lock lock(m_views_mutex);
	m_views.ForEachInRange(region_base, TRACKER_REGION_SIZE, [&](uint64_t start, uint64_t end) {
		covered->SetRange(static_cast<size_t>((start - region_base) / PAGE_SIZE),
		                  static_cast<size_t>((end - region_base + PAGE_SIZE - 1) / PAGE_SIZE));
	});
}

bool WriteWatch::IsCovered(uint64_t vaddr, uint64_t size) const noexcept {
	std::shared_lock lock(m_views_mutex);
	return m_views.Contains(vaddr, size);
}

void WriteWatch::ScanChunk(uint64_t start, uint64_t end,
                           std::vector<std::pair<uint64_t, uint64_t>>* runs) noexcept {
#if defined(KYTY_WRITE_WATCH_SUPPORTED)
	static thread_local std::array<page_region, 512> vec {};
	uint64_t cursor = start;
	while (cursor < end) {
		pm_scan_arg arg {};
		arg.size          = sizeof(arg);
		arg.flags         = PM_SCAN_WP_MATCHING | PM_SCAN_CHECK_WPASYNC;
		arg.start         = cursor;
		arg.end           = end;
		arg.vec           = reinterpret_cast<uint64_t>(vec.data());
		arg.vec_len       = vec.size();
		arg.category_mask = PAGE_IS_WRITTEN;
		arg.return_mask   = PAGE_IS_WRITTEN;
		const long count  = ::ioctl(m_pagemap, PAGEMAP_SCAN, &arg);
		g_ioctls.fetch_add(1, std::memory_order_relaxed);
		if (count < 0 || arg.walk_end <= cursor) {
			// A failed call may already have re-protected pages it could not report (the
			// prototype's EPERM case): everything from the cursor on counts as written.
			g_errors.fetch_add(1, std::memory_order_relaxed);
			static std::atomic<uint32_t> reported {0};
			if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
				LOGF("WriteWatch: scan of 0x%016" PRIx64 "-0x%016" PRIx64 " failed (%s); range "
				     "treated as written\n",
				     cursor, end, count < 0 ? std::strerror(errno) : "no progress");
			}
			AppendRun(runs, cursor, end);
			return;
		}
		for (long i = 0; i < count; i++) {
			AppendRun(runs, vec[i].start, vec[i].end);
		}
		cursor = arg.walk_end;
	}
#else
	(void)start;
	(void)end;
	(void)runs;
#endif
}

bool WriteWatch::AnyWritten(uint64_t start, uint64_t end) noexcept {
#if defined(KYTY_WRITE_WATCH_SUPPORTED)
	page_region region {};
	pm_scan_arg arg {};
	arg.size          = sizeof(arg);
	arg.start         = start;
	arg.end           = end;
	arg.vec           = reinterpret_cast<uint64_t>(&region);
	arg.vec_len       = 1;
	arg.max_pages     = 1;
	arg.category_mask = PAGE_IS_WRITTEN;
	arg.return_mask   = PAGE_IS_WRITTEN;
	const long count  = ::ioctl(m_pagemap, PAGEMAP_SCAN, &arg);
	g_ioctls.fetch_add(1, std::memory_order_relaxed);
	return count != 0; // an error is "written"
#else
	(void)start;
	(void)end;
	return true;
#endif
}

void WriteWatch::NoteLabelRuns(const std::vector<std::pair<uint64_t, uint64_t>>& runs) noexcept {
	if (!m_has_labels.load(std::memory_order_acquire) || runs.empty()) {
		return;
	}
	std::lock_guard lock(m_labels_mutex);
	for (const auto& [start, end]: runs) {
		auto it = std::lower_bound(m_labels.begin(), m_labels.end(), Common::AlignDown(start, PAGE_SIZE),
		                           [](const Label& label, uint64_t page) { return label.page < page; });
		for (; it != m_labels.end() && it->page < end; ++it) {
			it->written = true;
		}
	}
}

bool WriteWatch::TouchesLabels(uint64_t vaddr, uint64_t size) noexcept {
	if (!m_has_labels.load(std::memory_order_acquire)) {
		return false;
	}
	std::lock_guard lock(m_labels_mutex);
	const auto      it = std::lower_bound(m_labels.begin(), m_labels.end(), Common::AlignDown(vaddr, PAGE_SIZE),
	                                      [](const Label& label, uint64_t page) { return label.page < page; });
	return it != m_labels.end() && it->page < vaddr + size;
}

void WriteWatch::Collect(uint64_t vaddr, uint64_t size, const RunFunc& func) noexcept {
	if (size == 0) {
		return;
	}
	// PAGEMAP_SCAN rejects an unaligned start (EINVAL), which would report the range as written.
	const auto first = Common::AlignDown(vaddr, PAGE_SIZE);
	const auto last  = Common::AlignUp(vaddr + size, PAGE_SIZE);
	vaddr            = first;
	size             = last - first;
	const auto begin = std::chrono::steady_clock::now();
	g_collects.fetch_add(1, std::memory_order_relaxed);
	// A label page found written is re-protected by the scan before NoteLabelRuns flags it; checks
	// in between must not trust the page's clean state (WasLabelWritten).
	const bool labels = TouchesLabels(vaddr, size);
	if (labels) {
		m_label_scans_active.fetch_add(1, std::memory_order_seq_cst);
	}
	static thread_local std::vector<std::pair<uint64_t, uint64_t>> runs;
	runs.clear();
	{
		// Shared: a view is unregistered (exclusive) before its placeholder replaces it, so no
		// scan sees a half-unmapped view.
		std::shared_lock lock(m_views_mutex);
		m_views.ForEachInRange(vaddr, size, [&](uint64_t start, uint64_t end) {
			for (auto chunk = start; chunk < end;) {
				const auto chunk_end = std::min(end, Common::AlignDown(chunk, SCAN_CHUNK) + SCAN_CHUNK);
				ScanChunk(chunk, chunk_end, &runs);
				chunk = chunk_end;
			}
		});
	}
	NoteLabelRuns(runs);
	LogRuns("collect", vaddr, size, runs);
	if (labels) {
		m_label_scan_seq.fetch_add(1, std::memory_order_seq_cst);
		m_label_scans_active.fetch_sub(1, std::memory_order_seq_cst);
	}
	g_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                         std::chrono::steady_clock::now() - begin)
	                                         .count()),
	               std::memory_order_relaxed);
	// The callback may re-enter Collect (nested trackers); work on a copy.
	const auto reported = runs;
	for (const auto& [start, end]: reported) {
		g_pages.fetch_add((end - start) / PAGE_SIZE, std::memory_order_relaxed);
		func(start, end - start);
	}
}

void WriteWatch::Rearm(uint64_t vaddr, uint64_t size) noexcept {
	Collect(vaddr, size, [](uint64_t, uint64_t) {});
}

bool WriteWatch::WasLabelWritten(uint64_t vaddr, uint64_t size) noexcept {
	g_label_checks.fetch_add(1, std::memory_order_relaxed);
	const auto seq_before = m_label_scan_seq.load(std::memory_order_seq_cst);
	if (m_label_scans_active.load(std::memory_order_seq_cst) != 0) {
		g_label_written.fetch_add(1, std::memory_order_relaxed);
		return true;
	}
	const auto first = Common::AlignDown(vaddr, PAGE_SIZE);
	const auto last  = Common::AlignUp(vaddr + std::max<uint64_t>(size, 1), PAGE_SIZE);
	bool       known = true;
	bool       flagged = false;
	{
		std::lock_guard lock(m_labels_mutex);
		for (auto page = first; page < last; page += PAGE_SIZE) {
			auto it = std::lower_bound(m_labels.begin(), m_labels.end(), page,
			                           [](const Label& label, uint64_t p) { return label.page < p; });
			if (it == m_labels.end() || it->page != page) {
				known = false;
				if (m_labels.size() < MAX_LABELS) {
					m_labels.insert(it, Label {page, false});
					m_has_labels.store(true, std::memory_order_release);
				}
			} else if (it->written) {
				flagged = true;
			}
		}
	}
	// Unknown pages were never re-protected for this purpose, and uncovered pages cannot be
	// checked: both mean "possibly written".
	bool written = !known || flagged || !IsCovered(first, last - first) || AnyWritten(first, last);
	// A scan that touched label pages overlapped this check: its re-protect may have hidden the
	// write and its flag may have come too late.
	written = written || m_label_scans_active.load(std::memory_order_seq_cst) != 0 ||
	          m_label_scan_seq.load(std::memory_order_seq_cst) != seq_before;
	if (written) {
		g_label_written.fetch_add(1, std::memory_order_relaxed);
	}
	return written;
}

void WriteWatch::CollectLabels(const RunFunc& func) noexcept {
	if (!m_has_labels.load(std::memory_order_acquire)) {
		return;
	}
	std::vector<uint64_t> pages;
	{
		std::lock_guard lock(m_labels_mutex);
		pages.reserve(m_labels.size());
		for (const auto& label: m_labels) {
			pages.push_back(label.page);
		}
	}
	// Adjacent label pages are scanned together.
	RangeSet ranges;
	for (const auto page: pages) {
		ranges.Add(page, PAGE_SIZE);
	}
	m_label_scans_active.fetch_add(1, std::memory_order_seq_cst);
	std::vector<std::pair<uint64_t, uint64_t>> runs;
	{
		std::shared_lock lock(m_views_mutex);
		ranges.ForEach([&](uint64_t range_start, uint64_t range_end) {
			m_views.ForEachInRange(range_start, range_end - range_start,
			                       [&](uint64_t start, uint64_t end) { ScanChunk(start, end, &runs); });
		});
	}
	{
		// Cleared after the label scans and before the caller's span scans: a label written
		// after this point is either still unprotected or flagged by the scan that protects it.
		std::lock_guard lock(m_labels_mutex);
		for (auto& label: m_labels) {
			label.written = false;
		}
	}
	m_label_scan_seq.fetch_add(1, std::memory_order_seq_cst);
	m_label_scans_active.fetch_sub(1, std::memory_order_seq_cst);
	for (const auto& [start, end]: runs) {
		g_pages.fetch_add((end - start) / PAGE_SIZE, std::memory_order_relaxed);
		func(start, end - start);
	}
}

WriteWatch::Stats WriteWatch::TakeStats() noexcept {
	Stats stats;
	stats.collects      = g_collects.exchange(0, std::memory_order_relaxed);
	stats.ioctls        = g_ioctls.exchange(0, std::memory_order_relaxed);
	stats.pages         = g_pages.exchange(0, std::memory_order_relaxed);
	stats.errors        = g_errors.exchange(0, std::memory_order_relaxed);
	stats.ns            = g_ns.exchange(0, std::memory_order_relaxed);
	stats.label_checks  = g_label_checks.exchange(0, std::memory_order_relaxed);
	stats.label_written = g_label_written.exchange(0, std::memory_order_relaxed);
	return stats;
}

} // namespace Libs::Graphics
