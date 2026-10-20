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
#include <thread>
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
constexpr uint64_t RANGE_CHUNK = 256ull * 1024ull * 1024ull; // CollectRanges' per-ioctl span
constexpr uint64_t SMALL_MERGE_GAP = 8ull * 1024ull * 1024ull; // pre-arm and label passes
constexpr size_t   MAX_LABELS = 4096;

// KYTY_WRITE_WATCH_MERGE_GAP_MB (default 64): full collections scan watched spans closer than this
// as one range.
uint64_t FullMergeGap() {
	static const uint64_t gap = [] {
		const char* value = std::getenv("KYTY_WRITE_WATCH_MERGE_GAP_MB");
		const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 64ull;
		return std::min<uint64_t>(mib, 1ull << 20) * 1024ull * 1024ull;
	}();
	return gap;
}

std::atomic<uint64_t> g_collects {0};
std::atomic<uint64_t> g_ioctls {0};
std::atomic<uint64_t> g_pages {0};
std::atomic<uint64_t> g_errors {0};
std::atomic<uint64_t> g_ns {0};
std::atomic<uint64_t> g_label_checks {0};
std::atomic<uint64_t> g_label_written {0};
std::atomic<uint64_t> g_range_passes {0};
std::atomic<uint64_t> g_rearms {0};
std::atomic<uint64_t> g_rearm_skipped {0};
std::atomic<uint64_t> g_label_scanning {0};
std::atomic<uint64_t> g_label_unknown {0};
std::atomic<uint64_t> g_label_flagged {0};
std::atomic<uint64_t> g_label_uncovered {0};
std::atomic<uint64_t> g_label_scanned {0};
std::atomic<uint64_t> g_label_raced {0};
std::atomic<uint64_t> g_label_set {0};
std::atomic<uint64_t> g_label_evicted {0};
std::atomic<uint64_t> g_apply_waits {0};
std::atomic<uint64_t> g_apply_timeouts {0};

// ApplyScopes this thread holds (nested ones included), so OtherAppliesInFlight leaves them out.
thread_local uint32_t t_applies_in_flight = 0;

// KYTY_WRITE_WATCH_LABELS: "consumed" (default) or "set" (WriteWatch::WasLabelWritten).
bool ConsumedLabels() {
	static const bool consumed = [] {
		const char* value = std::getenv("KYTY_WRITE_WATCH_LABELS");
		return value == nullptr || std::string(value) != "set";
	}();
	return consumed;
}

struct SourceCounters {
	std::atomic<uint64_t> calls {0};
	std::atomic<uint64_t> ioctls {0};
	std::atomic<uint64_t> ns {0};
	std::atomic<uint64_t> walked {0};
	std::atomic<uint64_t> ranges {0};
	std::atomic<uint64_t> pages {0};
	std::atomic<uint64_t> requested {0};
};
std::array<SourceCounters, static_cast<size_t>(WriteWatch::Source::Count)> g_sources;

SourceCounters& Counters(WriteWatch::Source source) {
	return g_sources[static_cast<size_t>(source)];
}

uint64_t NsSince(std::chrono::steady_clock::time_point begin) {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
}

// Sorted, page-aligned [start, end) ranges merged across gaps of at most gap bytes.
WriteWatch::Ranges MergeRanges(WriteWatch::Ranges ranges, uint64_t gap) {
	for (auto& [start, end]: ranges) {
		start = Common::AlignDown(start, PAGE_SIZE);
		end   = Common::AlignUp(end, PAGE_SIZE);
	}
	std::sort(ranges.begin(), ranges.end());
	WriteWatch::Ranges merged;
	for (const auto& [start, end]: ranges) {
		if (!merged.empty() && start <= merged.back().second + gap) {
			merged.back().second = std::max(merged.back().second, end);
		} else {
			merged.emplace_back(start, end);
		}
	}
	return merged;
}

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
	// Seqlock as for a scan (entered once the lock is held, so a check made while this waits for a
	// collection is not counted as overlapping): a check cannot fall between the remap or arming
	// below and the record after it.
	m_label_scans_active.fetch_add(1, std::memory_order_seq_cst);
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
	lock.unlock();
	// Arming re-protects without reporting, and a part mapped over a live view (MapFixed maps over
	// it without UnregisterView) is a new VMA that is not armed at all: freed page tables there read
	// as not written. A label anywhere in the view counts as written until the next label pass.
	NoteLabelRuns(Ranges {{vaddr, vaddr + size}});
	m_label_scan_seq.fetch_add(1, std::memory_order_seq_cst);
	m_label_scans_active.fetch_sub(1, std::memory_order_seq_cst);
#else
	(void)vaddr;
#endif
}

void WriteWatch::UnregisterView(uint64_t vaddr, uint64_t size) noexcept {
	if (size == 0) {
		return;
	}
	std::unique_lock lock(m_views_mutex);
	// Seqlock: a label check that passed IsCovered before this must not trust a scan after it.
	m_label_scans_active.fetch_add(1, std::memory_order_seq_cst);
	// The placeholder mapping that replaces the view drops the kernel registration itself.
	m_views.Subtract(vaddr, size);
	m_views_epoch.fetch_add(1, std::memory_order_acq_rel);
	m_label_scan_seq.fetch_add(1, std::memory_order_seq_cst);
	m_label_scans_active.fetch_sub(1, std::memory_order_seq_cst);
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

bool WriteWatch::ScanChunk(uint64_t start, uint64_t end, Ranges* runs, bool check_async,
                           Source source) noexcept {
#if defined(KYTY_WRITE_WATCH_SUPPORTED)
	static thread_local std::array<page_region, 512> vec {};
	auto&    counters = Counters(source);
	uint64_t cursor   = start;
	while (cursor < end) {
		pm_scan_arg arg {};
		arg.size          = sizeof(arg);
		arg.flags         = PM_SCAN_WP_MATCHING | (check_async ? PM_SCAN_CHECK_WPASYNC : 0);
		arg.start         = cursor;
		arg.end           = end;
		arg.vec           = reinterpret_cast<uint64_t>(vec.data());
		arg.vec_len       = vec.size();
		arg.category_mask = PAGE_IS_WRITTEN;
		arg.return_mask   = PAGE_IS_WRITTEN;
		const long count  = ::ioctl(m_pagemap, PAGEMAP_SCAN, &arg);
		g_ioctls.fetch_add(1, std::memory_order_relaxed);
		counters.ioctls.fetch_add(1, std::memory_order_relaxed);
		counters.walked.fetch_add(end - cursor, std::memory_order_relaxed);
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
			return false;
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
	(void)check_async;
	(void)source;
#endif
	return true;
}

bool WriteWatch::ScanMerged(const Ranges& merged, uint64_t chunk, Ranges* runs, Source source) noexcept {
	bool ok = true;
	for (const auto& [start, end]: merged) {
		// The registered extent of this range; the kernel skips unregistered VMAs inside it.
		uint64_t low  = UINT64_MAX;
		uint64_t high = 0;
		m_views.ForEachInRange(start, end - start, [&](uint64_t view_start, uint64_t view_end) {
			low  = std::min(low, view_start);
			high = std::max(high, view_end);
		});
		for (auto cursor = low; cursor < high;) {
			const auto chunk_end = std::min(high, Common::AlignDown(cursor, chunk) + chunk);
			ok = ScanChunk(cursor, chunk_end, runs, /*check_async=*/false, source) && ok;
			cursor = chunk_end;
		}
	}
	return ok;
}

bool WriteWatch::AnyWritten(uint64_t start, uint64_t end) noexcept {
#if defined(KYTY_WRITE_WATCH_SUPPORTED)
	page_region region {};
	pm_scan_arg arg {};
	arg.size          = sizeof(arg);
	arg.flags         = PM_SCAN_CHECK_WPASYNC; // an unregistered VMA fails (EPERM): "written"
	arg.start         = start;
	arg.end           = end;
	arg.vec           = reinterpret_cast<uint64_t>(&region);
	arg.vec_len       = 1;
	arg.max_pages     = 1;
	arg.category_mask = PAGE_IS_WRITTEN;
	arg.return_mask   = PAGE_IS_WRITTEN;
	const long count  = ::ioctl(m_pagemap, PAGEMAP_SCAN, &arg);
	g_ioctls.fetch_add(1, std::memory_order_relaxed);
	Counters(Source::LabelCheck).ioctls.fetch_add(1, std::memory_order_relaxed);
	Counters(Source::LabelCheck).walked.fetch_add(end - start, std::memory_order_relaxed);
	return count != 0; // an error is "written"
#else
	(void)start;
	(void)end;
	return true;
#endif
}

void WriteWatch::NoteLabelRuns(const std::vector<std::pair<uint64_t, uint64_t>>& runs) noexcept {
	if (runs.empty()) {
		return;
	}
	if (ConsumedLabels()) {
		std::lock_guard lock(m_labels_mutex);
		for (const auto& [start, end]: runs) {
			m_consumed.Add(start, end - start);
		}
		return;
	}
	if (!m_has_labels.load(std::memory_order_acquire)) {
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
	if (ConsumedLabels()) {
		return true; // any page can be a label
	}
	if (!m_has_labels.load(std::memory_order_acquire)) {
		return false;
	}
	std::lock_guard lock(m_labels_mutex);
	const auto      it = std::lower_bound(m_labels.begin(), m_labels.end(), Common::AlignDown(vaddr, PAGE_SIZE),
	                                      [](const Label& label, uint64_t page) { return label.page < page; });
	return it != m_labels.end() && it->page < vaddr + size;
}

void WriteWatch::Collect(uint64_t vaddr, uint64_t size, const RunFunc& func) noexcept {
	CollectFrom(vaddr, size, func, Source::Inline);
}

void WriteWatch::CollectFrom(uint64_t vaddr, uint64_t size, const RunFunc& func, Source source) noexcept {
	if (size == 0) {
		return;
	}
	ApplyScope applying(this);
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
				(void)ScanChunk(chunk, chunk_end, &runs, /*check_async=*/true, source);
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
	const auto ns = NsSince(begin);
	g_ns.fetch_add(ns, std::memory_order_relaxed);
	auto& counters = Counters(source);
	counters.calls.fetch_add(1, std::memory_order_relaxed);
	counters.ns.fetch_add(ns, std::memory_order_relaxed);
	// The callback may re-enter Collect (nested trackers); work on a copy.
	const auto reported = runs;
	for (const auto& [start, end]: reported) {
		g_pages.fetch_add((end - start) / PAGE_SIZE, std::memory_order_relaxed);
		counters.pages.fetch_add((end - start) / PAGE_SIZE, std::memory_order_relaxed);
		func(start, end - start);
	}
}

void WriteWatch::Rearm(uint64_t vaddr, uint64_t size) noexcept {
	g_rearms.fetch_add(1, std::memory_order_relaxed);
	CollectFrom(vaddr, size, [](uint64_t, uint64_t) {}, Source::Rearm);
}

void WriteWatch::NoteRearmSkipped(uint64_t pages) noexcept {
	g_rearm_skipped.fetch_add(pages, std::memory_order_relaxed);
}

WriteWatch::ApplyScope::ApplyScope(WriteWatch* write_watch) noexcept: m_write_watch(write_watch) {
	if (m_write_watch == nullptr) {
		return;
	}
	m_bucket = m_write_watch->m_apply_epoch.load(std::memory_order_seq_cst) & 1u;
	m_write_watch->m_applies_in_flight[m_bucket].fetch_add(1, std::memory_order_seq_cst);
	++t_applies_in_flight;
}

WriteWatch::ApplyScope::~ApplyScope() {
	if (m_write_watch == nullptr) {
		return;
	}
	--t_applies_in_flight;
	m_write_watch->m_applies_in_flight[m_bucket].fetch_sub(1, std::memory_order_seq_cst);
}

bool WriteWatch::OtherAppliesInFlight() const noexcept {
	return m_applies_in_flight[0].load(std::memory_order_seq_cst) +
	           m_applies_in_flight[1].load(std::memory_order_seq_cst) >
	       t_applies_in_flight;
}

bool WriteWatch::WaitForOtherApplies() noexcept {
	if (t_applies_in_flight != 0) {
		return !OtherAppliesInFlight(); // cannot wait for a counter that holds this thread's scope
	}
	// A scope that reads the old bucket after this flip increments it before its scan: if this
	// wait missed it, its scan comes after the caller's clear and records what it consumes.
	const uint32_t old_bucket = m_apply_epoch.fetch_add(1, std::memory_order_seq_cst) & 1u;
	if (m_applies_in_flight[old_bucket].load(std::memory_order_seq_cst) == 0) {
		return true;
	}
	g_apply_waits.fetch_add(1, std::memory_order_relaxed);
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
	while (m_applies_in_flight[old_bucket].load(std::memory_order_seq_cst) != 0) {
		if (std::chrono::steady_clock::now() > deadline) {
			g_apply_timeouts.fetch_add(1, std::memory_order_relaxed);
			static std::atomic<uint32_t> reported {0};
			if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
				LOGF("WriteWatch: another thread's collection still applying after 50 ms\n");
			}
			return false;
		}
		std::this_thread::yield();
	}
	return true;
}

void WriteWatch::CollectRanges(Ranges ranges, const RunFunc& func, Source source, Ranges* scanned) noexcept {
	if (scanned != nullptr) {
		scanned->clear();
	}
	if (ranges.empty()) {
		return;
	}
	ApplyScope applying(this);
	uint64_t   requested = 0;
	for (const auto& [start, end]: ranges) {
		requested += end - start;
	}
	Counters(source).requested.fetch_add(requested, std::memory_order_relaxed);
	const auto merged =
	    MergeRanges(std::move(ranges), source == Source::Prearm ? SMALL_MERGE_GAP : FullMergeGap());
	const auto begin = std::chrono::steady_clock::now();
	g_collects.fetch_add(1, std::memory_order_relaxed);
	if (source == Source::Full) {
		g_range_passes.fetch_add(1, std::memory_order_relaxed);
	}
	bool labels = false;
	for (const auto& [start, end]: merged) {
		labels = labels || TouchesLabels(start, end - start);
	}
	if (labels) {
		m_label_scans_active.fetch_add(1, std::memory_order_seq_cst);
	}
	Ranges runs;
	bool   ok = true;
	{
		std::shared_lock lock(m_views_mutex);
		ok = ScanMerged(merged, RANGE_CHUNK, &runs, source);
	}
	NoteLabelRuns(runs);
	LogRuns(source == Source::Prearm ? "prearm" : "collect-ranges", merged.front().first,
	        merged.back().second - merged.front().first, runs);
	if (labels) {
		m_label_scan_seq.fetch_add(1, std::memory_order_seq_cst);
		m_label_scans_active.fetch_sub(1, std::memory_order_seq_cst);
	}
	const auto ns = NsSince(begin);
	g_ns.fetch_add(ns, std::memory_order_relaxed);
	auto& counters = Counters(source);
	counters.calls.fetch_add(1, std::memory_order_relaxed);
	counters.ns.fetch_add(ns, std::memory_order_relaxed);
	counters.ranges.fetch_add(merged.size(), std::memory_order_relaxed);
	if (scanned != nullptr && ok) {
		*scanned = merged;
	}
	for (const auto& [start, end]: runs) {
		g_pages.fetch_add((end - start) / PAGE_SIZE, std::memory_order_relaxed);
		counters.pages.fetch_add((end - start) / PAGE_SIZE, std::memory_order_relaxed);
		func(start, end - start);
	}
}

bool WriteWatch::WasLabelWritten(uint64_t vaddr, uint64_t size) noexcept {
	const auto begin = std::chrono::steady_clock::now();
	g_label_checks.fetch_add(1, std::memory_order_relaxed);
	// reason: the counter of the first cause that made the answer "written", or null.
	const auto finish = [&](std::atomic<uint64_t>* reason) {
		auto& counters = Counters(Source::LabelCheck);
		counters.calls.fetch_add(1, std::memory_order_relaxed);
		counters.ns.fetch_add(NsSince(begin), std::memory_order_relaxed);
		if (reason == nullptr) {
			return false;
		}
		reason->fetch_add(1, std::memory_order_relaxed);
		g_label_written.fetch_add(1, std::memory_order_relaxed);
		return true;
	};
	const auto seq_before = m_label_scan_seq.load(std::memory_order_seq_cst);
	if (m_label_scans_active.load(std::memory_order_seq_cst) != 0) {
		return finish(&g_label_scanning);
	}
	const auto first = Common::AlignDown(vaddr, PAGE_SIZE);
	const auto last  = Common::AlignUp(vaddr + std::max<uint64_t>(size, 1), PAGE_SIZE);
	const bool consumed_mode = ConsumedLabels();
	bool       known         = true;
	bool       flagged       = false;
	{
		std::lock_guard lock(m_labels_mutex);
		const auto      clock = ++m_label_clock;
		const auto      find  = [&](uint64_t page) {
			return std::lower_bound(m_labels.begin(), m_labels.end(), page,
			                        [](const Label& label, uint64_t p) { return label.page < p; });
		};
		for (auto page = first; page < last; page += PAGE_SIZE) {
			auto it = find(page);
			if (it == m_labels.end() || it->page != page) {
				known = false;
				if (consumed_mode && m_labels.size() >= MAX_LABELS) {
					// The set only picks the pages the label pass re-protects first here.
					m_labels.erase(std::min_element(m_labels.begin(), m_labels.end(),
					                                [](const Label& a, const Label& b) { return a.seen < b.seen; }));
					g_label_evicted.fetch_add(1, std::memory_order_relaxed);
					it = find(page);
				}
				if (m_labels.size() < MAX_LABELS) {
					m_labels.insert(it, Label {page, false, clock});
					m_has_labels.store(true, std::memory_order_release);
					g_label_set.store(m_labels.size(), std::memory_order_relaxed);
				}
			} else {
				it->seen = clock;
				flagged  = flagged || it->written;
			}
		}
		if (consumed_mode) {
			// A scan that re-protected the page since the last label pass may have consumed a
			// label write; the kernel bit covers the rest, whether or not the page was known.
			known   = true;
			flagged = m_consumed.Intersects(first, last - first);
		}
	}
	// Unknown pages were never re-protected for this purpose, and uncovered pages cannot be
	// checked: both mean "possibly written".
	if (!known) {
		return finish(&g_label_unknown);
	}
	if (flagged) {
		return finish(&g_label_flagged);
	}
	if (!IsCovered(first, last - first)) {
		return finish(&g_label_uncovered);
	}
	if (AnyWritten(first, last)) {
		return finish(&g_label_scanned);
	}
	// A scan that touched label pages overlapped this check: its re-protect may have hidden the
	// write and its flag may have come too late.
	if (m_label_scans_active.load(std::memory_order_seq_cst) != 0 ||
	    m_label_scan_seq.load(std::memory_order_seq_cst) != seq_before) {
		return finish(&g_label_raced);
	}
	return finish(nullptr);
}

void WriteWatch::CollectLabels(const RunFunc& func) noexcept {
	// "consumed" mode starts a new label epoch even with no label known yet.
	if (!m_has_labels.load(std::memory_order_acquire) && !ConsumedLabels()) {
		return;
	}
	ApplyScope applying(this);
	std::vector<uint64_t> pages;
	{
		std::lock_guard lock(m_labels_mutex);
		pages.reserve(m_labels.size());
		for (const auto& label: m_labels) {
			pages.push_back(label.page);
		}
	}
	// Label pages closer than 8 MiB are scanned as one range (ng-168: about 2,500 single-page
	// ioctls per full collection, scanned one by one). A merged range also re-protects the
	// written pages between the labels and reports them; the caller applies them like any
	// collected run, and the span pass that follows covers every write-watched page again, so
	// a data write the label pass missed is still collected by this full collection.
	const auto begin = std::chrono::steady_clock::now();
	Ranges     wanted;
	wanted.reserve(pages.size());
	for (const auto page: pages) {
		wanted.emplace_back(page, page + PAGE_SIZE);
	}
	const auto merged = MergeRanges(std::move(wanted), SMALL_MERGE_GAP);
	m_label_scans_active.fetch_add(1, std::memory_order_seq_cst);
	Ranges runs;
	{
		std::shared_lock lock(m_views_mutex);
		(void)ScanMerged(merged, SCAN_CHUNK, &runs, Source::Labels);
	}
	{
		// Cleared after the label scans and before the caller's span scans: a label written
		// after this point is either still unprotected or flagged (consumed) by the scan that
		// protects it. A label write consumed before this point preceded the span pass, which
		// collects every write-watched page written before it.
		std::lock_guard lock(m_labels_mutex);
		if (ConsumedLabels()) {
			m_consumed.Clear();
		} else {
			for (auto& label: m_labels) {
				label.written = false;
			}
		}
	}
	m_label_scan_seq.fetch_add(1, std::memory_order_seq_cst);
	m_label_scans_active.fetch_sub(1, std::memory_order_seq_cst);
	auto& counters = Counters(Source::Labels);
	counters.calls.fetch_add(1, std::memory_order_relaxed);
	counters.ns.fetch_add(NsSince(begin), std::memory_order_relaxed);
	counters.ranges.fetch_add(merged.size(), std::memory_order_relaxed);
	for (const auto& [start, end]: runs) {
		g_pages.fetch_add((end - start) / PAGE_SIZE, std::memory_order_relaxed);
		counters.pages.fetch_add((end - start) / PAGE_SIZE, std::memory_order_relaxed);
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
	stats.range_passes  = g_range_passes.exchange(0, std::memory_order_relaxed);
	stats.rearms        = g_rearms.exchange(0, std::memory_order_relaxed);
	stats.rearm_skipped = g_rearm_skipped.exchange(0, std::memory_order_relaxed);
	for (size_t i = 0; i < stats.sources.size(); i++) {
		auto& counters        = g_sources[i];
		stats.sources[i].calls  = counters.calls.exchange(0, std::memory_order_relaxed);
		stats.sources[i].ioctls = counters.ioctls.exchange(0, std::memory_order_relaxed);
		stats.sources[i].ns     = counters.ns.exchange(0, std::memory_order_relaxed);
		stats.sources[i].walked = counters.walked.exchange(0, std::memory_order_relaxed);
		stats.sources[i].ranges = counters.ranges.exchange(0, std::memory_order_relaxed);
		stats.sources[i].pages  = counters.pages.exchange(0, std::memory_order_relaxed);
		stats.sources[i].requested = counters.requested.exchange(0, std::memory_order_relaxed);
	}
	stats.label_scanning  = g_label_scanning.exchange(0, std::memory_order_relaxed);
	stats.label_unknown   = g_label_unknown.exchange(0, std::memory_order_relaxed);
	stats.label_flagged   = g_label_flagged.exchange(0, std::memory_order_relaxed);
	stats.label_uncovered = g_label_uncovered.exchange(0, std::memory_order_relaxed);
	stats.label_scanned   = g_label_scanned.exchange(0, std::memory_order_relaxed);
	stats.label_raced     = g_label_raced.exchange(0, std::memory_order_relaxed);
	stats.label_set       = g_label_set.load(std::memory_order_relaxed);
	stats.label_evicted   = g_label_evicted.exchange(0, std::memory_order_relaxed);
	stats.apply_waits     = g_apply_waits.exchange(0, std::memory_order_relaxed);
	stats.apply_timeouts  = g_apply_timeouts.exchange(0, std::memory_order_relaxed);
	stats.label_consumed  = ConsumedLabels();
	return stats;
}

} // namespace Libs::Graphics
