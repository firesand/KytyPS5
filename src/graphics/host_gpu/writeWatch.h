#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_WRITEWATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_WRITEWATCH_H_

#include "common/common.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Kernel write-watch for guest memory views (FPS Stage 1; KYTY_WRITE_WATCH=1, Linux only).
// Every view of the guest memfd is registered with one userfaultfd in async write-protect mode:
// the first CPU write to a protected page clears its write-protect bit inside the kernel, with no
// signal and no mprotect. PAGEMAP_SCAN later reports the written pages and re-protects them in
// one call. Pages of registered views therefore need no write protection; read protection
// (PROT_NONE for GPU-owned pages) stays on mprotect and coexists. The state lives in each view's
// page tables: a write through an unregistered alias (the emulator's backing view, used for GPU
// readbacks) is not reported. Prototype and measurements: tools/writewatch-proto in the workspace.
//
// Invariant the callers keep: the kernel's "written since re-protected" state of a page is only
// reset by Collect, and every Collect result reaches the memory tracker (applied or queued).
class WriteWatch final {
public:
	using RunFunc = std::function<void(uint64_t vaddr, uint64_t size)>;
	using Ranges  = std::vector<std::pair<uint64_t, uint64_t>>; // [start, end)

	// Who asked for a scan (per-source statistics).
	enum class Source : uint8_t {
		Inline,     // Collect: bindings, fault passes, single ranges
		Rearm,      // Rearm: a run gaining its first write watcher
		Full,       // CollectRanges: the span pass of a full collection
		Prearm,     // CollectRanges: the regions a BDA sync is about to upload
		Labels,     // CollectLabels: the label pass of a full collection
		LabelCheck, // WasLabelWritten's report-only scan
		Count
	};
	struct SourceStats {
		uint64_t calls  = 0;
		uint64_t ioctls = 0;
		uint64_t ns     = 0;
		uint64_t walked = 0; // bytes handed to PAGEMAP_SCAN (a re-walk after a full vector counts again)
		uint64_t ranges = 0; // merged ranges scanned (CollectRanges, CollectLabels)
		uint64_t pages  = 0; // written pages reported
	};

	struct Stats {
		uint64_t collects      = 0; // Collect and CollectRanges calls (Rearm included, labels not)
		uint64_t ioctls        = 0; // PAGEMAP_SCAN calls (every source)
		uint64_t pages         = 0; // written pages reported
		uint64_t errors        = 0; // failed scans, reported as fully written
		uint64_t ns            = 0; // time inside Collect and CollectRanges (not labels)
		uint64_t label_checks  = 0; // WasLabelWritten calls
		uint64_t label_written = 0; // ... that returned true
		uint64_t range_passes  = 0; // full collections (CollectRanges, Source::Full)
		uint64_t rearms        = 0; // Rearm calls (pages gaining their first write watcher)
		uint64_t rearm_skipped = 0; // pages that gained it inside a PageManager::PrearmScope
		std::array<SourceStats, static_cast<size_t>(Source::Count)> sources {};
		// Why WasLabelWritten answered "written" (first reason that applied).
		uint64_t label_scanning  = 0; // a label scan was running when the check began
		uint64_t label_unknown   = 0; // page not in the label set yet (or the set is full)
		uint64_t label_flagged   = 0; // a collection re-protected the page after a write
		uint64_t label_uncovered = 0; // page outside every registered view
		uint64_t label_scanned   = 0; // the report-only scan found the page written
		uint64_t label_raced     = 0; // a label scan overlapped the check
		uint64_t label_set       = 0; // label pages known (gauge, bounded)
	};

	// The process-wide instance, or nullptr when write-watch is off: KYTY_WRITE_WATCH unset, a
	// non-Linux host, or a kernel without userfaultfd WP_ASYNC / PAGEMAP_SCAN (logged once; the
	// caller then keeps today's mprotect path). Thread-safe; never destroyed.
	static WriteWatch* Get() noexcept;

	KYTY_CLASS_NO_COPY(WriteWatch);

	// A guest view was mapped (GuestBackingStore::MapView): register and write-protect it. A view
	// the kernel refuses is fatal (its pages could be watched without protection). Any thread.
	void RegisterView(uint64_t vaddr, uint64_t size) noexcept;
	// A guest view is about to be replaced by a placeholder: stop scanning it. Any thread.
	void UnregisterView(uint64_t vaddr, uint64_t size) noexcept;

	// Changes whenever the set of registered views changes.
	[[nodiscard]] uint64_t ViewsEpoch() const noexcept {
		return m_views_epoch.load(std::memory_order_acquire);
	}
	// Bit i set: page i of the tracker region at region_base lies in a registered view.
	void CoveredPages(uint64_t region_base, RegionBits* covered) const noexcept;
	[[nodiscard]] bool IsCovered(uint64_t vaddr, uint64_t size) const noexcept;

	// Reports, and re-protects in the kernel, the pages of [vaddr, vaddr + size) written since
	// their last collection, restricted to registered views: func(run_vaddr, run_size) per run of
	// whole pages (the range is widened to page bounds), called after every internal lock is
	// released. A failed scan reports its whole chunk. Any thread.
	void Collect(uint64_t vaddr, uint64_t size, const RunFunc& func) noexcept;
	// One pass over many [start, end) ranges (a full collection), same result contract as Collect.
	// Ranges closer than a merge gap are scanned as one: a gap's placeholders and unwatched pages
	// cost a short walk, a separate ioctl costs more (ng-154: about 4200 ioctls per collection
	// when every watched span had its own). The gap is KYTY_WRITE_WATCH_MERGE_GAP_MB (default 64)
	// for Source::Full and 8 MiB for Source::Prearm (4 MiB regions; the prototype puts the
	// break-even between an ioctl and a walk at about 4-10 MiB). The merged ranges cross
	// unregistered placeholders, so the scan skips unregistered VMAs instead of failing on them;
	// every registered guest view is in the registry (RegisterView/UnregisterView), so none is
	// skipped by mistake. scanned (optional) receives the merged ranges, sorted and disjoint:
	// every page of them was re-protected by this pass (left empty when a scan failed).
	void CollectRanges(Ranges ranges, const RunFunc& func, Source source = Source::Full,
	                   Ranges* scanned = nullptr) noexcept;
	// Re-protects [vaddr, vaddr + size) and drops what it reports: for pages gaining their first
	// write watcher, whose earlier writes nobody tracked (PageManager; the protect a fault-based
	// watcher would have applied). Label pages it re-protects are still flagged. Any thread.
	void Rearm(uint64_t vaddr, uint64_t size) noexcept;

	// Acquire points (WAIT_REG_MEM, COND_EXEC, ...). A label page is re-protected by every Collect
	// that covers it and by CollectLabels; WasLabelWritten(page) is then true iff the page was
	// written after that collection, or the page is unknown or uncovered (conservative). Pages
	// seen for the first time join the label set (bounded). Any thread.
	[[nodiscard]] bool WasLabelWritten(uint64_t vaddr, uint64_t size) noexcept;
	// Collects every known label page (call first in a full collection; same result contract).
	// Label pages closer than 8 MiB are scanned as one range, so the pass may also report other
	// written pages; the caller applies them like any collected run.
	void CollectLabels(const RunFunc& func) noexcept;

	// PageManager: pages that gained their first write watcher inside a PrearmScope, not re-armed.
	static void NoteRearmSkipped(uint64_t pages) noexcept;

	static Stats TakeStats() noexcept;

private:
	WriteWatch(int uffd, int pagemap) noexcept;
	~WriteWatch() = default;

	static WriteWatch* Create() noexcept;

	void CollectFrom(uint64_t vaddr, uint64_t size, const RunFunc& func, Source source) noexcept;
	// One PAGEMAP_SCAN pass over [start, end); appends written runs. check_async: fail on an
	// unregistered VMA (one registered view) instead of skipping it (a merged range).
	// Returns false when a call failed (its rest was reported as written, not re-protected).
	bool ScanChunk(uint64_t start, uint64_t end, Ranges* runs, bool check_async, Source source) noexcept;
	// The registered parts of the merged ranges, in chunks of at most chunk bytes, check_async off.
	// Returns false when a scan failed.
	bool ScanMerged(const Ranges& merged, uint64_t chunk, Ranges* runs, Source source) noexcept;
	// Report-only: any written page in [start, end) of a registered view?
	[[nodiscard]] bool AnyWritten(uint64_t start, uint64_t end) noexcept;
	void NoteLabelRuns(const std::vector<std::pair<uint64_t, uint64_t>>& runs) noexcept;
	[[nodiscard]] bool TouchesLabels(uint64_t vaddr, uint64_t size) noexcept;

	int m_uffd    = -1;
	int m_pagemap = -1;

	mutable std::shared_mutex m_views_mutex;
	RangeSet                  m_views;
	std::atomic<uint64_t>     m_views_epoch {1};

	struct Label {
		uint64_t page    = 0;
		bool     written = false; // a Collect re-protected the page after a write to it
	};
	std::mutex         m_labels_mutex;
	std::vector<Label> m_labels; // sorted by page
	std::atomic<bool>  m_has_labels {false};
	// Seqlock between scans that re-protect label pages and WasLabelWritten: a scan re-protects a
	// label page before it can flag it, so a check overlapping such a scan answers "written".
	std::atomic<uint32_t> m_label_scans_active {0};
	std::atomic<uint64_t> m_label_scan_seq {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_WRITEWATCH_H_
