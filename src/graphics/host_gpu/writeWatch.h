#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_WRITEWATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_WRITEWATCH_H_

#include "common/common.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>
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

	struct Stats {
		uint64_t collects      = 0; // Collect calls
		uint64_t ioctls        = 0; // PAGEMAP_SCAN calls (Collect and WasWritten)
		uint64_t pages         = 0; // written pages reported
		uint64_t errors        = 0; // failed scans, reported as fully written
		uint64_t ns            = 0; // time inside Collect
		uint64_t label_checks  = 0; // WasLabelWritten calls
		uint64_t label_written = 0; // ... that returned true
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
	void CollectLabels(const RunFunc& func) noexcept;

	static Stats TakeStats() noexcept;

private:
	WriteWatch(int uffd, int pagemap) noexcept;
	~WriteWatch() = default;

	static WriteWatch* Create() noexcept;

	// One PAGEMAP_SCAN pass over [start, end) (one registered view); appends written runs.
	void ScanChunk(uint64_t start, uint64_t end, std::vector<std::pair<uint64_t, uint64_t>>* runs) noexcept;
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
