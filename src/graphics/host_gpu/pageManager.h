#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;
	// Diagnostics: host protection changes since the last call (calls, bytes).
	static std::pair<uint64_t, uint64_t> TakeProtectStats();

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

	// Kernel write-watch (writeWatch.h): pages inside registered guest views keep write access
	// whatever their write watchers; read watchers still protect them.
	// Spans of tracker regions that hold write-watched pages, adjacent regions merged:
	// func(vaddr, size). Any thread; func runs with no PageManager lock held.
	void ForEachWriteWatchedSpan(const std::function<void(uint64_t, uint64_t)>& func) const;
	// Runs of pages in [vaddr, vaddr + size) with write watchers, the pages a CPU write would have
	// faulted on without write-watch: func(vaddr, size). Any thread; no lock held in func.
	void ForEachWriteWatchedRun(uint64_t vaddr, uint64_t size,
	                            const std::function<void(uint64_t, uint64_t)>& func) const;

	// Kernel write-watch: while a scope lives, covered pages inside its ranges (sorted, disjoint)
	// that gain their first write watcher on this thread are not re-armed run by run. The owner
	// re-protected the ranges in one pass (WriteWatch::CollectRanges, Source::Prearm) before any of
	// those pages was cleared, and copies each cleared page after the clear (BDA sync), so a write
	// before the pass is in the copy and a write after it is reported later, as a write after the
	// re-arm would be. Scopes nest; the innermost wins.
	class PrearmScope final {
	public:
		explicit PrearmScope(std::vector<std::pair<uint64_t, uint64_t>> ranges) noexcept;
		~PrearmScope();
		KYTY_CLASS_NO_COPY(PrearmScope);

	private:
		std::vector<std::pair<uint64_t, uint64_t>>        m_ranges;
		const std::vector<std::pair<uint64_t, uint64_t>>* m_previous = nullptr;
	};

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
