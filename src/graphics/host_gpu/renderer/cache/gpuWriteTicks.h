#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUWRITETICKS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUWRITETICKS_H_

#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

// Per-page "last tracked GPU write tick" stamps, paged array instead of an unordered_map
// (NoteGpuWrite was 2 % of the GPU thread's samples). Semantics are those of the map it replaces:
// a page is either unstamped (Find == nullptr) or stamped with a monotonically non-decreasing tick
// (0 = stamped, completion unknown/complete). Stamps are never erased. Pages outside the 44-bit
// guest address space cannot be stored; callers must treat them conservatively (see Note()).
template <uint32_t PageBits>
class GpuWriteTicks final {
public:
	struct Stamp {
		uint64_t tick    = 0;
		bool     present = false;
	};
	using Table = MultiLevelPageTable<Stamp, PageBits, 44, 20>;

	// Returns false (and stores nothing) when the page is outside the representable range.
	bool Note(uint64_t page, uint64_t tick) {
		if (!Table::IsValidPage(page)) return false;
		auto& stamp   = m_table.GetOrCreate(static_cast<size_t>(page));
		stamp.present = true;
		stamp.tick    = std::max(stamp.tick, tick);
		return true;
	}
	// nullptr: no stamp for this page.
	[[nodiscard]] const uint64_t* Find(uint64_t page) const {
		if (!Table::IsValidPage(page)) return nullptr;
		const auto* stamp = m_table.Find(static_cast<size_t>(page));
		return stamp != nullptr && stamp->present ? &stamp->tick : nullptr;
	}
	[[nodiscard]] bool     Contains(uint64_t page) const { return Find(page) != nullptr; }
	[[nodiscard]] uint64_t Get(uint64_t page) const {
		const auto* tick = Find(page);
		return tick != nullptr ? *tick : 0;
	}

private:
	Table m_table;
};

} // namespace Libs::Graphics

#endif
