#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HAZARDSTATE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HAZARDSTATE_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdlib>

namespace Libs::Graphics {

// KYTY_LAZY_BARRIERS: 0 (unset/0) = unconditional barriers exactly as before; 1 = hazard-tracked
// lazy barriers; 2 = tracked lazily AND the old unconditional barriers are still emitted (the
// tracker's decisions are counted but the result cannot depend on them).
// Read on first use (not at static init) so a test may setenv() before building its harness.
inline uint32_t LazyBarrierMode() noexcept {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_LAZY_BARRIERS");
		if (value == nullptr || value[0] == '\0' || value[0] == '0') return 0u;
		return value[0] == '2' ? 2u : 1u;
	}();
	return mode;
}
inline bool LazyBarriersActive() noexcept { return LazyBarrierMode() != 0; }
// True when the pre-tracker barriers are still recorded (mode 0 and the mode 2 cross-check).
inline bool LazyBarriersKeepOld() noexcept { return LazyBarrierMode() != 1; }
// KYTY_STRICT_POST_DRAW_BARRIER=1: bisection switch, keeps EndRendering + ShaderWriteBarrier after
// every draw with shader writes even when the tracker is on.
inline bool StrictPostDrawBarrier() noexcept {
	static const bool strict = [] {
		const char* value = std::getenv("KYTY_STRICT_POST_DRAW_BARRIER");
		return value != nullptr && value[0] == '1';
	}();
	return strict;
}

// Shader/transfer writes and reads recorded into the current command buffer whose barrier has not
// been emitted yet. One per recording command buffer (CommandBuffer::m_hazards, reset by
// CommandBuffer::Begin()). A hazard is satisfied by a barrier with source stages covering it and
// destination eAllCommands / memory read+write, so one emission clears every field.
struct PendingHazardState {
	vk::PipelineStageFlags written_stages {};        // shader stages that wrote storage/BDA memory
	vk::PipelineStageFlags read_stages {};           // stages that read since the last write barrier
	vk::PipelineStageFlags indirect_write_stages {}; // writers whose data may feed indirect args
	vk::AccessFlags        written_access {};
	bool                   transfer_wrote = false;   // upload/fill since the last barrier

	// Pending state that a consumer other than a lazy-aware dispatch must see resolved: shader
	// writes (RAW), compute reads (WAR for anything that may write later), indirect writers.
	// Graphics reads and the transfer flag are covered by the barriers of their own consumers.
	[[nodiscard]] bool NeedsExternalFlush() const noexcept {
		return written_stages || indirect_write_stages ||
		       (read_stages & vk::PipelineStageFlagBits::eComputeShader);
	}
	void Clear() noexcept { *this = {}; }
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HAZARDSTATE_H_
