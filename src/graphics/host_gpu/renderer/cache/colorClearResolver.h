#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COLORCLEARRESOLVER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COLORCLEARRESOLVER_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

class CommandBuffer;
class CommandScheduler;
struct GraphicContext;

// Resolves native color fast clears on the GPU. The guest clears a DCC/CMask surface by writing a
// clear code into its metadata; when that metadata was written by the GPU, reading it back on the
// CPU drains the whole pipeline (dozens of times per frame in UE5 titles). Instead a compute pass
// checks every metadata slice for a uniform, valid code and writes per-(slice, code) flags, and
// VK_EXT_conditional_rendering predicates one vkCmdClearAttachments per candidate code on them.
class ColorClearResolver {
public:
	static constexpr uint32_t CodeSlots = 8; // code >> 5 for 0x00, 0x20, 0x40, 0x80, 0xc0

	struct Request {
		vk::Buffer                                 meta_buffer;
		uint64_t                                   meta_offset = 0;
		uint32_t                                   slice_bytes = 0;
		uint32_t                                   slice_count = 0;
		uint32_t                                   valid_mask  = 0; // bit k: code (k << 5) is a clear
		std::array<vk::ClearColorValue, CodeSlots> colors {};
		vk::ImageView                              attachment; // 2D array view over the slices
		vk::Extent2D                               extent;
	};

	ColorClearResolver(GraphicContext& graphics, CommandScheduler& scheduler);
	~ColorClearResolver();
	KYTY_CLASS_NO_COPY(ColorClearResolver);

	// KYTY_GPU_COLOR_CLEAR_RESOLVE=0 disables; also requires VK_EXT_conditional_rendering.
	[[nodiscard]] static bool Enabled();

	void Resolve(CommandBuffer& command, const Request& request);

private:
	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	Buffer                  m_flags;
	uint64_t                m_flags_cursor         = 0;
	vk::DescriptorSetLayout m_descriptor_layout    = nullptr;
	vk::PipelineLayout      m_pipeline_layout      = nullptr;
	vk::Pipeline            m_pipeline             = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COLORCLEARRESOLVER_H_
