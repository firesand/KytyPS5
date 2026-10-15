#include "graphics/host_gpu/renderer/cache/colorClearResolver.h"
#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include "gpu_tiler_shaders/color_clear_check_spv.h"
#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdlib>

namespace Libs::Graphics {

namespace {

constexpr uint64_t FlagRingSize = 1u << 20; // per-resolve 32 B * slices, GPU-internal ring

struct CheckPush {
	uint32_t slice_dwords;
	uint32_t slice_count;
	uint32_t valid_mask;
	uint32_t padding;
};

} // namespace

bool ColorClearResolver::Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_COLOR_CLEAR_RESOLVE");
		if (value != nullptr && value[0] == '0' && value[1] == '\0') {
			return false;
		}
		const bool available = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBeginConditionalRenderingEXT != nullptr &&
		                       VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdEndConditionalRenderingEXT != nullptr;
		Log::WriteToConsoleAndLog(available
		                              ? "GPU: color fast clears resolve on the GPU (conditional rendering).\n"
		                              : "GPU: VK_EXT_conditional_rendering unavailable, color fast clears "
		                                "read metadata back on the CPU.\n");
		return available;
	}();
	return enabled;
}

ColorClearResolver::ColorClearResolver(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler),
      m_flags(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
              vk::BufferUsageFlagBits::eStorageBuffer |
                  vk::BufferUsageFlagBits::eConditionalRenderingEXT |
                  vk::BufferUsageFlagBits::eTransferDst,
              FlagRingSize) {
	SetVulkanObjectNameF(m_graphics.device, m_flags.Handle(), "Color Clear Flags");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                                 &m_descriptor_layout),
	                     "create color-clear descriptor layout");

	const auto module = CompileSPV(COLOR_CLEAR_CHECK_SPV, m_graphics.device);

	vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0, sizeof(CheckPush)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                            &m_pipeline_layout),
	                     "create color-clear pipeline layout");

	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create color-clear check pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Color Clear Check");
}

ColorClearResolver::~ColorClearResolver() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
}

void ColorClearResolver::Resolve(CommandBuffer& command, const Request& request) {
	EXIT_IF(request.slice_count == 0 || request.slice_bytes == 0 || (request.slice_bytes & 3u) != 0 ||
	        request.meta_buffer == nullptr || request.attachment == nullptr);
	const auto cmd        = command.Handle();
	const auto flag_bytes = uint64_t {request.slice_count} * CodeSlots * sizeof(uint32_t);
	EXIT_IF(flag_bytes > FlagRingSize);
	if (m_flags_cursor + flag_bytes > FlagRingSize) {
		m_flags_cursor = 0;
	}
	const auto flags_offset = m_flags_cursor;
	m_flags_cursor          = Common::AlignUp(m_flags_cursor + flag_bytes, 256);
	const auto meta_bytes   = uint64_t {request.slice_bytes} * request.slice_count;

	// Order earlier metadata writes and earlier predicate reads of the ring slot before the check.
	{
		vk::BufferMemoryBarrier2 barriers[2] {};
		barriers[0].srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		barriers[0].srcAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead;
		barriers[0].dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
		barriers[0].dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
		barriers[0].buffer        = request.meta_buffer;
		barriers[0].offset        = request.meta_offset;
		barriers[0].size          = meta_bytes;
		barriers[1]               = barriers[0];
		barriers[1].srcStageMask  = vk::PipelineStageFlagBits2::eConditionalRenderingEXT |
		                           vk::PipelineStageFlagBits2::eComputeShader;
		barriers[1].srcAccessMask = vk::AccessFlagBits2::eConditionalRenderingReadEXT |
		                            vk::AccessFlagBits2::eShaderWrite;
		barriers[1].buffer = m_flags.Handle();
		barriers[1].offset = flags_offset;
		barriers[1].size   = flag_bytes;
		vk::DependencyInfo dependency {};
		dependency.bufferMemoryBarrierCount = 2;
		dependency.pBufferMemoryBarriers    = barriers;
		{ GpuTimestampScope ts_scope(cmd, GpuTimestamps::Kind::BarrierBuffer); cmd.pipelineBarrier2(dependency); }
	}

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	const vk::DescriptorBufferInfo infos[] {
	    {request.meta_buffer, request.meta_offset, meta_bytes},
	    {m_flags.Handle(), flags_offset, flag_bytes},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	const CheckPush push {request.slice_bytes / 4u, request.slice_count, request.valid_mask, 0u};
	cmd.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	{ GpuTimestampScope ts_scope(cmd, GpuTimestamps::Kind::CacheDispatchColorClear); cmd.dispatch(request.slice_count, 1, 1); }

	// Flags feed the predicates; the expanded metadata is visible to everything after.
	{
		vk::BufferMemoryBarrier2 barriers[2] {};
		barriers[0].srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
		barriers[0].srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
		barriers[0].dstStageMask  = vk::PipelineStageFlagBits2::eConditionalRenderingEXT;
		barriers[0].dstAccessMask = vk::AccessFlagBits2::eConditionalRenderingReadEXT;
		barriers[0].buffer        = m_flags.Handle();
		barriers[0].offset        = flags_offset;
		barriers[0].size          = flag_bytes;
		barriers[1]               = barriers[0];
		barriers[1].dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		barriers[1].dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		barriers[1].buffer        = request.meta_buffer;
		barriers[1].offset        = request.meta_offset;
		barriers[1].size          = meta_bytes;
		vk::DependencyInfo dependency {};
		dependency.bufferMemoryBarrierCount = 2;
		dependency.pBufferMemoryBarriers    = barriers;
		{ GpuTimestampScope ts_scope(cmd, GpuTimestamps::Kind::BarrierBuffer); cmd.pipelineBarrier2(dependency); }
	}

	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = request.attachment;
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = request.extent;
	rendering.layerCount           = request.slice_count;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	GpuTimestampScope clear_scope(cmd, GpuTimestamps::Kind::Clear);
	cmd.beginRendering(&rendering);
	for (uint32_t slice = 0; slice < request.slice_count; ++slice) {
		for (uint32_t code = 0; code < CodeSlots; ++code) {
			if ((request.valid_mask & (1u << code)) == 0) {
				continue;
			}
			vk::ConditionalRenderingBeginInfoEXT begin {};
			begin.buffer = m_flags.Handle();
			begin.offset = flags_offset + (uint64_t {slice} * CodeSlots + code) * sizeof(uint32_t);
			cmd.beginConditionalRenderingEXT(&begin);
			vk::ClearAttachment clear {};
			clear.aspectMask      = vk::ImageAspectFlagBits::eColor;
			clear.colorAttachment = 0;
			clear.clearValue.color = request.colors[code];
			vk::ClearRect rect {};
			rect.rect.extent    = request.extent;
			rect.baseArrayLayer = slice;
			rect.layerCount     = 1;
			cmd.clearAttachments(1, &clear, 1, &rect);
			cmd.endConditionalRenderingEXT();
		}
	}
	cmd.endRendering();
}

} // namespace Libs::Graphics
