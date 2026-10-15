#include "graphics/host_gpu/renderer/cache/meshIndirectTranslator.h"
#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include "gpu_tiler_shaders/mesh_indirect_translate_spv.h"
#include "common/alignment.h"
#include "common/assert.h"
#include <array>
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/vulkanCommon.h"

namespace Libs::Graphics {

namespace {

struct TranslatePush {
	uint32_t args_dword;
	uint32_t stride_dwords;
	uint32_t count;
	uint32_t index_buffer_size;
	uint32_t index_base_lo;
	uint32_t index_base_hi;
	uint32_t element_size;
	uint32_t prim_size;
	uint32_t prim_step;
	uint32_t prims_per_group;
	uint32_t max_groups_x;
	uint32_t max_groups_y;
	uint32_t max_groups_total;
	uint32_t command_dword;
	uint32_t record_dword;
	uint32_t padding;
};
static_assert(sizeof(TranslatePush) == 64);

} // namespace

MeshIndirectTranslator::MeshIndirectTranslator(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler),
      m_scratch(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
                    vk::BufferUsageFlagBits::eShaderDeviceAddress,
                RingSize) {
	SetVulkanObjectNameF(m_graphics.device, m_scratch.Handle(), "Mesh Indirect Scratch");

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
	                     "create mesh-indirect descriptor layout");

	const auto module = CompileSPV(MESH_INDIRECT_TRANSLATE_SPV, m_graphics.device);

	vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0, sizeof(TranslatePush)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                            &m_pipeline_layout),
	                     "create mesh-indirect pipeline layout");

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
	RequireVulkanSuccess(result, "create mesh-indirect translate pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Mesh Indirect Translate");
}

MeshIndirectTranslator::~MeshIndirectTranslator() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
}

MeshIndirectTranslator::Result MeshIndirectTranslator::Translate(vk::CommandBuffer cmd,
                                                                 const Request& request) {
	EXIT_IF(!Fits(request.count) || request.stride < 20u || (request.stride & 3u) != 0 ||
	        (request.args_offset & 3u) != 0 || request.index_buffer_size == 0 ||
	        request.primitives_per_group == 0 || request.primitive_step == 0 ||
	        request.args_buffer == nullptr);
	const auto bytes = ScratchBytes(request.count);
	const bool wrapped = NeedsWrapBarrier(m_cursor, bytes);
	if (wrapped) {
		m_cursor = 0;
	}
	const auto base          = m_cursor;
	m_cursor                 = Common::AlignUp(m_cursor + bytes, 256);
	const auto command_bytes = Common::AlignUp(uint64_t {request.count} * CommandBytes, 16);

	// Order the producers of the argument records (when GPU-written) before the translate pass, and,
	// only when the ring wrapped, every earlier reader of the reused slots (indirect command reads,
	// mesh shader record loads, the previous translate pass) before it writes. Regions are handed
	// out sequentially, so without a wrap a pass never overlaps an earlier pass's output.
	{
		vk::MemoryBarrier2 barriers[2] {};
		uint32_t           count = 0;
		if (request.producer_barrier) {
			barriers[count].srcStageMask  = vk::PipelineStageFlagBits2::eAllGraphics |
			                               vk::PipelineStageFlagBits2::eComputeShader |
			                               vk::PipelineStageFlagBits2::eAllTransfer;
			barriers[count].srcAccessMask = vk::AccessFlagBits2::eShaderWrite |
			                                vk::AccessFlagBits2::eTransferWrite;
			barriers[count].dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
			barriers[count].dstAccessMask = vk::AccessFlagBits2::eShaderRead;
			count++;
		}
		if (wrapped) {
			barriers[count].srcStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect |
			                               vk::PipelineStageFlagBits2::eMeshShaderEXT |
			                               vk::PipelineStageFlagBits2::eComputeShader;
			barriers[count].srcAccessMask = vk::AccessFlagBits2::eIndirectCommandRead |
			                                vk::AccessFlagBits2::eShaderRead |
			                                vk::AccessFlagBits2::eShaderWrite;
			barriers[count].dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
			barriers[count].dstAccessMask = vk::AccessFlagBits2::eShaderRead |
			                                vk::AccessFlagBits2::eShaderWrite;
			count++;
		}
		if (count != 0) {
			vk::DependencyInfo dependency {};
			dependency.memoryBarrierCount = count;
			dependency.pMemoryBarriers    = barriers;
			GpuTimestampScope ts_scope(cmd, GpuTimestamps::Kind::BarrierBuffer);
			cmd.pipelineBarrier2(dependency);
		}
	}

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	const vk::DescriptorBufferInfo infos[] {
	    {request.args_buffer, 0, request.args_range},
	    {m_scratch.Handle(), 0, RingSize},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	const TranslatePush push {
	    .args_dword        = static_cast<uint32_t>(request.args_offset / 4u),
	    .stride_dwords     = request.stride / 4u,
	    .count             = request.count,
	    .index_buffer_size = request.index_buffer_size,
	    .index_base_lo     = static_cast<uint32_t>(request.index_base),
	    .index_base_hi     = static_cast<uint32_t>(request.index_base >> 32u),
	    .element_size      = request.element_size,
	    .prim_size         = request.primitive_size,
	    .prim_step         = request.primitive_step,
	    .prims_per_group   = request.primitives_per_group,
	    .max_groups_x      = request.max_groups_x,
	    .max_groups_y      = request.max_groups_y,
	    .max_groups_total  = request.max_groups_total,
	    .command_dword     = static_cast<uint32_t>(base / 4u),
	    .record_dword      = static_cast<uint32_t>((base + command_bytes) / 4u),
	    .padding           = 0,
	};
	cmd.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	{
		GpuTimestampScope ts_scope(cmd, GpuTimestamps::Kind::CacheDispatch);
		cmd.dispatch((request.count + 63u) / 64u, 1, 1);
	}

	// The commands feed vkCmdDrawMeshTasksIndirectEXT; the records feed the mesh shader.
	{
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
		barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
		barrier.dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect |
		                        vk::PipelineStageFlagBits2::eMeshShaderEXT;
		barrier.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead |
		                        vk::AccessFlagBits2::eShaderRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		GpuTimestampScope ts_scope(cmd, GpuTimestamps::Kind::BarrierBuffer);
		cmd.pipelineBarrier2(dependency);
	}
	return {.buffer = m_scratch.Handle(),
	        .commands_offset = base,
	        .records_address = m_scratch.BufferDeviceAddress() + base + command_bytes};
}

} // namespace Libs::Graphics
