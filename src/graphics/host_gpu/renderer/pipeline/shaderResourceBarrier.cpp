#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include "common/assert.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderBindings.h"

#include <cstring>

namespace Libs::Graphics {

vk::ShaderStageFlagBits NativeShaderStage(ShaderType stage) {
	switch (stage) {
		case ShaderType::Local:
		case ShaderType::Vertex: return vk::ShaderStageFlagBits::eVertex;
		case ShaderType::Mesh: return vk::ShaderStageFlagBits::eMeshEXT;
		case ShaderType::TessellationControl: return vk::ShaderStageFlagBits::eTessellationControl;
		case ShaderType::TessellationEvaluation:
			return vk::ShaderStageFlagBits::eTessellationEvaluation;
		case ShaderType::Pixel: return vk::ShaderStageFlagBits::eFragment;
		case ShaderType::Compute: return vk::ShaderStageFlagBits::eCompute;
		default: EXIT("unknown native shader stage\n");
	}
}

vk::PipelineStageFlags ShaderPipelineStages(vk::ShaderStageFlags stages) {
	vk::PipelineStageFlags result = {};
	if (stages & vk::ShaderStageFlagBits::eVertex) {
		result |= vk::PipelineStageFlagBits::eVertexShader;
	}
	if (stages & vk::ShaderStageFlagBits::eMeshEXT) {
		result |= vk::PipelineStageFlagBits::eMeshShaderEXT;
	}
	if (stages & vk::ShaderStageFlagBits::eTessellationControl) {
		result |= vk::PipelineStageFlagBits::eTessellationControlShader;
	}
	if (stages & vk::ShaderStageFlagBits::eTessellationEvaluation) {
		result |= vk::PipelineStageFlagBits::eTessellationEvaluationShader;
	}
	if (stages & vk::ShaderStageFlagBits::eFragment) {
		result |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (stages & vk::ShaderStageFlagBits::eCompute) {
		result |= vk::PipelineStageFlagBits::eComputeShader;
	}
	EXIT_IF(!result);
	return result;
}

vk::MemoryBarrier MakeShaderWriteDependency() {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
	                        vk::AccessFlagBits::eVertexAttributeRead |
	                        vk::AccessFlagBits::eIndexRead | vk::AccessFlagBits::eUniformRead |
	                        vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eColorAttachmentRead |
	                        vk::AccessFlagBits::eColorAttachmentWrite;
	return barrier;
}

vk::MemoryBarrier MakeShaderAccessDependency() {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	return barrier;
}

vk::MemoryBarrier MakeShaderWriteHazardDependency() {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	return barrier;
}

vk::BufferMemoryBarrier MakeGdsDependency(vk::Buffer buffer) {
	EXIT_IF(buffer == nullptr);

	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer;
	barrier.offset              = 0;
	barrier.size                = VK_WHOLE_SIZE;
	return barrier;
}

bool HasShaderBufferWrites(const ShaderStageRuntime& runtime) {
	EXIT_IF(!runtime);
	const auto& program   = *runtime.program;
	const auto& resources = *runtime.resources;
	EXIT_IF(resources.buffers.size() != program.info.buffers.size());
	bool has_writes = false;
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		if (!program.info.buffers[i].written) {
			continue;
		}
		const auto& value = resources.buffers[i];
		EXIT_IF(value.dword_count < 4);
		ShaderBufferResource descriptor;
		std::memcpy(descriptor.fields, value.dwords.data(), sizeof(descriptor.fields));
		// A zero stride means byte addressing. For either addressing mode a nonzero record
		// count is exactly the condition for a nonempty descriptor range.
		has_writes |= descriptor.Base48() != 0 && descriptor.NumRecords() != 0;
	}
	return has_writes;
}

void ShaderAccessBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages) {
	EXIT_IF(vk_buffer == nullptr || !source_stages);
	const auto barrier = MakeShaderAccessDependency();
	{ GpuTimestampScope ts_scope(vk_buffer, GpuTimestamps::Kind::BarrierGlobal); vk_buffer.pipelineBarrier(source_stages, vk::PipelineStageFlagBits::eAllCommands,
	                          vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr); }
}

void ShaderWriteHazardBarrier(vk::CommandBuffer      vk_buffer,
                              vk::PipelineStageFlags destination_stages) {
	EXIT_IF(vk_buffer == nullptr || !destination_stages);
	const auto barrier = MakeShaderWriteHazardDependency();
	{ GpuTimestampScope ts_scope(vk_buffer, GpuTimestamps::Kind::BarrierGlobal); vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, destination_stages,
	                          vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr); }
}

void ShaderWriteBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages) {
	EXIT_IF(vk_buffer == nullptr || !source_stages);
	const auto barrier = MakeShaderWriteDependency();
	{ GpuTimestampScope ts_scope(vk_buffer, GpuTimestamps::Kind::BarrierGlobal); vk_buffer.pipelineBarrier(source_stages,
	                          vk::PipelineStageFlagBits::eComputeShader |
	                              vk::PipelineStageFlagBits::eAllGraphics |
	                              vk::PipelineStageFlagBits::eTransfer,
	                          vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr); }
}

bool EmitPendingHazardBarrier(vk::CommandBuffer vk_buffer, PendingHazardState& hazards,
                              BarrierProbe::Kind kind) {
	EXIT_IF(vk_buffer == nullptr);
	auto source = hazards.written_stages | hazards.read_stages | hazards.indirect_write_stages;
	if (hazards.transfer_wrote) {
		source |= vk::PipelineStageFlagBits::eTransfer;
	}
	if (!source) {
		hazards.Clear();
		return false;
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
	                        hazards.written_access;
	if (hazards.transfer_wrote) {
		barrier.srcAccessMask |= vk::AccessFlagBits::eTransferWrite;
	}
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	{ GpuTimestampScope ts_scope(vk_buffer, GpuTimestamps::Kind::BarrierGlobal); vk_buffer.pipelineBarrier(source, vk::PipelineStageFlagBits::eAllCommands,
	                          vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr); }
	hazards.Clear();
	BarrierProbe::NoteEmitted(kind);
	return true;
}

bool EmitLazyDispatchBarrier(vk::CommandBuffer vk_buffer, PendingHazardState& hazards, bool writes,
                             bool indirect, BarrierProbe::Kind kind) {
	EXIT_IF(vk_buffer == nullptr);
	// RAW/WAW: an earlier writer is still unresolved. WAR: this dispatch writes and something read
	// since the last barrier. Indirect: the arguments may have been written by a shader or a copy.
	const bool compute_hazard  = hazards.written_stages || (writes && hazards.read_stages);
	const bool indirect_hazard = indirect && (hazards.indirect_write_stages || hazards.transfer_wrote);
	if (!compute_hazard && !indirect_hazard) {
		return false;
	}
	// Pure RAW/indirect barriers leave the read side pending: the readers were not part of the
	// source scope, so a later writer still needs its WAR barrier.
	const auto pending_read = hazards.read_stages;
	if (!writes) {
		hazards.read_stages = {};
	}
	const bool emitted = EmitPendingHazardBarrier(vk_buffer, hazards, kind);
	if (!writes) {
		hazards.read_stages = pending_read;
	}
	return emitted;
}

void NoteDispatchHazards(PendingHazardState& hazards, bool writes) noexcept {
	hazards.read_stages |= vk::PipelineStageFlagBits::eComputeShader;
	if (writes) {
		hazards.written_stages |= vk::PipelineStageFlagBits::eComputeShader;
		hazards.indirect_write_stages |= vk::PipelineStageFlagBits::eComputeShader;
		hazards.written_access |= vk::AccessFlagBits::eShaderWrite;
	}
}

} // namespace Libs::Graphics
