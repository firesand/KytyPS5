#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostReadStats.h"
#include "graphics/host_gpu/renderer/hazardState.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

namespace Libs::Graphics {

struct ShaderStageRuntime;

vk::ShaderStageFlagBits NativeShaderStage(ShaderType stage);
vk::PipelineStageFlags  ShaderPipelineStages(vk::ShaderStageFlags stages);
vk::MemoryBarrier       MakeShaderAccessDependency();
vk::MemoryBarrier       MakeShaderWriteHazardDependency();
vk::MemoryBarrier       MakeShaderWriteDependency();
vk::BufferMemoryBarrier MakeGdsDependency(vk::Buffer buffer);
bool HasShaderBufferWrites(const ShaderStageRuntime& runtime);
bool HasShaderBufferReads(const ShaderStageRuntime& runtime);
void ShaderAccessBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages);
void ShaderWriteHazardBarrier(vk::CommandBuffer      vk_buffer,
                              vk::PipelineStageFlags destination_stages);
void ShaderWriteBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages);

// Emits the one barrier that resolves every pending hazard (source: the pending writer/reader
// stages, destination: eAllCommands with memory read+write, so graphics, transfer, indirect and
// later dispatches are all covered) and clears the state. No-op when nothing is pending. Never
// call inside a dynamic render pass.
bool EmitPendingHazardBarrier(vk::CommandBuffer vk_buffer, PendingHazardState& hazards,
                              BarrierProbe::Kind kind);
// The merged pre-dispatch barrier of the lazy path. `writes`: the dispatch writes memory (WAR is
// checked); `indirect`: the dispatch reads its arguments from memory (the eDrawIndirect dependency
// is checked). Returns true when a barrier was recorded.
bool EmitLazyDispatchBarrier(vk::CommandBuffer vk_buffer, PendingHazardState& hazards, bool writes,
                             bool indirect, BarrierProbe::Kind kind);
// Records the effect of a dispatch on the tracker.
void NoteDispatchHazards(PendingHazardState& hazards, bool writes) noexcept;

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
