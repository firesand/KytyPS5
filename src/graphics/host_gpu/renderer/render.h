#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/hostReadStats.h"
#include "graphics/host_gpu/renderer/hazardState.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	EopWrite,
	EopInterrupt,
	EopWriteBack,
	EopFlip,
	EopWriteBackFlip,
	EopOnlyFlip,
	DispatchIndirect,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// KYTY_CP_GPU_INDIRECT: an indirect draw whose argument records stay on the GPU. The renderer
// issues vkCmdDraw[Indexed]Indirect[Count] over them instead of the command processor reading the
// records on the host. For indexed draws the whole INDEX_BUFFER_SIZE range at index_base is bound.
struct GpuIndirectDraw {
	uint64_t             args_addr          = 0;
	uint64_t             count_addr         = 0; // 0: draw_count records, else a GPU-read count dword
	uint32_t             draw_count         = 0; // exact count, or the maximum when count_addr != 0
	uint32_t             stride             = 0;
	uint64_t             index_base         = 0;
	uint32_t             index_buffer_size  = 0; // in elements
	ShaderDrawParameters parameters;
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	const GpuIndirectDraw* gpu_indirect         = nullptr;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	const GpuIndirectDraw* gpu_indirect         = nullptr;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;
	[[nodiscard]] bool IsRendering() const noexcept { return m_rendering; }

	// Handle() is the funnel for every consumer that records commands the hazard tracker does not
	// know about (uploads, copies, image transitions, readbacks): under KYTY_LAZY_BARRIERS it first
	// resolves pending shader hazards when no render pass is open (EndRendering resolves them
	// otherwise). Lazy-aware dispatch/draw recording uses RawHandle().
	[[nodiscard]] vk::CommandBuffer Handle() const;
	[[nodiscard]] vk::CommandBuffer RawHandle() const;
	[[nodiscard]] PendingHazardState& Hazards() const noexcept { return m_hazards; }
	// Ends the render pass and emits the barrier for pending hazards, if the tracker is on and
	// any consumer-visible hazard is pending. Called before every draw.
	void PrepareForDraw() const;
	void FlushHazards(BarrierProbe::Kind kind) const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	mutable PendingHazardState m_hazards;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute);

	// True when an indirect draw can be issued with GPU-resident arguments: the device supports it
	// and nothing about the bound state needs the argument values on the host (mesh/NGG or
	// tessellation shaders, quad lists, 8-bit indices, custom primitive-restart values).
	// `parameters` names the user-data slots the packet patches; a vertex shader whose resource plan
	// evaluates one of them on the host (or one not compiled yet) keeps the host path (Srt).
	// MeshDenied / Denied: the bisection hash lists (gpuIndirectFilter.h) keep the program on the
	// host path. MeshCountAddr: the pass-through task shader reads `draw_count` records and cannot
	// honour a GPU count dword. MeshPrim: an input primitive type the task shader's group formula
	// does not cover. `shader` (optional) receives the bound vertex-stage facts for the caller's log.
	enum class GpuIndirectBlock { None, Mesh, MeshDenied, MeshCountAddr, MeshPrim, Denied, Srt, Other };
	struct GpuIndirectShaderInfo {
		bool     known       = false; // a valid vertex-stage shader is bound
		bool     mesh        = false;
		uint64_t hash        = 0;     // VertexShaderIsMesh's hash (ES / front half)
		uint32_t fast_launch = 0;
		uint32_t primitive   = 0;
		uint32_t ge_primitive_group_size = 0;
	};
	[[nodiscard]] bool CanDrawIndirectOnGpu(CommandBuffer& buffer, bool indexed,
	                                        uint32_t index_type_and_size, uint32_t draw_count,
	                                        const ShaderDrawParameters& parameters,
	                                        GpuIndirectBlock* block = nullptr,
	                                        GpuIndirectShaderInfo* shader = nullptr,
	                                        bool has_count_addr = false);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           PrepareBufferImageWrites(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);

private:
	void DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	void DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	// Fast-clear-eliminate / DCC- and FMask-decompress "draws" (CB_COLOR_CONTROL modes 2/5/6) carry
	// no color of their own. Returns true when the draw was consumed. KYTY_FCE_MATERIALIZE=1 also
	// applies the tracked color clear of CB0 at that point (see renderDraw.cpp).
	[[nodiscard]] bool ConsumeMetadataColorOperation(CommandBuffer& buffer,
	                                                 uint32_t       render_target_slice_offset);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state,
	                                          const ShaderDrawParameters* draw_parameters = nullptr);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable,
	                         const GpuIndirectDraw* gpu_indirect = nullptr);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                              CommandBuffer& command, uint32_t group_x,
	                                              uint32_t group_y, uint32_t group_z, uint32_t mode);

	// KYTY_SUBMIT_AHEAD: the learned producer dispatch recorded last (0 = none), taken once by the
	// command processor, which submits before its next work packet.
	[[nodiscard]] uint64_t ConsumeSubmitAheadProducer(uint64_t& tick) noexcept {
		const auto producer      = m_submit_ahead_producer;
		tick                     = m_submit_ahead_tick;
		m_submit_ahead_producer = 0;
		return producer;
	}

	// Device address of the task shader's MeshTaskCounters (KYTY_PROBE_FAULT_HISTOGRAM only,
	// else 0); adds what the GPU counted since the previous call to the HostReadProbe counters.
	[[nodiscard]] uint64_t TaskCountersAddress();

	RenderContext&                        m_context;
	std::unique_ptr<Buffer>               m_task_counters;
	std::array<uint32_t, 2>               m_task_counters_seen {};
	uint64_t                              m_submit_ahead_producer = 0;
	uint64_t                              m_submit_ahead_tick     = 0; // command buffer it was recorded in
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
