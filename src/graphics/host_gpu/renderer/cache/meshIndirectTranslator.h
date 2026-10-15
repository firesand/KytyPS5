#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_MESHINDIRECTTRANSLATOR_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_MESHINDIRECTTRANSLATOR_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <cstdint>

namespace Libs::Graphics {

class CommandScheduler;
struct GraphicContext;

// Mesh-stage GPU-side indirect draws (KYTY_CP_GPU_INDIRECT). A compute pass turns the guest's
// DrawIndexedIndirect records into VkDrawMeshTasksIndirectCommandEXT commands and a per-record
// scratch (8 dwords, ShaderRecompiler::IR::MeshRecord*) that the mesh_record shader variant reads
// through DrawIndex. Output lives in a GPU-internal ring; every pass is ordered after the previous
// readers of the ring slot, so a wrap is safe.
class MeshIndirectTranslator {
public:
	static constexpr uint64_t RingSize     = 8ull << 20;
	static constexpr uint32_t CommandBytes = 12;  // VkDrawMeshTasksIndirectCommandEXT
	static constexpr uint32_t RecordBytes  = 32;

	struct Request {
		vk::Buffer args_buffer;
		uint64_t   args_offset = 0; // bytes into args_buffer, 4-aligned
		uint64_t   args_range  = 0; // bytes from the start of args_buffer through the last record
		uint32_t   count       = 0;
		uint32_t   stride      = 0; // bytes, 4-aligned, >= 20
		uint32_t   index_buffer_size = 0; // elements
		uint64_t   index_base  = 0;
		uint32_t   element_size = 0;      // guest index bytes
		uint32_t   primitive_size = 0;    // ShaderMeshInputInfo::InputPrimitiveSize / Step
		uint32_t   primitive_step = 0;
		uint32_t   primitives_per_group = 0;
		uint32_t   max_groups_x = 0;      // mesh shader launch limits
		uint32_t   max_groups_y = 0;
		uint32_t   max_groups_total = 0;
		// The records may have been written by the GPU in this command stream.
		bool       producer_barrier = false;
	};

	struct Result {
		vk::Buffer        buffer;           // VkDrawMeshTasksIndirectCommandEXT array lives here
		uint64_t          commands_offset = 0;
		vk::DeviceAddress records_address = 0; // scratch records, MeshRecordDwords apart
	};

	MeshIndirectTranslator(GraphicContext& graphics, CommandScheduler& scheduler);
	~MeshIndirectTranslator();
	KYTY_CLASS_NO_COPY(MeshIndirectTranslator);

	[[nodiscard]] static constexpr uint64_t ScratchBytes(uint32_t count) {
		return ((uint64_t {count} * CommandBytes + 15u) & ~uint64_t {15}) +
		       uint64_t {count} * RecordBytes;
	}
	// A packet needing more than half the ring falls back to the host path.
	[[nodiscard]] static constexpr bool Fits(uint32_t count) {
		return count != 0 && ScratchBytes(count) <= RingSize / 2;
	}

	// Records into `command` (outside any dynamic rendering instance). Returns where the draw reads.
	[[nodiscard]] Result Translate(vk::CommandBuffer command, const Request& request);

private:
	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	Buffer                  m_scratch;
	uint64_t                m_cursor            = 0;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::Pipeline            m_pipeline          = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_MESHINDIRECTTRANSLATOR_H_
