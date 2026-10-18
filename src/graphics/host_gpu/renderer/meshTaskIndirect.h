#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHTASKINDIRECT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHTASKINDIRECT_H_

#include <cstdint>

namespace Libs::Graphics {

// Push block of the pass-through task shader (shaders/pass_through_task.task, keep in sync). It
// shares push-constant dwords with the mesh program, which on the task path reads none of draw
// dwords 0-5 from push constants (EmitMeshDrawParameter) but starts its shader-data window at
// dword MeshDrawDwordCount (6): the block must fit in the first 24 bytes.
struct MeshTaskPushConstants {
	uint64_t args_bda       = 0; // first DrawIndirectArgs record
	uint32_t stride         = 0; // bytes between records
	uint32_t primitive_info = 0; // PackMeshTaskPrimitiveInfo
};
inline constexpr uint32_t kTaskPushBytes = sizeof(MeshTaskPushConstants);

// primitives_per_group (bits 0-15), input primitive size (bits 16-23) and step (bits 24-31), as
// ShaderMeshInputInfo reports them (primitives_per_group <= 0x1ff, size and step <= 3).
[[nodiscard]] constexpr uint32_t PackMeshTaskPrimitiveInfo(uint32_t primitives_per_group,
                                                           uint32_t primitive_size,
                                                           uint32_t primitive_step) {
	return (primitives_per_group & 0xffffu) | ((primitive_size & 0xffu) << 16u) |
	       ((primitive_step & 0xffu) << 24u);
}

// Specialization constant ids of the task shader: the device's mesh workgroup limits.
inline constexpr uint32_t kTaskSpecMaxGroupsX = 0;
inline constexpr uint32_t kTaskSpecMaxGroupsY = 1;

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHTASKINDIRECT_H_
