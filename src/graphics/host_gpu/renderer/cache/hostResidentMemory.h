#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOSTRESIDENTMEMORY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOSTRESIDENTMEMORY_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

// KYTY_HOST_RESIDENT=1 (default off; any other value or unset = no behaviour change).
inline bool HostResidentRequested() noexcept {
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_HOST_RESIDENT");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return requested;
}

// Host-resident guest memory prototype. The guest's direct memory is one memfd with a permanent,
// never write-protected MAP_SHARED view. With VK_EXT_external_memory_host that view is imported as
// device buffers (1 GiB chunks), so the GPU reads CPU-written guest data in place over PCIe: no
// VRAM copy, no upload, no page protection for the ranges bound this way.
//
// Coherence assumption of the prototype: the memory type is host-coherent, so a CPU write that
// completes before the GPU executes the draw is visible to it. Vulkan formally guarantees that for
// writes made before vkQueueSubmit; the emulator submits earlier than real hardware executes, so a
// guest write between submit and execution is also seen (documented, not enforced).
class HostResidentMemory {
public:
	static constexpr uint64_t CHUNK_SIZE = uint64_t {1} << 30;
	static constexpr vk::BufferUsageFlags USAGE =
	    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
	    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress;

	using OffsetResolver = std::function<bool(uint64_t vaddr, uint64_t size, uint64_t* offset)>;

	// Production: the kernel's backing view and vaddr -> direct-memory offset query.
	explicit HostResidentMemory(GraphicContext& graphics);
	// Explicit view and resolver (tests).
	HostResidentMemory(GraphicContext& graphics, uint8_t* base, uint64_t size,
	                   OffsetResolver resolver);
	~HostResidentMemory();
	HostResidentMemory(const HostResidentMemory&)            = delete;
	HostResidentMemory& operator=(const HostResidentMemory&) = delete;

	// True when the extension is enabled and a backing view exists. Chunks are imported lazily on
	// first use (importing pins the pages, so an untouched chunk is never committed).
	[[nodiscard]] bool Available() { return TryInit(); }
	// Imports every chunk now; returns the number of chunks that imported.
	uint32_t ImportAll();

	// Guest vaddr range -> imported buffer + byte offset. False if unavailable, not direct memory,
	// not one contiguous run of the backing object, crossing a chunk boundary, or import failed.
	[[nodiscard]] bool Resolve(uint64_t vaddr, uint64_t size, vk::Buffer& buffer, uint64_t& offset);
	// Same for a direct-memory (backing object) offset.
	[[nodiscard]] bool ResolveOffset(uint64_t backing_offset, uint64_t size, vk::Buffer& buffer,
	                                 uint64_t& offset);
	[[nodiscard]] vk::DeviceAddress ChunkAddress(uint64_t chunk) const;
	[[nodiscard]] uint64_t          ChunkCount() const noexcept { return m_chunks.size(); }

private:
	struct Chunk {
		std::atomic<uint8_t> state {0}; // 0 not imported, 1 imported, 2 failed
		vk::Buffer           buffer  = nullptr;
		vk::DeviceMemory     memory  = nullptr;
		vk::DeviceAddress    address = 0;
		uint64_t             bytes   = 0;
	};

	// Idempotent. Production looks the kernel backing view up on first use (graphics init may
	// precede kernel memory init) and retries until it exists; a disabled extension latches.
	bool TryInit();
	bool ImportChunk(uint64_t index);

	GraphicContext*              m_graphics = nullptr;
	uint8_t*                     m_base     = nullptr;
	uint64_t                     m_size     = 0;
	OffsetResolver               m_resolver;
	std::atomic<bool>            m_available {false};
	std::atomic<bool>            m_latched_off {false};
	bool                         m_production = false;
	std::mutex                   m_mutex;
	std::unique_ptr<Chunk[]>     m_chunk_storage;
	std::vector<Chunk*>          m_chunks;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOSTRESIDENTMEMORY_H_
