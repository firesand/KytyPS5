#include "graphics/host_gpu/renderer/cache/hostResidentMemory.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>

namespace Libs::Graphics {

HostResidentMemory::HostResidentMemory(GraphicContext& graphics)
    : m_graphics(&graphics), m_production(true) {
	m_resolver = [](uint64_t vaddr, uint64_t size, uint64_t* offset) {
		return LibKernel::Memory::QueryBackingOffset(vaddr, size, offset);
	};
}

HostResidentMemory::HostResidentMemory(GraphicContext& graphics, uint8_t* base, uint64_t size,
                                       OffsetResolver resolver)
    : m_graphics(&graphics), m_base(base), m_size(size), m_resolver(std::move(resolver)) {
	TryInit();
}

bool HostResidentMemory::TryInit() {
	if (m_available.load(std::memory_order_acquire)) {
		return true;
	}
	if (m_latched_off.load(std::memory_order_relaxed)) {
		return false;
	}
	std::lock_guard lock(m_mutex);
	if (m_available.load(std::memory_order_relaxed)) {
		return true;
	}
	if (!m_graphics->external_memory_host_enabled) {
		LOGF("HostResident: disabled, VK_EXT_external_memory_host is not enabled\n");
		m_latched_off = true;
		return false;
	}
	if (m_production) {
		m_base = LibKernel::Memory::GetBackingHostView(&m_size);
		if (m_base == nullptr) {
			return false; // kernel memory not initialised yet; retry on the next call
		}
	}
	const uint64_t alignment = std::max<uint64_t>(m_graphics->min_imported_host_pointer_alignment, 1);
	if (m_base == nullptr || m_size == 0 || reinterpret_cast<uintptr_t>(m_base) % alignment != 0) {
		LOGF("HostResident: disabled, no usable backing view (base=%p size=%" PRIu64
		     " alignment=%" PRIu64 ")\n",
		     static_cast<void*>(m_base), m_size, alignment);
		m_latched_off = true;
		return false;
	}
	m_size -= m_size % alignment;
	const uint64_t count = (m_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
	m_chunk_storage      = std::make_unique<Chunk[]>(count);
	for (uint64_t i = 0; i < count; ++i) {
		m_chunks.push_back(&m_chunk_storage[i]);
		m_chunks.back()->bytes = std::min(CHUNK_SIZE, m_size - i * CHUNK_SIZE);
	}
	m_available.store(true, std::memory_order_release);
	LOGF("HostResident: enabled, backing %" PRIu64 " MiB in %" PRIu64
	     " chunk(s) of up to %" PRIu64 " MiB, import alignment %" PRIu64 ", chunks import lazily\n",
	     m_size >> 20, count, CHUNK_SIZE >> 20, alignment);
	return true;
}

HostResidentMemory::~HostResidentMemory() {
	if (m_graphics == nullptr || m_graphics->device == nullptr) {
		return;
	}
	for (auto* chunk: m_chunks) {
		if (chunk->buffer != nullptr) {
			m_graphics->device.destroyBuffer(chunk->buffer, nullptr);
		}
		if (chunk->memory != nullptr) {
			m_graphics->device.freeMemory(chunk->memory, nullptr);
		}
	}
}

uint32_t HostResidentMemory::ImportAll() {
	uint32_t imported = 0;
	for (uint64_t i = 0; TryInit() && i < m_chunks.size(); ++i) {
		imported += ImportChunk(i) ? 1 : 0;
	}
	return imported;
}

bool HostResidentMemory::ImportChunk(uint64_t index) {
	auto& chunk = *m_chunks[index];
	if (const auto state = chunk.state.load(std::memory_order_acquire); state != 0) {
		return state == 1;
	}
	std::lock_guard lock(m_mutex);
	if (const auto state = chunk.state.load(std::memory_order_acquire); state != 0) {
		return state == 1;
	}
	const auto fail = [&](const char* reason, vk::Result result = vk::Result::eSuccess) {
		LOGF("HostResident: chunk %" PRIu64 " import failed: %s%s%s (feature stays off for it)\n",
		     index, reason, result == vk::Result::eSuccess ? "" : " ",
		     result == vk::Result::eSuccess ? "" : vk::to_string(result).c_str());
		if (chunk.buffer != nullptr) {
			m_graphics->device.destroyBuffer(chunk.buffer, nullptr);
			chunk.buffer = nullptr;
		}
		if (chunk.memory != nullptr) {
			m_graphics->device.freeMemory(chunk.memory, nullptr);
			chunk.memory = nullptr;
		}
		chunk.state.store(2, std::memory_order_release);
		return false;
	};
	const auto begin = std::chrono::steady_clock::now();
	auto&      device = m_graphics->device;
	auto*      pointer = m_base + index * CHUNK_SIZE;

	vk::MemoryHostPointerPropertiesEXT pointer_properties {};
	if (auto r = device.getMemoryHostPointerPropertiesEXT(
	        vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, pointer, &pointer_properties);
	    r != vk::Result::eSuccess) {
		return fail("vkGetMemoryHostPointerPropertiesEXT", r);
	}

	vk::ExternalMemoryBufferCreateInfo external_buffer {};
	external_buffer.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
	vk::BufferCreateInfo buffer_info {};
	buffer_info.pNext = &external_buffer;
	buffer_info.size  = chunk.bytes;
	buffer_info.usage = USAGE;
	if (auto r = device.createBuffer(&buffer_info, nullptr, &chunk.buffer);
	    r != vk::Result::eSuccess) {
		return fail("vkCreateBuffer", r);
	}
	vk::MemoryRequirements requirements {};
	device.getBufferMemoryRequirements(chunk.buffer, &requirements);
	if (requirements.size > chunk.bytes) {
		return fail("buffer memory requirements exceed the host chunk");
	}
	const auto bits = pointer_properties.memoryTypeBits & requirements.memoryTypeBits;
	const auto& memory_properties = m_graphics->GetPhysicalDeviceMemoryProperties();
	constexpr auto wanted = vk::MemoryPropertyFlagBits::eHostVisible |
	                        vk::MemoryPropertyFlagBits::eHostCoherent;
	uint32_t type = UINT32_MAX;
	for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
		if ((bits & (1u << i)) != 0 &&
		    (memory_properties.memoryTypes[i].propertyFlags & wanted) == wanted) {
			// Prefer cached (fast host reads) over write-combined types when several qualify.
			if (type == UINT32_MAX ||
			    ((memory_properties.memoryTypes[i].propertyFlags &
			      vk::MemoryPropertyFlagBits::eHostCached) &&
			     !(memory_properties.memoryTypes[type].propertyFlags &
			       vk::MemoryPropertyFlagBits::eHostCached))) {
				type = i;
			}
		}
	}
	if (type == UINT32_MAX) {
		return fail("no host-visible host-coherent memory type for the host pointer");
	}

	vk::ImportMemoryHostPointerInfoEXT import_info {};
	import_info.handleType   = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
	import_info.pHostPointer = pointer;
	vk::MemoryAllocateFlagsInfo flags_info {};
	flags_info.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
	flags_info.pNext = &import_info;
	vk::MemoryAllocateInfo allocate_info {};
	allocate_info.pNext           = &flags_info;
	allocate_info.allocationSize  = chunk.bytes;
	allocate_info.memoryTypeIndex = type;
	if (auto r = device.allocateMemory(&allocate_info, nullptr, &chunk.memory);
	    r != vk::Result::eSuccess) {
		return fail("vkAllocateMemory (import)", r);
	}
	if (auto r = device.bindBufferMemory(chunk.buffer, chunk.memory, 0); r != vk::Result::eSuccess) {
		return fail("vkBindBufferMemory", r);
	}
	vk::BufferDeviceAddressInfo address_info {};
	address_info.buffer = chunk.buffer;
	chunk.address       = device.getBufferAddress(address_info);
	chunk.state.store(1, std::memory_order_release);

	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                    std::chrono::steady_clock::now() - begin)
	                    .count();
	LOGF("HostResident: imported %" PRIu64 " MiB at %p (chunk %" PRIu64
	     ", memory type %u flags 0x%x, address 0x%" PRIx64 ", %lld ms)\n",
	     chunk.bytes >> 20, static_cast<void*>(pointer), index, type,
	     static_cast<uint32_t>(memory_properties.memoryTypes[type].propertyFlags),
	     static_cast<uint64_t>(chunk.address), static_cast<long long>(ms));
	return true;
}

bool HostResidentMemory::ResolveOffset(uint64_t backing_offset, uint64_t size, vk::Buffer& buffer,
                                       uint64_t& offset) {
	if (!TryInit() || size == 0 || backing_offset >= m_size || size > m_size - backing_offset) {
		return false;
	}
	const uint64_t index = backing_offset / CHUNK_SIZE;
	if ((backing_offset + size - 1) / CHUNK_SIZE != index || !ImportChunk(index)) {
		return false;
	}
	buffer = m_chunks[index]->buffer;
	offset = backing_offset - index * CHUNK_SIZE;
	return true;
}

bool HostResidentMemory::Resolve(uint64_t vaddr, uint64_t size, vk::Buffer& buffer,
                                 uint64_t& offset) {
	if (!TryInit() || size == 0) {
		return false;
	}
	uint64_t backing_offset = 0;
	if (!m_resolver(vaddr, size, &backing_offset)) {
		return false;
	}
	return ResolveOffset(backing_offset, size, buffer, offset);
}

vk::DeviceAddress HostResidentMemory::ChunkAddress(uint64_t chunk) const {
	return chunk < m_chunks.size() ? m_chunks[chunk]->address : 0;
}

} // namespace Libs::Graphics
