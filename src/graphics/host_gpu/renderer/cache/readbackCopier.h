#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKCOPIER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKCOPIER_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
class MasterSemaphore;
struct GraphicContext;

// KYTY_READBACK_COPY=1 (default off; GraphicContext::ReadbackCopyRequested): a guest-fault readback
// whose producers have all completed (TryBarReadback's gate) is copied on the device's transfer-only
// queue into host-cached staging, instead of read over the uncached BAR mapping. Alternative A of
// the readback design: the A-bench (2026-10-21) measured 4 KiB in 19 us vs 39 us over BAR with queue
// 0 busy, 64 KiB in 47 us vs 532 us; ng-249 read 1351 pages/s at ~91 us each over BAR.
// GPU thread only (the readback runs inside SendCommandSync), synchronous: one copy at a time.
class ReadbackCopier {
public:
	ReadbackCopier(GraphicContext& graphics, CommandScheduler& scheduler);
	~ReadbackCopier();
	KYTY_CLASS_NO_COPY(ReadbackCopier);

	[[nodiscard]] uint64_t Capacity() const noexcept;
	// Copies the ranges [copy.srcOffset, +copy.size) of `source` into the guest backing at
	// cpu_address + copy.srcOffset. The caller guarantees every write to them completed (the BAR
	// gate); the submit still waits on the master timeline's known value as the formal cross-queue
	// dependency. False (nothing written) when the ranges exceed the staging: read them over BAR.
	bool Copy(const Buffer& source, uint64_t cpu_address, std::span<const vk::BufferCopy> copies);

private:
	GraphicContext&               m_graphics;
	MasterSemaphore&              m_master;
	std::unique_ptr<Buffer>       m_staging;
	vk::CommandPool               m_pool = nullptr;
	vk::CommandBuffer             m_command = nullptr;
	vk::Semaphore                 m_done = nullptr; // timeline, signalled by each copy
	uint64_t                      m_done_value = 0;
	std::vector<vk::BufferCopy>   m_regions;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKCOPIER_H_
