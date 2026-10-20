#include "graphics/host_gpu/renderer/cache/readbackCopier.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "kernel/memory.h"

#include <chrono>
#include <cinttypes>

namespace Libs::Graphics {

namespace {
// Staging size: a readback larger than this is read over BAR (ng-249: 3.7 KiB average).
constexpr uint64_t kStagingBytes = 4ull << 20;
// Spin this long on the copy's timeline value, then block (A-bench: blocking adds 25-45 us of
// wake-up; a copy of up to 256 KiB finishes within ~110 us even under VRAM pressure).
constexpr auto kSpin = std::chrono::microseconds(300);
} // namespace

ReadbackCopier::ReadbackCopier(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_master(scheduler.GetMasterSemaphore()) {
	EXIT_IF(graphics.readback_queue == nullptr);
	m_staging = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
	                                     vk::BufferUsageFlagBits::eTransferDst, kStagingBytes);
	EXIT_IF(m_staging->Mapped().empty());

	vk::CommandPoolCreateInfo pool_info {};
	pool_info.queueFamilyIndex = graphics.readback_queue_family;
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	EXIT_NOT_IMPLEMENTED(graphics.device.createCommandPool(&pool_info, nullptr, &m_pool) != vk::Result::eSuccess);
	vk::CommandBufferAllocateInfo alloc {};
	alloc.commandPool        = m_pool;
	alloc.level              = vk::CommandBufferLevel::ePrimary;
	alloc.commandBufferCount = 1;
	EXIT_NOT_IMPLEMENTED(graphics.device.allocateCommandBuffers(&alloc, &m_command) != vk::Result::eSuccess);

	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;
	vk::SemaphoreCreateInfo semaphore_info {};
	semaphore_info.pNext = &type_info;
	EXIT_NOT_IMPLEMENTED(graphics.device.createSemaphore(&semaphore_info, nullptr, &m_done) != vk::Result::eSuccess);
	LOGF("ReadbackCopy: enabled, transfer queue family %u, %" PRIu64 " KiB host-%s staging\n",
	     graphics.readback_queue_family, kStagingBytes >> 10u, m_staging->IsCoherent() ? "coherent" : "cached");
}

ReadbackCopier::~ReadbackCopier() {
	// Copies are synchronous, so none is in flight here.
	if (m_done != nullptr) m_graphics.device.destroySemaphore(m_done);
	if (m_pool != nullptr) m_graphics.device.destroyCommandPool(m_pool);
}

uint64_t ReadbackCopier::Capacity() const noexcept {
	return m_staging->Size();
}

bool ReadbackCopier::Copy(const Buffer& source, uint64_t cpu_address, std::span<const vk::BufferCopy> copies) {
	m_regions.clear();
	uint64_t total = 0;
	for (const auto& copy: copies) {
		m_regions.push_back({copy.srcOffset, total, copy.size});
		total += Common::AlignUp(copy.size, uint64_t {64}); // separate cache lines, as the copy path packs
	}
	if (total == 0 || total > m_staging->Size()) {
		return false;
	}

	const auto& device = m_graphics.device;
	EXIT_NOT_IMPLEMENTED(m_command.reset({}) != vk::Result::eSuccess);
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	EXIT_NOT_IMPLEMENTED(m_command.begin(&begin) != vk::Result::eSuccess);
	m_command.copyBuffer(source.Handle(), m_staging->Handle(), static_cast<uint32_t>(m_regions.size()),
	                     m_regions.data());
	// The host reads the staging after the timeline wait below (review: host-read barrier).
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = m_staging->Handle();
	barrier.offset              = 0;
	barrier.size                = total;
	m_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 0,
	                          nullptr, 1, &barrier, 0, nullptr);
	EXIT_NOT_IMPLEMENTED(m_command.end() != vk::Result::eSuccess);

	// Wait on what queue 0 is known to have completed (review: KnownGpuTick, not the producer's
	// stamp, which can be 0 or older than writes FlushWriteFaults completed). It is already reached,
	// so it costs nothing; it is the memory dependency between the two queues.
	const uint64_t wait_value   = m_master.KnownGpuTick();
	const uint64_t signal_value = ++m_done_value;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount   = 1;
	timeline.pWaitSemaphoreValues      = &wait_value;
	timeline.signalSemaphoreValueCount = 1;
	timeline.pSignalSemaphoreValues    = &signal_value;
	const vk::Semaphore          master     = m_master.Handle();
	const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
	vk::SubmitInfo               submit {};
	submit.pNext                = &timeline;
	submit.waitSemaphoreCount   = 1;
	submit.pWaitSemaphores      = &master;
	submit.pWaitDstStageMask    = &wait_stage;
	submit.commandBufferCount   = 1;
	submit.pCommandBuffers      = &m_command;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores    = &m_done;
	const auto submitted = m_graphics.readback_queue.submit(1, &submit, nullptr);
	if (submitted != vk::Result::eSuccess) {
		EXIT("ReadbackCopy: vkQueueSubmit failed: %s\n", vk::to_string(submitted).c_str());
	}

	// Spin, then block (review: a block fallback after a few hundred us).
	const auto spin_until = std::chrono::steady_clock::now() + kSpin;
	uint64_t   value      = 0;
	for (;;) {
		const auto read = device.getSemaphoreCounterValue(m_done, &value);
		if (read != vk::Result::eSuccess) {
			EXIT("ReadbackCopy: vkGetSemaphoreCounterValue failed: %s\n", vk::to_string(read).c_str());
		}
		if (value >= signal_value || std::chrono::steady_clock::now() >= spin_until) break;
	}
	if (value < signal_value) {
		vk::SemaphoreWaitInfo wait {};
		wait.semaphoreCount = 1;
		wait.pSemaphores    = &m_done;
		wait.pValues        = &signal_value;
		const auto waited   = device.waitSemaphores(&wait, UINT64_MAX);
		if (waited != vk::Result::eSuccess) {
			EXIT("ReadbackCopy: vkWaitSemaphores failed: %s\n", vk::to_string(waited).c_str());
		}
	}

	if (!m_staging->IsCoherent()) {
		m_staging->Invalidate(0, total);
	}
	const auto* staged = m_staging->Mapped().data();
	for (const auto& region: m_regions) {
		Libs::LibKernel::Memory::WriteBacking(cpu_address + region.srcOffset, staged + region.dstOffset, region.size);
	}
	return true;
}

} // namespace Libs::Graphics
