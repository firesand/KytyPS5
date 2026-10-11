#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/gpuDiagnostics.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	if (result != vk::Result::eSuccess) {
		std::printf("MasterSemaphore::Refresh: vkGetSemaphoreCounterValue = %s\n",
		            vk::to_string(result).c_str());
		ReportDeviceFault(m_graphics.device);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	// Diagnostic slices never turn an incomplete submission into a successful wait.
	// Default behavior and the eventual Vulkan error handling remain unchanged.
	vk::Result result;
	bool reported = false;
	do {
		result = m_graphics.device.waitSemaphores(
		    &wait_info, GpuDiagnostics::Enabled() ? 5'000'000'000ull : UINT64_MAX);
		if (result == vk::Result::eTimeout && GpuDiagnostics::Enabled() && !reported) {
			Refresh();
			GpuDiagnostics::DumpStall(GpuDiagnostics::HandleValue(m_semaphore), tick,
			                          KnownGpuTick(), CurrentTick());
			reported = true;
		}
		if (result == vk::Result::eTimeout && GpuDiagnostics::Enabled()) {
			// Claude probe: is the device lost? vkGetSemaphoreCounterValue returns
			// VK_ERROR_DEVICE_LOST without blocking; VK_EXT_device_fault names the address.
			static std::atomic<uint32_t> timeouts = 0;
			const auto n = timeouts.fetch_add(1, std::memory_order_relaxed) + 1;
			uint64_t   counter = 0;
			const auto status  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
			std::printf("MasterSemaphore::Wait: timeout #%u waiting for %" PRIu64 ", counter=%" PRIu64
			            " status=%s\n", n, tick, counter, vk::to_string(status).c_str());
			if (status != vk::Result::eSuccess || n == 2) {
				ReportDeviceFault(m_graphics.device);
			}
			std::fflush(stdout);
		}
	} while (result == vk::Result::eTimeout && GpuDiagnostics::Enabled());
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

} // namespace Libs::Graphics
