#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/hostReadStats.h"
#include "graphics/host_gpu/renderer/gpuDiagnostics.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include "common/logging/log.h"
#include <chrono>
#include <utility>
#include <algorithm>
#include <array>
#include <mutex>

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
		Common::LockGuard lock(m_graphics.queue_mutex);
		ReportDeviceFault(m_graphics.device, m_graphics.queue);
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
	BudgetScope budget(BudgetProbe::Wait, GuestGpu::IsGpuThread());
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
		    &wait_info, GpuDiagnostics::Enabled() ? 5'000'000'000ull : 10'000'000'000ull);
		if (result == vk::Result::eTimeout && !GpuDiagnostics::Enabled()) {
			CheckStall("MasterSemaphore::Wait", 0.0);
		}
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
			if (status != vk::Result::eSuccess) {
				Common::LockGuard lock(m_graphics.queue_mutex);
				ReportDeviceFault(m_graphics.device, m_graphics.queue);
			} else if (n == 1) {
				Common::LockGuard lock(m_graphics.queue_mutex);
				ReportQueueCheckpoints(m_graphics.queue);
			}
			if (n == 1) {
				// Is the queue alive? An empty submission with a fence either completes (the hung
				// batch was dropped by the driver), stays pending (queue blocked behind it) or
				// fails with DEVICE_LOST (the context was reset).
				vk::FenceCreateInfo fence_info {};
				vk::Fence           fence {};
				if (m_graphics.device.createFence(&fence_info, nullptr, &fence) == vk::Result::eSuccess) {
					vk::Result submit_result;
					{
						Common::LockGuard lock(m_graphics.queue_mutex);
						submit_result = m_graphics.queue.submit(0, nullptr, fence);
					}
					vk::Result fence_result  = vk::Result::eNotReady;
					if (submit_result == vk::Result::eSuccess) {
						fence_result = m_graphics.device.waitForFences(1, &fence, VK_TRUE, 3'000'000'000ull);
					}
					std::printf("MasterSemaphore::Wait: liveness probe: empty submit=%s fence wait(3s)=%s\n",
					            vk::to_string(submit_result).c_str(), vk::to_string(fence_result).c_str());
					if (fence_result == vk::Result::eSuccess || submit_result != vk::Result::eSuccess) {
						m_graphics.device.destroyFence(fence, nullptr);
					}
					if (submit_result != vk::Result::eSuccess) {
						Common::LockGuard lock(m_graphics.queue_mutex);
						ReportDeviceFault(m_graphics.device, m_graphics.queue);
					}
				}
			}
			std::fflush(stdout);
		}
	} while (result == vk::Result::eTimeout);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

void MasterSemaphore::CheckStall(const char* where, double seconds) {
	{
		// The blocked-queue idle loop spins every 100 us; one semaphore query per 250 ms is plenty.
		std::lock_guard lock(m_stall_mutex);
		const auto now = std::chrono::steady_clock::now();
		if (seconds > 0.0 && now - m_stall_checked < std::chrono::milliseconds(250)) {
			return;
		}
		m_stall_checked = now;
	}
	Refresh();
	const auto submitted = CurrentTick() - 1; // NextTick() hands out the tick of the last submit
	const auto known     = KnownGpuTick();
	std::lock_guard lock(m_stall_mutex);
	if (known >= submitted) {
		m_stall_tick = 0;
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (m_stall_tick != known + 1) {
		m_stall_tick  = known + 1;
		m_stall_since = now;
		return;
	}
	const double waited = std::chrono::duration<double>(now - m_stall_since).count();
	if (waited < seconds || m_stall_reported_tick == m_stall_tick) {
		return;
	}
	m_stall_reported_tick = m_stall_tick;
	ReportStall(where, m_stall_tick, waited);
}

namespace SubmitLog {
namespace {
struct Record {
	vk::Semaphore                         timeline {};
	uint64_t                              tick            = 0;
	uint32_t                              wait_count      = 0;
	vk::Semaphore                         first_wait {};
	uint64_t                              first_wait_value = 0;
	std::chrono::steady_clock::time_point when {};
};
constexpr size_t      RING = 48;
std::mutex            g_mutex;
std::array<Record, RING> g_ring {};
size_t                g_next  = 0;
size_t                g_count = 0;
} // namespace

void Record(vk::Semaphore timeline, uint64_t tick, uint32_t wait_count, vk::Semaphore first_wait,
            uint64_t first_wait_value) {
	std::lock_guard lock(g_mutex);
	g_ring[g_next] = {timeline, tick, wait_count, first_wait, first_wait_value,
	                  std::chrono::steady_clock::now()};
	g_next         = (g_next + 1) % RING;
	g_count        = std::min(g_count + 1, RING);
}

void Dump(vk::Device device, vk::Semaphore reporting_timeline) {
	std::array<struct Record, RING> ring {};
	size_t                          count = 0;
	size_t                          first = 0;
	{
		std::lock_guard lock(g_mutex);
		ring  = g_ring;
		count = g_count;
		first = (g_next + RING - g_count) % RING;
	}
	// Current counter of every timeline that appears in the ring.
	std::array<std::pair<vk::Semaphore, uint64_t>, 8> counters {};
	size_t                                            num_counters = 0;
	for (size_t i = 0; i < count; i++) {
		const auto& r     = ring[(first + i) % RING];
		bool        known = false;
		for (size_t k = 0; k < num_counters; k++) {
			known |= counters[k].first == r.timeline;
		}
		if (!known && num_counters < counters.size()) {
			uint64_t   value  = 0;
			const auto status = device.getSemaphoreCounterValue(r.timeline, &value);
			counters[num_counters++] = {r.timeline, value};
			LOGF("GPU stall: timeline 0x%" PRIx64 "%s counter=%" PRIu64 " status=%s\n",
			     static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<VkSemaphore>(r.timeline))),
			     r.timeline == reporting_timeline ? " (stalled)" : "", value,
			     vk::to_string(status).c_str());
		}
	}
	const auto now = std::chrono::steady_clock::now();
	for (size_t i = 0; i < count; i++) {
		const auto& r       = ring[(first + i) % RING];
		uint64_t    counter = 0;
		for (size_t k = 0; k < num_counters; k++) {
			if (counters[k].first == r.timeline) {
				counter = counters[k].second;
			}
		}
		const double age = std::chrono::duration<double>(now - r.when).count();
		if (r.wait_count == 0) {
			LOGF("GPU stall:   submit tl=0x%" PRIx64 " tick=%" PRIu64 " %s age=%.1fs\n",
			     static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<VkSemaphore>(r.timeline))),
			     r.tick, counter >= r.tick ? "done" : "PENDING", age);
		} else {
			LOGF("GPU stall:   submit tl=0x%" PRIx64 " tick=%" PRIu64 " %s age=%.1fs waits=%u first=0x%"
			     PRIx64 " value=%" PRIu64 "\n",
			     static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<VkSemaphore>(r.timeline))),
			     r.tick, counter >= r.tick ? "done" : "PENDING", age, r.wait_count,
			     static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<VkSemaphore>(r.first_wait))),
			     r.first_wait_value);
		}
	}
}
} // namespace SubmitLog

void MasterSemaphore::ReportStall(const char* where, uint64_t tick, double waited_s) {
	// Zero Company 2026-10-12: four hangs where a submitted tick never completed (the GPU thread
	// either waited on it forever or sat idle while the game waited for its end-of-pipe event).
	LOGF("GPU stall: tick %" PRIu64 " incomplete for %.0f s (newest submitted %" PRIu64
	     ", completed %" PRIu64 ") at %s\n",
	     tick, waited_s, CurrentTick() - 1, KnownGpuTick(), where);
	std::printf("GPU stall: tick %" PRIu64 " incomplete for %.0f s at %s\n", tick, waited_s, where);
	if (GpuDiagnostics::Enabled()) {
		GpuDiagnostics::DumpStall(GpuDiagnostics::HandleValue(m_semaphore), tick, KnownGpuTick(),
		                          CurrentTick());
	}
	uint64_t   counter = 0;
	const auto status  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	LOGF("GPU stall: semaphore counter=%" PRIu64 " status=%s\n", counter,
	     vk::to_string(status).c_str());
	SubmitLog::Dump(m_graphics.device, m_semaphore);
	{
		Common::LockGuard lock(m_graphics.queue_mutex);
		ReportQueueCheckpoints(m_graphics.queue);
	}
	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence {};
	if (m_graphics.device.createFence(&fence_info, nullptr, &fence) == vk::Result::eSuccess) {
		vk::Result submit_result;
		{
			Common::LockGuard lock(m_graphics.queue_mutex);
			submit_result = m_graphics.queue.submit(0, nullptr, fence);
		}
		vk::Result fence_result = vk::Result::eNotReady;
		if (submit_result == vk::Result::eSuccess) {
			fence_result = m_graphics.device.waitForFences(1, &fence, VK_TRUE, 3'000'000'000ull);
		}
		LOGF("GPU stall: liveness probe empty submit=%s fence wait(3s)=%s\n",
		     vk::to_string(submit_result).c_str(), vk::to_string(fence_result).c_str());
		std::printf("GPU stall: liveness probe empty submit=%s fence wait(3s)=%s\n",
		            vk::to_string(submit_result).c_str(), vk::to_string(fence_result).c_str());
		if (fence_result == vk::Result::eSuccess || submit_result != vk::Result::eSuccess) {
			m_graphics.device.destroyFence(fence, nullptr);
		}
		if (submit_result != vk::Result::eSuccess || status != vk::Result::eSuccess) {
			Common::LockGuard lock(m_graphics.queue_mutex);
			ReportDeviceFault(m_graphics.device, m_graphics.queue);
		}
	}
	std::fflush(stdout);
	Log::Flush();
}

} // namespace Libs::Graphics
