#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <chrono>
#include <mutex>

namespace Libs::Graphics {

struct GraphicContext;

// Always-on ring of the most recent queue submissions (every CommandScheduler, including the
// presenter's). ReportStall prints it with each timeline's current counter so a stall shows which
// submission is stuck and whether it waited on another semaphore (e.g. a swapchain acquire).
namespace SubmitLog {
void Record(vk::Semaphore timeline, uint64_t tick, uint32_t wait_count, vk::Semaphore first_wait,
            uint64_t first_wait_value);
void Dump(vk::Device device, vk::Semaphore reporting_timeline);
} // namespace SubmitLog

class MasterSemaphore {
public:
	explicit MasterSemaphore(GraphicContext& graphics);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		return m_current_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		return m_gpu_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] bool     IsFree(uint64_t tick) const noexcept { return KnownGpuTick() >= tick; }
	[[nodiscard]] uint64_t NextTick() noexcept {
		return m_current_tick.fetch_add(1, std::memory_order_release);
	}
	[[nodiscard]] vk::Semaphore Handle() const noexcept { return m_semaphore; }

	void Refresh();
	void Wait(uint64_t tick);
	// Always-on stall watchdog: when the newest submitted tick has stayed incomplete for `seconds`,
	// report it once (log + stdout), dump the diagnostics ring if enabled, and probe device
	// liveness (empty submit + fence, VK_EXT_device_fault). Callable from any thread.
	void CheckStall(const char* where, double seconds = 10.0);

private:
	void ReportStall(const char* where, uint64_t tick, double waited_s);

	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
	std::mutex                            m_stall_mutex;
	uint64_t                              m_stall_tick          = 0;
	uint64_t                              m_stall_reported_tick = 0;
	std::chrono::steady_clock::time_point m_stall_since {};
	std::chrono::steady_clock::time_point m_stall_checked {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
