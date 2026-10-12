#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_

#include <SDL3/SDL.h>

#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <utility>
#include <chrono>
#include <mutex>
#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {

class Presenter;
class RenderContext;

struct SurfaceCapabilities {
	vk::SurfaceCapabilitiesKHR        capabilities {};
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR>   present_modes;
};

struct WindowLoopState {
	SDL_Event        event {};
	bool             need_exit = false;
	std::atomic_bool paused    = false;
	std::atomic<uint64_t> presented_frames {0};
	// Present-to-present intervals (ms) since the last Status line, for frame-time percentiles
	// (1-second averages hide hitches; reviews asked for p95 / worst frame).
	std::mutex                              frame_times_mutex;
	std::vector<float>                      frame_times_ms;
	std::chrono::steady_clock::time_point   last_present {};
	void RecordPresent() {
		const auto now = std::chrono::steady_clock::now();
		std::lock_guard lock(frame_times_mutex);
		if (last_present.time_since_epoch().count() != 0 && frame_times_ms.size() < 4096) {
			frame_times_ms.push_back(
			    std::chrono::duration<float, std::milli>(now - last_present).count());
		}
		last_present = now;
	}
	std::vector<float> TakeFrameTimes() {
		std::lock_guard lock(frame_times_mutex);
		return std::exchange(frame_times_ms, {});
	}
};

struct WindowContext {
	WindowContext();
	~WindowContext();
	KYTY_CLASS_NO_COPY(WindowContext);

	[[nodiscard]] static vk::PhysicalDeviceVulkan11Features RequiredVulkan11Features() noexcept;
	[[nodiscard]] static vk::PhysicalDeviceVulkan12Features RequiredVulkan12Features() noexcept;
	[[nodiscard]] static vk::PhysicalDeviceVulkan13Features RequiredVulkan13Features() noexcept;
	[[nodiscard]] static uint32_t InitialWindowFlags(bool fullscreen) noexcept;
	void                                                    CreateVulkan();
	void                                                    RecreateSurface();
	void                                                    RefreshSurfaceCapabilities();
	void                                                    UpdateIcon();
	void UpdateTitle(uint64_t frame_num, double current_fps);
	void                                                    Resize(int width, int height);
	void ProcessWindowEvent(const SDL_WindowEvent& event);
	void ProcessDisplayEvent(const SDL_DisplayEvent& event);
	void ProcessEvent(double time_seconds);
	void Run();

	GraphicContext                 graphic_ctx;
	SDL_Window*                    window        = nullptr;
	vk::SurfaceKHR                 surface       = nullptr;
	SurfaceCapabilities            surface_capabilities;
	std::unique_ptr<RenderContext> render_context;
	std::unique_ptr<Presenter>     presenter;
	WindowLoopState                loop;

	Common::Mutex mutex;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_
