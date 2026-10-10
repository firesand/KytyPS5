#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Renderer event counters, printed as per-frame averages every two seconds when
// KYTY_PERF_STATS is set. They answer "what does a frame cost" without a profiler attached.
namespace Libs::Graphics::PerfStats {

struct Counters {
	std::atomic_uint64_t draws {0};
	std::atomic_uint64_t dispatches {0};
	std::atomic_uint64_t program_gets {0};
	std::atomic_uint64_t buffer_binds {0};
	std::atomic_uint64_t buffer_scans {0};
	std::atomic_uint64_t bda_walks {0};
	std::atomic_uint64_t submits {0};
	std::atomic_uint64_t gpu_waits {0};
	std::atomic_uint64_t downloads {0};
	std::atomic_uint64_t faults {0};
	std::atomic_uint64_t unmaps {0};
};

inline Counters g_counters;

inline bool Enabled() noexcept {
	static const bool enabled = std::getenv("KYTY_PERF_STATS") != nullptr;
	return enabled;
}

inline void Count(std::atomic_uint64_t& counter) noexcept {
	if (Enabled()) {
		counter.fetch_add(1, std::memory_order_relaxed);
	}
}

// Call once per guest flip.
inline void Frame() noexcept {
	if (!Enabled()) {
		return;
	}
	static uint64_t frames = 0;
	static auto     last   = std::chrono::steady_clock::now();
	frames++;
	const auto now     = std::chrono::steady_clock::now();
	const auto elapsed = std::chrono::duration<double>(now - last).count();
	if (elapsed < 2.0) {
		return;
	}
	const auto take = [](std::atomic_uint64_t& counter) {
		return static_cast<double>(counter.exchange(0, std::memory_order_relaxed));
	};
	const auto per_frame = 1.0 / static_cast<double>(frames);
	std::printf("PerfStats: fps=%.1f | per frame: draws=%.0f dispatches=%.0f program_gets=%.0f "
	            "buffer_binds=%.0f buffer_scans=%.0f bda_walks=%.0f submits=%.1f gpu_waits=%.1f "
	            "downloads=%.1f faults=%.1f unmaps=%.1f\n",
	            static_cast<double>(frames) / elapsed, take(g_counters.draws) * per_frame,
	            take(g_counters.dispatches) * per_frame, take(g_counters.program_gets) * per_frame,
	            take(g_counters.buffer_binds) * per_frame, take(g_counters.buffer_scans) * per_frame,
	            take(g_counters.bda_walks) * per_frame, take(g_counters.submits) * per_frame,
	            take(g_counters.gpu_waits) * per_frame, take(g_counters.downloads) * per_frame,
	            take(g_counters.faults) * per_frame, take(g_counters.unmaps) * per_frame);
	std::fflush(stdout);
	frames = 0;
	last   = now;
}

} // namespace Libs::Graphics::PerfStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_
