#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SUBMITAHEAD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SUBMITAHEAD_H_

#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics {

// KYTY_SUBMIT_AHEAD: submit the open command buffer early (no wait) so the GPU runs while the CPU
// records, and a later host read of GPU-written memory finds the producing work finished (BAR
// readback, no drain) or waits only for the tail. Without it a waited readback submits and drains
// everything recorded since the last submit (ng-155: ~10 ms per draw-args wait; ng-159 with a
// 2 ms timer: 3.6 ms, 3.017 -> 4.118 fps).
//   unset / 0     off (default)
//   1 or t        timer: before a work packet once the open command buffer is older than
//                 KYTY_SUBMIT_AHEAD_MS (default 4 ms; while a render pass is open, 3x that)
//   KYTY_SUBMIT_AHEAD_DISPATCH_MS (optional, default off): a shorter threshold once the open
//                 command buffer holds a dispatch (GPU-built data the host reads back comes from
//                 dispatches: arguments, specialization values).
// The learned-producer trigger of 806f2b3 ("p") was dropped: in ng-157 it issued ~1330 submits per
// interval to turn ~25 waited readbacks into BAR reads, and lost fps against the timer alone.
struct SubmitAheadConfig {
	uint64_t timer_ns          = 0;
	uint64_t dispatch_timer_ns = 0;

	[[nodiscard]] bool Enabled() const { return timer_ns != 0; }

	// Milliseconds in (0, 1000) -> nanoseconds, else `fallback`.
	[[nodiscard]] static uint64_t ParseMs(const char* ms, uint64_t fallback) {
		if (ms == nullptr || ms[0] == '\0') {
			return fallback;
		}
		char*        end   = nullptr;
		const double value = std::strtod(ms, &end);
		if (end == ms || *end != '\0' || !(value > 0.0 && value < 1000.0)) {
			return fallback;
		}
		return static_cast<uint64_t>(value * 1e6);
	}

	[[nodiscard]] static SubmitAheadConfig Parse(const char* mode, const char* ms,
	                                             const char* dispatch_ms = nullptr) {
		SubmitAheadConfig config;
		if (mode == nullptr || (mode[0] != '1' && mode[0] != 't') || mode[1] != '\0') {
			return config;
		}
		config.timer_ns          = ParseMs(ms, 4000000u);
		config.dispatch_timer_ns = ParseMs(dispatch_ms, 0);
		return config;
	}

	[[nodiscard]] static const SubmitAheadConfig& Get() {
		static const auto config =
		    Parse(std::getenv("KYTY_SUBMIT_AHEAD"), std::getenv("KYTY_SUBMIT_AHEAD_MS"),
		          std::getenv("KYTY_SUBMIT_AHEAD_DISPATCH_MS"));
		return config;
	}
};

// Per command processor: decides, before each PM4 packet, whether to submit the open command buffer.
class SubmitAheadTrigger {
public:
	// Register: SET_*_REG; Benign: NOPs, index state, CE/DE sync; Draw / Dispatch: GPU work;
	// Other: packets with side effects (events, waits, DMA, ...). Submits happen only before Draw,
	// Dispatch and Other packets.
	enum class Packet { Register, Benign, Draw, Dispatch, Other };
	enum class Reason { None, Timer, DispatchTimer };

	// `tick`: the open command buffer; `age_ns`: time since it began; `rendering`: a render pass is
	// open (submitting would end it).
	[[nodiscard]] Reason Before(Packet packet, uint64_t tick, uint64_t age_ns, bool rendering,
	                            const SubmitAheadConfig& config) {
		if (tick != m_tick) {
			m_tick     = tick;
			m_work     = 0;
			m_dispatch = false;
		}
		if (packet == Packet::Register || packet == Packet::Benign) {
			return Reason::None;
		}
		auto reason = Reason::None;
		if (m_work != 0 && config.timer_ns != 0) {
			const bool dispatch  = m_dispatch && config.dispatch_timer_ns != 0 &&
			                       config.dispatch_timer_ns < config.timer_ns;
			const auto threshold = dispatch ? config.dispatch_timer_ns : config.timer_ns;
			if (age_ns >= threshold && (!rendering || age_ns >= 3 * threshold)) {
				reason = dispatch ? Reason::DispatchTimer : Reason::Timer;
			}
		}
		if (reason == Reason::None) {
			Recorded(packet);
		}
		return reason;
	}

	// The caller submitted; the packet goes into the new command buffer `tick`.
	void Submitted(Packet packet, uint64_t tick) {
		m_tick     = tick;
		m_work     = 0;
		m_dispatch = false;
		Recorded(packet);
	}

private:
	void Recorded(Packet packet) {
		m_work++;
		m_dispatch = m_dispatch || packet == Packet::Dispatch;
	}

	uint64_t m_tick     = 0;
	uint32_t m_work     = 0;     // work packets recorded into m_tick
	bool     m_dispatch = false; // one of them was a dispatch
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SUBMITAHEAD_H_
