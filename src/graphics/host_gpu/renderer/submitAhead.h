#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SUBMITAHEAD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SUBMITAHEAD_H_

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

namespace Libs::Graphics {

// KYTY_SUBMIT_AHEAD: submit the open command buffer early (no wait) so that a later host read of
// GPU-written memory finds the producing work submitted, ideally finished (BAR readback, no drain).
// Today a waited readback submits and drains everything recorded since the last submit.
//   unset / 0  off (default)
//   1          both triggers
//   p          learned producers only: after a dispatch whose written pages a waited readback hit
//   t          timer only: before a work packet once the open command buffer is older than
//              KYTY_SUBMIT_AHEAD_MS (default 4 ms; while a render pass is open, 3x that)
struct SubmitAheadConfig {
	bool     producers = false;
	uint64_t timer_ns  = 0;

	[[nodiscard]] bool Enabled() const { return producers || timer_ns != 0; }

	[[nodiscard]] static SubmitAheadConfig Parse(const char* mode, const char* ms) {
		SubmitAheadConfig config;
		if (mode == nullptr || mode[0] == '\0' || mode[0] == '0') {
			return config;
		}
		const bool producers = mode[0] == '1' || mode[0] == 'p';
		const bool timer     = mode[0] == '1' || mode[0] == 't';
		if (!producers && !timer) {
			return config;
		}
		config.producers = producers;
		if (timer) {
			double milliseconds = 4.0;
			if (ms != nullptr && ms[0] != '\0') {
				char*        end   = nullptr;
				const double value = std::strtod(ms, &end);
				if (end != ms && *end == '\0' && value > 0.0 && value < 1000.0) {
					milliseconds = value;
				}
			}
			config.timer_ns = static_cast<uint64_t>(milliseconds * 1e6);
		}
		return config;
	}

	[[nodiscard]] static const SubmitAheadConfig& Get() {
		static const auto config =
		    Parse(std::getenv("KYTY_SUBMIT_AHEAD"), std::getenv("KYTY_SUBMIT_AHEAD_MS"));
		return config;
	}
};

// Learns which dispatches produce data the host later reads (GPU thread). Pages are the caller's
// tracking pages (BufferCache: CACHING_PAGEBITS).
class SubmitAheadLearner {
public:
	static constexpr uint64_t kMaxWritePages   = 64;      // larger written ranges (heaps) are ignored
	static constexpr size_t   kMaxTrackedPages = 1u << 16; // the page map is cleared beyond this
	static constexpr uint32_t kIdleLimit       = 256;     // submit-aheads without a readback of its pages

	// `writer` (a compute shader hash, 0 = none) recorded a write to pages [first, end).
	void NoteWrite(uint64_t first, uint64_t end, uint64_t writer) {
		if (writer == 0 || end <= first || end - first > kMaxWritePages) {
			return;
		}
		if (m_page_writer.size() + (end - first) > kMaxTrackedPages) {
			m_page_writer.clear();
		}
		for (auto page = first; page < end; ++page) {
			m_page_writer[page] = writer;
		}
	}

	// A synchronous readback touched pages [first, end); `waited`: it was served by a copy (drain),
	// not over BAR. Known producers of those pages are marked as used; when it waited, their writers
	// become producers. Returns the number of new producers (`learned`: the last one).
	uint32_t NoteReadback(uint64_t first, uint64_t end, bool waited, uint64_t* learned = nullptr) {
		uint32_t added = 0;
		end            = std::min(end, first + kMaxWritePages);
		for (auto page = first; page < end; ++page) {
			const auto writer = m_page_writer.find(page);
			if (writer == m_page_writer.end()) {
				continue;
			}
			if (auto producer = m_producers.find(writer->second); producer != m_producers.end()) {
				producer->second = 0;
			} else if (waited) {
				m_producers.emplace(writer->second, 0u);
				added++;
				if (learned != nullptr) {
					*learned = writer->second;
				}
			}
		}
		return added;
	}

	[[nodiscard]] bool IsProducer(uint64_t writer) const {
		return writer != 0 && m_producers.contains(writer);
	}

	// A submit-ahead was issued after `writer`; a producer whose pages were not read back for
	// kIdleLimit of them is forgotten (it is relearned by the next waited readback).
	void NoteSubmitAhead(uint64_t writer) {
		if (auto producer = m_producers.find(writer);
		    producer != m_producers.end() && ++producer->second > kIdleLimit) {
			m_producers.erase(producer);
		}
	}

	[[nodiscard]] size_t ProducerCount() const { return m_producers.size(); }

private:
	std::unordered_map<uint64_t, uint64_t> m_page_writer; // page -> last writer
	std::unordered_map<uint64_t, uint32_t> m_producers;   // writer -> idle submit-aheads
};

// Per command processor: decides, before each PM4 packet, whether to submit the open command buffer.
class SubmitAheadTrigger {
public:
	// Register: SET_*_REG; Benign: NOPs, index state, CE/DE sync; Work: anything that records GPU
	// work or has side effects. Submits happen only before Work packets.
	enum class Packet { Register, Benign, Work };
	enum class Reason { None, Producer, Timer };

	// A learned producer was recorded in command buffer `tick`.
	void ProducerRecorded(uint64_t tick, uint64_t writer) {
		m_producer      = writer;
		m_producer_tick = tick;
	}

	// `tick`: the open command buffer; `age_ns`: time since it began; `rendering`: a render pass is
	// open (submitting would end it).
	[[nodiscard]] Reason Before(Packet packet, uint64_t tick, uint64_t age_ns, bool rendering,
	                            const SubmitAheadConfig& config) {
		if (tick != m_tick) {
			m_tick = tick;
			m_work = 0;
		}
		if (packet != Packet::Work) {
			return Reason::None;
		}
		auto reason = Reason::None;
		if (m_work != 0) {
			if (config.producers && m_producer != 0 && m_producer_tick == tick) {
				reason = Reason::Producer;
			} else if (config.timer_ns != 0 && age_ns >= config.timer_ns &&
			           (!rendering || age_ns >= 3 * config.timer_ns)) {
				reason = Reason::Timer;
			}
		}
		if (reason == Reason::None) {
			m_work++;
		}
		return reason;
	}

	// The caller submitted for `reason` (!= None); the packet goes into command buffer `tick`.
	// Returns the producer the submit was for (0 for the timer).
	uint64_t Submitted(Reason reason, uint64_t tick) {
		const auto producer = reason == Reason::Producer ? m_producer : 0;
		m_producer      = 0;
		m_producer_tick = 0;
		m_tick          = tick;
		m_work          = 1;
		return producer;
	}

private:
	uint64_t m_tick          = 0;
	uint32_t m_work          = 0; // work packets recorded into m_tick
	uint64_t m_producer      = 0;
	uint64_t m_producer_tick = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SUBMITAHEAD_H_
