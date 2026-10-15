#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_

#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"

#include <array>
#include <atomic>
#include <string>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace Libs::Graphics {

// Opt-in (KYTY_GPU_TIMESTAMPS=1) per-shader GPU time attribution. Every command buffer gets a
// slot of a timestamp query pool: query 0/1 bracket the whole command buffer (the tick span) and
// each recorded draw/dispatch writes a bottom-of-pipe pair around itself. Once the tick has
// completed, the pairs are read back and summed per {shader hash, kind}. A pair measures from the
// completion of everything earlier to the completion of this op, so the sum of the pairs is
// bounded by the span and the remainder is "unattributed" (barriers, copies, idle waits).
class GpuTimestamps {
public:
	enum class Kind : uint8_t {
		Draw = 0,
		Dispatch,
		DispatchIndirect,
		BarrierGlobal,
		BarrierBuffer,
		BarrierImage,
		CopyUpload,
		CopyImage,
		Blit,
		Fill,
		Clear,
		CacheDispatch,
		RenderPass,
		// Sub-kinds: where a cache dispatch / copy came from (printed as kind/sub).
		CacheDispatchDetile,
		CacheDispatchTile,
		CacheDispatchFault,
		CacheDispatchColorClear,
		CacheDispatchMeshPrepass,
		CopyImageRefresh,
		CopyImageUpload,
		CopyImageDownload,
		CopyImagePreserve,
		CopyImageReadback,
		CopyImageImage,
		CopyImageResolve,
		CopyImageViaBuffer,
		CopyImageDepth,
		CopyUploadStaging,
		CopyUploadDownload,
		CopyUploadTexture,
		CopyUploadStream,
		Count
	};
	// Kinds that carry a shader hash and appear in the "top" list.
	static bool HasShader(Kind kind) noexcept {
		return kind == Kind::Draw || kind == Kind::Dispatch || kind == Kind::DispatchIndirect;
	}
	static const char* KindName(Kind kind);
	// The instance currently recording into this command buffer, or null.
	static GpuTimestamps* ForCommand(vk::CommandBuffer command) noexcept;

	static constexpr uint32_t SlotCount      = 32;
	static constexpr uint32_t QueriesPerSlot = 32768;
	static constexpr uint32_t TopEntries     = 12;

	struct Entry {
		uint64_t ns    = 0;
		uint64_t count = 0;
		uint64_t bytes = 0;
	};
	// Calling context for copies that are issued deep inside Image::Upload/Download: set by the
	// texture cache entry points (thread-local, only touched when profiling is enabled).
	enum class Origin : uint8_t { None, Refresh, Download, Preserve };
	static Origin& CurrentOrigin() noexcept;
	struct Snapshot {
		uint64_t ticks           = 0;
		uint64_t span_ns         = 0;
		uint64_t ops             = 0;
		uint64_t attributed_ns   = 0;
		uint64_t unattributed_ns = 0;
		uint64_t overflow        = 0; // ops not tagged because the tick's range was full
		uint64_t no_slot         = 0; // ticks recorded without a slot (all slots in flight)
		uint64_t violations      = 0; // end timestamp before begin timestamp
		std::array<Entry, static_cast<size_t>(Kind::Count)> kinds {};
		// key: kind, primary hash (PS for draws, CS for dispatches), secondary hash (VS)
		std::map<std::tuple<uint8_t, uint64_t, uint64_t>, Entry> shaders;
	};

	[[nodiscard]] static bool Enabled() noexcept;

	explicit GpuTimestamps(GraphicContext& graphics);
	~GpuTimestamps();
	KYTY_CLASS_NO_COPY(GpuTimestamps);

	[[nodiscard]] bool Valid() const noexcept { return m_pool != nullptr; }

	// Command buffer lifecycle, called by the scheduler. `completed_tick` is the newest tick the
	// GPU is known to have finished.
	void BeginCommand(vk::CommandBuffer command, uint64_t tick, uint64_t completed_tick);
	void EndCommand(vk::CommandBuffer command);

	// Per-op bracket, called through GpuTimestampScope.
	// Returns true when the op is being timed (the caller must then call EndOp).
	bool BeginOp(vk::CommandBuffer command, Kind kind, uint64_t hash, uint64_t hash2);
	void SetOpBytes(uint64_t bytes);
	void EndOp(vk::CommandBuffer command);

	// Read back every completed slot into the accumulators.
	void Harvest(uint64_t completed_tick);
	// Harvest, then return and reset the accumulators.
	Snapshot Take(uint64_t completed_tick);
	// Harvest and, once per interval, LOGF one summary line and reset the accumulators.
	void Dump(uint64_t completed_tick, double interval_seconds = 2.0);

	static std::string Format(const Snapshot& snapshot);

private:
	enum class SlotState { Free, Recording, Pending };
	struct Op {
		uint64_t hash;
		uint64_t hash2;
		uint64_t bytes;
		uint32_t query;
		Kind     kind;
	};
	struct Slot {
		SlotState       state    = SlotState::Free;
		uint64_t        tick     = 0;
		uint32_t        used     = 0;
		uint64_t        overflow = 0;
		std::vector<Op> ops;
	};

	void HarvestLocked(uint64_t completed_tick);

	GraphicContext&                       m_graphics;
	vk::QueryPool                         m_pool = nullptr;
	double                                m_period_ns = 1.0;
	uint64_t                              m_mask      = ~uint64_t {0};
	std::mutex                            m_mutex;
	std::array<Slot, SlotCount>           m_slots;
	int                                   m_current  = -1;
	uint32_t                              m_depth    = 0;
	std::atomic<VkCommandBuffer>          m_recording {nullptr};
	bool                                  m_op_open  = false;
	Snapshot                              m_acc;
	std::chrono::steady_clock::time_point m_last_dump {};
	std::vector<uint64_t>                 m_results;
};

class CommandScheduler;

// Total bytes of a vk::BufferCopy-like range (only evaluate it when a scope is active).
template <class Range>
inline uint64_t SumCopySizes(const Range& copies) {
	uint64_t total = 0;
	for (const auto& copy: copies) total += copy.size;
	return total;
}

// RAII bracket for one draw/dispatch. Costs one static bool check when profiling is off.
class GpuTimestampScope {
public:
	GpuTimestampScope(CommandScheduler& scheduler, vk::CommandBuffer command,
	                  GpuTimestamps::Kind kind, uint64_t hash, uint64_t hash2 = 0) noexcept;
	// Command-buffer-only form for sites without a scheduler at hand (barriers, copies, ...).
	GpuTimestampScope(vk::CommandBuffer command, GpuTimestamps::Kind kind, uint64_t hash = 0,
	                  uint64_t hash2 = 0) noexcept;
	~GpuTimestampScope();
	[[nodiscard]] bool Active() const noexcept { return m_timestamps != nullptr; }
	void SetBytes(uint64_t bytes) const {
		if (m_timestamps != nullptr) m_timestamps->SetOpBytes(bytes);
	}
	GpuTimestampScope(const GpuTimestampScope&)            = delete;
	GpuTimestampScope& operator=(const GpuTimestampScope&) = delete;

private:
	GpuTimestamps*    m_timestamps = nullptr;
	vk::CommandBuffer m_command;
};

} // namespace Libs::Graphics

namespace Libs::Graphics {
// Sets the thread's copy origin for the lifetime of the object (no-op when profiling is off).
class GpuTimestampOrigin {
public:
	explicit GpuTimestampOrigin(GpuTimestamps::Origin origin) noexcept {
		if (GpuTimestamps::Enabled()) {
			m_previous                     = GpuTimestamps::CurrentOrigin();
			GpuTimestamps::CurrentOrigin() = origin;
			m_active                       = true;
		}
	}
	~GpuTimestampOrigin() {
		if (m_active) GpuTimestamps::CurrentOrigin() = m_previous;
	}
	GpuTimestampOrigin(const GpuTimestampOrigin&)            = delete;
	GpuTimestampOrigin& operator=(const GpuTimestampOrigin&) = delete;

private:
	GpuTimestamps::Origin m_previous = GpuTimestamps::Origin::None;
	bool                  m_active   = false;
};
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_
