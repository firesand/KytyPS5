#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUDIAGNOSTICS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUDIAGNOSTICS_H_

#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace Libs::Graphics {

// CPU records of commands actually emitted to Vulkan, with explicitly named
// preparation/snapshot markers. Separate from CommandBuffer::SetDebugInfo.
struct GpuDiagnosticEvent {
	const char* kind = "unknown";
	uint64_t guest_submit = 0;
	uint64_t pipeline = 0;
	std::array<uint64_t, 6> args {};
	struct Shader {
		uint64_t hash = 0;
		uint64_t variant = 0;
		uint32_t stage = 0;
	};
	std::array<Shader, 4> shaders {};
};

class GpuDiagnostics {
public:
	static bool Enabled() {
		static const bool enabled = Flag("KYTY_GPU_DIAGNOSTICS");
		return enabled;
	}
	static bool Serialize() {
		static const bool enabled = Flag("KYTY_GPU_DIAGNOSTICS_SYNC");
		return Enabled() && enabled;
	}
	static bool SerializePreparation() {
		static const bool enabled = Flag("KYTY_GPU_DIAGNOSTICS_SYNC_PREP");
		return Serialize() && enabled;
	}
	static bool DumpShaders() {
		static const bool enabled = Flag("KYTY_GPU_DIAGNOSTICS_DUMP_SHADERS");
		return Enabled() && enabled;
	}
	template <typename Handle> static uint64_t HandleValue(Handle handle) {
		static_assert(sizeof(handle) <= sizeof(uint64_t));
		uint64_t value = 0;
		std::memcpy(&value, &handle, sizeof(handle));
		return value;
	}
	static void Record(const void* scheduler, uint64_t semaphore, uint64_t tick,
	                   const GpuDiagnosticEvent& event) {
		if (!Enabled()) return;
		auto& state = State();
		std::lock_guard lock(state.mutex);
		const auto sequence = ++state.sequence;
		state.ring[(sequence - 1) % Capacity] = {sequence, scheduler, semaphore, tick, event};
	}
	static void DumpStall(uint64_t semaphore, uint64_t target, uint64_t completed,
	                      uint64_t next_tick) {
		if (!Enabled()) return;
		auto& state = State();
		std::lock_guard lock(state.mutex);
		const auto first = state.sequence > Capacity ? state.sequence - Capacity + 1 : 1;
		std::fprintf(state.output,
		    "GPU_DIAGNOSTICS stall semaphore=0x%016" PRIx64 " target=%" PRIu64
		    " completed=%" PRIu64 " first_incomplete=%" PRIu64
		    " next_tick=%" PRIu64 " retained_seq=%" PRIu64
		    "..%" PRIu64 " capacity=%zu sync=%u\n",
		    semaphore, target, completed, completed + 1, next_tick, first, state.sequence, Capacity,
		    Serialize() ? 1u : 0u);
		// Include the last completed tick and every incomplete tick of this timeline.
		// The retained range makes any ring truncation explicit in the report.
		for (auto sequence = first; sequence <= state.sequence; ++sequence) {
			const auto& item = state.ring[(sequence - 1) % Capacity];
			if (item.semaphore != semaphore || item.tick < completed) continue;
			const auto& event = item.event;
			std::fprintf(state.output,
			    "GPU_DIAGNOSTICS seq=%" PRIu64 " scheduler=%p semaphore=0x%016" PRIx64
			    " tick=%" PRIu64 " kind=%s guest_submit=%" PRIu64
			    " pipeline=0x%016" PRIx64 " args=",
			    item.sequence, item.scheduler, item.semaphore, item.tick, event.kind,
			    event.guest_submit, event.pipeline);
			for (auto arg : event.args) std::fprintf(state.output, "0x%016" PRIx64 ",", arg);
			for (const auto& shader : event.shaders) {
				if (shader.stage != 0) {
					std::fprintf(state.output, " shader(stage=%u,hash=0x%016" PRIx64
					    ",variant=0x%016" PRIx64 ")", shader.stage, shader.hash, shader.variant);
				}
			}
			std::fputc('\n', state.output);
		}
		std::fflush(state.output);
	}

private:
	static constexpr size_t Capacity = 4096;
	struct Entry {
		uint64_t sequence = 0;
		const void* scheduler = nullptr;
		uint64_t semaphore = 0;
		uint64_t tick = 0;
		GpuDiagnosticEvent event;
	};
	struct Storage {
		std::mutex mutex;
		std::array<Entry, Capacity> ring {};
		uint64_t sequence = 0;
		FILE* output = stderr;
		Storage() {
			const auto* path = std::getenv("KYTY_GPU_DIAGNOSTICS_PATH");
			if (path != nullptr && *path != '\0') {
				if (auto* file = std::fopen(path, "a")) output = file;
				else std::fprintf(stderr, "GPU_DIAGNOSTICS cannot open %s; using stderr\n", path);
			}
			std::fprintf(output, "GPU_DIAGNOSTICS enabled capacity=%zu sync=%u sync_prep=%u\n",
			             Capacity, Serialize() ? 1u : 0u, SerializePreparation() ? 1u : 0u);
			std::fputs("GPU_DIAGNOSTICS args: draw=count,instances,first,vertex_offset_u32,"
			           "first_instance,index_guest_addr; draw-mesh=groups,instances,1,"
			           "guest_count,first_instance,index_guest_addr; dispatch=x,y,z,mode,"
			           "local_x,local_y_or_local_z_shift32; dispatch-indirect=guest_args,"
			           "buffer,offset,mode,local_x,local_y_or_local_z_shift32; "
			           "readback-copy=src_buffer,dst_buffer,src_offset,dst_offset,size,guest_addr; "
			           "submit=command_buffer,wait_count,signal_count,result,queue,0; "
			           "submit-wait=semaphore,value,stage_mask,index,0,0; "
			           "submit-signal=semaphore,value,index,0,0,0\n", output);
			std::fputs("GPU_DIAGNOSTICS preparation markers (CS 2ef44268a059254f only; "
			           "not guest dispatch completion): preparation-boundary=groups_x,y,z,mode,"
			           "buffer_count,image_count; preparation-wait-complete=submitted_tick,"
			           "completed_tick,next_tick,0,0,0; cs-buffer-descriptor=index,guest_addr,"
			           "stride,records,format,source_or_read_bit32_or_write_bit33; "
			           "cs-buffer-binding=index,descriptor_index,buffer,offset,range,guest_size; "
			           "cs-image-descriptor=index,dword_count,raw_dwords01,23,45,67; "
			           "cs-image-memory=index,guest_addr,size,metadata_addr,metadata_size,"
			           "metadata_kind\n", output);
			std::fflush(output);
		}
		~Storage() { if (output != stderr) std::fclose(output); }
	};
	static bool Flag(const char* name) {
		const auto* value = std::getenv(name);
		return value != nullptr && std::strcmp(value, "1") == 0;
	}
	static Storage& State() { static Storage state; return state; }
};

} // namespace Libs::Graphics

#endif
