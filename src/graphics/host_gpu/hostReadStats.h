#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_HOSTREADSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_HOSTREADSTATS_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <utility>

namespace Libs::Graphics {

// KYTY_PROBE_FAULT_HISTOGRAM=1 counters, printed on the FaultHistogram line.
// - HostReadProbe: wall time the command processor spends reading GPU-written guest memory on the
//   host (DrawIndirect arguments and counts, dispatch arguments, predication results). Such a read
//   faults into HandleFault -> BufferCache::ReadMemory -> DownloadBufferMemory, which waits for the
//   GPU work that produced the bytes: one pipeline drain per read. ng-70 (2026-10-14) spent ~25 % of
//   wall time in download waits; these counters say which reads pay it.
// - SyncProbe: CPU-write synchronizations per interval (SyncCpuWrites executions and regions).
// Set once by the guest GPU thread at start; lets host_gpu probes ask "am I the GPU thread" without
// linking the guest-side GuestGpu (memory_tracker_tests links pageManager.cpp alone).
struct GpuThreadMark {
	inline static thread_local bool on = false;
};

struct ProbeEnabled {
	static bool Get() noexcept {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_PROBE_FAULT_HISTOGRAM");
			return value != nullptr && value[0] == '1';
		}();
		return enabled;
	}
};

struct HostReadCounters {
	std::atomic<uint64_t> calls {0};
	std::atomic<uint64_t> ns {0};
};

struct HostReadProbe {
	enum Site : uint32_t { DrawArgs = 0, DrawCount, DispatchArgs, Predication, SiteCount };
	// KYTY_CP_GPU_INDIRECT accounting: draws issued with GPU-side arguments, and the indirect draws
	// that stayed on the host path (first-index SGPR named, or any other reason; "mesh" is the
	// subset of "other" caused by NGG/mesh vertex shaders). GpuMeshDraws is the subset of GpuDraws
	// issued through the mesh translation prepass; FallbackMeshSrt counts mesh packets kept on the
	// host path because the mesh program's resource plan reads a batch SGPR (or is not compiled yet).
	// StaleMeshState counts later direct draws that fell back to m_num_instances (instance_count 0)
	// while a GPU mesh packet had left it at its pre-packet value: the host path persists the last
	// record's instance count and patched SGPRs, the GPU path cannot. Non-zero means the divergence
	// is observable.
	enum GpuIndirect : uint32_t { GpuDraws = 0, FallbackIndexOffset, FallbackOther, FallbackMesh, FallbackSrt,
	                              GpuMeshDraws, FallbackMeshSrt, StaleMeshState, GpuIndirectCount };
	struct Snapshot {
		std::array<std::pair<uint64_t, uint64_t>, SiteCount> sites {}; // calls, nanoseconds
		std::array<uint64_t, GpuIndirectCount>               gpu_indirect {};
		uint64_t                                             gpu_indirect_wait_ns = 0;
	};
	inline static std::array<HostReadCounters, SiteCount> s_sites {};
	inline static std::array<std::atomic<uint64_t>, GpuIndirectCount> s_gpu_indirect {};

	// Time the GPU-indirect path spends in PrepareHostRead (args, count, index range): it blocks when
	// the range intersects pending unbounded writes, which the `cp reads` counters no longer show.
	inline static std::atomic<uint64_t> s_gpu_indirect_wait_ns {0};
	static void AddGpuIndirectWait(uint64_t ns) noexcept {
		s_gpu_indirect_wait_ns.fetch_add(ns, std::memory_order_relaxed);
	}

	static void CountGpuIndirect(GpuIndirect kind) noexcept {
		s_gpu_indirect[kind].fetch_add(1, std::memory_order_relaxed);
	}

	static void Add(Site site, uint64_t ns) noexcept {
		s_sites[site].calls.fetch_add(1, std::memory_order_relaxed);
		s_sites[site].ns.fetch_add(ns, std::memory_order_relaxed);
	}
	static Snapshot Take() noexcept {
		Snapshot snapshot;
		for (uint32_t i = 0; i < SiteCount; ++i) {
			snapshot.sites[i] = {s_sites[i].calls.exchange(0, std::memory_order_relaxed),
			                     s_sites[i].ns.exchange(0, std::memory_order_relaxed)};
		}
		for (uint32_t i = 0; i < GpuIndirectCount; ++i) {
			snapshot.gpu_indirect[i] = s_gpu_indirect[i].exchange(0, std::memory_order_relaxed);
		}
		snapshot.gpu_indirect_wait_ns = s_gpu_indirect_wait_ns.exchange(0, std::memory_order_relaxed);
		return snapshot;
	}
	// Reads one dword per 4 KiB page of [vaddr, vaddr + size): a GPU-written range is published (and
	// its drain paid) here, in one place, before the caller walks it.
	static void Touch(uint64_t vaddr, uint64_t size) noexcept {
		if (size == 0) {
			return;
		}
		const uint64_t end = vaddr + size;
		for (uint64_t page = vaddr & ~uint64_t {0xfff}; page < end; page += 0x1000) {
			const auto address = page < vaddr ? vaddr : page;
			(void)*reinterpret_cast<const volatile uint32_t*>(address);
		}
	}
};

class HostReadScope {
public:
	explicit HostReadScope(HostReadProbe::Site site) noexcept
	    : m_site(site), m_enabled(ProbeEnabled::Get()),
	      m_begin(m_enabled ? std::chrono::steady_clock::now()
	                        : std::chrono::steady_clock::time_point {}) {}
	~HostReadScope() {
		if (!m_enabled) {
			return;
		}
		const auto elapsed = std::chrono::steady_clock::now() - m_begin;
		HostReadProbe::Add(m_site, static_cast<uint64_t>(
		                               std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed)
		                                   .count()));
	}
	HostReadScope(const HostReadScope&)            = delete;
	HostReadScope& operator=(const HostReadScope&) = delete;

private:
	HostReadProbe::Site                   m_site;
	bool                                  m_enabled;
	std::chrono::steady_clock::time_point m_begin;
};

// Wall-time budget of the GPU thread per interval: where the serial thread's time goes, including
// kernel time (perf records user space only on this machine). Categories overlap (Sync and Protect
// run inside Draw/Dispatch preparation); the reader subtracts.
struct BudgetCounters {
	std::atomic<uint64_t> calls {0};
	std::atomic<uint64_t> inclusive_ns {0};
	std::atomic<uint64_t> exclusive_ns {0};
};

struct BudgetProbe {
	enum Category : uint32_t { Wait = 0, Sync, Draw, Dispatch, Protect, Submit, CategoryCount };
	struct Entry {
		uint64_t calls = 0, inclusive_ns = 0, exclusive_ns = 0;
	};
	struct Snapshot {
		std::array<Entry, CategoryCount> categories {};
	};
	inline static std::array<BudgetCounters, CategoryCount> s_categories {};
	// Exclusive time: a scope's own time minus the time of scopes nested inside it on the same
	// thread (Codex review 2026-10-15: waits reach MasterSemaphore::Wait from inside sync/draw/
	// dispatch; inclusive numbers alone do not partition the thread's time).
	inline static thread_local uint64_t s_nested_ns = 0; // child time to subtract, per open scope
	static void Add(Category category, uint64_t inclusive_ns, uint64_t exclusive_ns) noexcept {
		s_categories[category].calls.fetch_add(1, std::memory_order_relaxed);
		s_categories[category].inclusive_ns.fetch_add(inclusive_ns, std::memory_order_relaxed);
		s_categories[category].exclusive_ns.fetch_add(exclusive_ns, std::memory_order_relaxed);
	}
	static Snapshot Take() noexcept {
		Snapshot snapshot;
		for (uint32_t i = 0; i < CategoryCount; ++i) {
			snapshot.categories[i] = {s_categories[i].calls.exchange(0, std::memory_order_relaxed),
			                          s_categories[i].inclusive_ns.exchange(0, std::memory_order_relaxed),
			                          s_categories[i].exclusive_ns.exchange(0, std::memory_order_relaxed)};
		}
		return snapshot;
	}
};

class BudgetScope {
public:
	// `active` lets a site that other threads also run (semaphore waits, protection updates) count
	// only the GPU thread.
	BudgetScope(BudgetProbe::Category category, bool active) noexcept
	    : m_category(category), m_enabled(active && ProbeEnabled::Get()),
	      m_begin(m_enabled ? std::chrono::steady_clock::now()
	                        : std::chrono::steady_clock::time_point {}) {
		if (m_enabled) {
			// Save the enclosing scope's child accumulator and start a fresh one for this scope.
			m_outer_nested_ns       = BudgetProbe::s_nested_ns;
			BudgetProbe::s_nested_ns = 0;
		}
	}
	~BudgetScope() {
		if (!m_enabled) {
			return;
		}
		const auto elapsed = std::chrono::steady_clock::now() - m_begin;
		const auto inclusive = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
		const auto children  = BudgetProbe::s_nested_ns;
		const auto exclusive = inclusive > children ? inclusive - children : 0;
		BudgetProbe::Add(m_category, inclusive, exclusive);
		// This scope's whole time is child time for the enclosing scope.
		BudgetProbe::s_nested_ns = m_outer_nested_ns + inclusive;
	}
	BudgetScope(const BudgetScope&)            = delete;
	BudgetScope& operator=(const BudgetScope&) = delete;

private:
	BudgetProbe::Category                 m_category;
	bool                                  m_enabled;
	std::chrono::steady_clock::time_point m_begin;
	uint64_t                              m_outer_nested_ns = 0;
};

// CPU time (user, system) of the calling thread and of the whole process, in milliseconds, read
// from /proc; zero on platforms without it.
struct CpuTimes {
	uint64_t thread_user_ms = 0, thread_sys_ms = 0, process_user_ms = 0, process_sys_ms = 0;
	static CpuTimes Read() noexcept;
};

struct SyncProbe {
	struct Snapshot {
		uint64_t calls = 0, regions = 0;
	};
	inline static std::atomic<uint64_t> s_calls {0};
	inline static std::atomic<uint64_t> s_regions {0};
	static void     NoteCall() noexcept { s_calls.fetch_add(1, std::memory_order_relaxed); }
	static void     NoteRegion() noexcept { s_regions.fetch_add(1, std::memory_order_relaxed); }
	static Snapshot Take() noexcept {
		return {s_calls.exchange(0, std::memory_order_relaxed),
		        s_regions.exchange(0, std::memory_order_relaxed)};
	}
};

// KYTY_HOST_RESIDENT=1: vertex/index bindings read straight from imported guest memory.
// Fallbacks are bindings that wanted the host path but took the VRAM-copy path (GPU-modified
// range, not direct memory, chunk crossing, 8-bit index widening, unaligned index offset).
struct HostResidentProbe {
	struct Snapshot {
		uint64_t binds = 0, bytes = 0, fallbacks = 0;
	};
	inline static std::atomic<uint64_t> s_binds {0};
	inline static std::atomic<uint64_t> s_bytes {0};
	inline static std::atomic<uint64_t> s_fallbacks {0};
	static void NoteBind(uint64_t bytes) noexcept {
		s_binds.fetch_add(1, std::memory_order_relaxed);
		s_bytes.fetch_add(bytes, std::memory_order_relaxed);
	}
	static void     NoteFallback() noexcept { s_fallbacks.fetch_add(1, std::memory_order_relaxed); }
	static Snapshot Take() noexcept {
		return {s_binds.exchange(0, std::memory_order_relaxed),
		        s_bytes.exchange(0, std::memory_order_relaxed),
		        s_fallbacks.exchange(0, std::memory_order_relaxed)};
	}
};


// KYTY_LAZY_BARRIERS counters (counted unconditionally; printed on the FaultHistogram line).
// Per kind: `emitted` = barriers the hazard tracker recorded, `old` = barriers the pre-tracker code
// would have recorded at that site (counted only where the tracker replaced a site), so
// elided = old - emitted (negative = the tracker added a barrier the old site did not have).
// Flush kinds have no old site: they are the places where deferred hazards were finally emitted.
struct BarrierProbe {
	enum Kind : uint32_t {
		DispatchDirect = 0, DispatchIndirect, DrawPost, DrawIndirectArgs,
		FlushEndRendering, FlushForeign, FlushDraw, FlushSubmit, KindCount
	};
	struct Snapshot {
		std::array<uint64_t, KindCount> emitted {}, old {};
	};
	inline static std::array<std::atomic<uint64_t>, KindCount> s_emitted {}, s_old {};
	inline static std::array<std::atomic<uint64_t>, KindCount> s_total_emitted {}, s_total_old {};
	static void NoteEmitted(Kind kind, uint64_t n = 1) noexcept {
		s_emitted[kind].fetch_add(n, std::memory_order_relaxed);
		s_total_emitted[kind].fetch_add(n, std::memory_order_relaxed);
	}
	static void NoteOld(Kind kind, uint64_t n = 1) noexcept {
		s_old[kind].fetch_add(n, std::memory_order_relaxed);
		s_total_old[kind].fetch_add(n, std::memory_order_relaxed);
	}
	static Snapshot Take() noexcept {
		Snapshot snapshot;
		for (uint32_t i = 0; i < KindCount; ++i) {
			snapshot.emitted[i] = s_emitted[i].exchange(0, std::memory_order_relaxed);
			snapshot.old[i]     = s_old[i].exchange(0, std::memory_order_relaxed);
		}
		return snapshot;
	}
	// Never reset by Take(); for tests.
	static Snapshot Total() noexcept {
		Snapshot snapshot;
		for (uint32_t i = 0; i < KindCount; ++i) {
			snapshot.emitted[i] = s_total_emitted[i].load(std::memory_order_relaxed);
			snapshot.old[i]     = s_total_old[i].load(std::memory_order_relaxed);
		}
		return snapshot;
	}
	static const char* Name(uint32_t kind) noexcept {
		static constexpr const char* names[KindCount] = {
		    "dispatch", "dispatch-indirect", "draw-post", "draw-indirect-args",
		    "flush-end-rendering", "flush-foreign", "flush-draw", "flush-submit"};
		return names[kind];
	}
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_HOSTREADSTATS_H_
