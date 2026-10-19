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
	// subset of "other" caused by NGG/mesh vertex shaders).
	// Stage 2 (mesh through the pass-through task shader). FallbackMesh stays the total of mesh
	// packets kept on the host path for a mesh reason; one reason counter each:
	// - MeshGpuDraws: the subset of GpuDraws recorded with drawMeshTasksEXT through the task shader;
	// - FallbackMeshDenied: the mesh hash is not in KYTY_CP_GPU_INDIRECT_MESH_HASH (unset = none) or
	//   is in KYTY_CP_GPU_INDIRECT_DENY_HASH;
	// - FallbackMeshCountAddr: the packet names a count dword (the task shader ignores it);
	// - FallbackMeshPrim: the input primitive type is not supported by the task shader;
	// - FallbackDenied: a vertex-stage hash in KYTY_CP_GPU_INDIRECT_DENY_HASH (counted in "other" too);
	// - MeshClampedRecords / MeshEmptyRecords: argument records the task shader clamped to the
	//   device's mesh workgroup limits / found empty (GPU atomics, read back with a delay);
	// - ArgsWaitSkipped (lazy mode 1): GPU packets whose argument range intersected pending
	//   unbounded writes and skipped the waiting publish (ObtainBuffer then avoids the stream path);
	// - ArgsUnbounded / ArgsGpuBound / ArgsCpuOnly: who wrote a GPU-indirect packet's argument
	//   range at draw time: pending fault-tracked (unbounded) writes, published bound GPU writes, or
	//   neither (CPU data);
	// - ArgsPageReadback: ReadMemory calls that touched a page recent GPU-indirect packets took
	//   arguments from (first 32 logged as "GpuIndirect(args-page-readback)");
	// - Merge*: consecutive GPU mesh packets (b1, can N packets become one task dispatch?).
	//   MergeSame: same mesh/pixel hash, identical GS/PS user SGPR values, nothing but SH register
	//   writes and NOPs in between. MergeUserData: same hashes, user SGPR values differ.
	//   MergeBreakCtx: context/uconfig register writes in between. MergeBreakShader: other hashes.
	//   MergeBreakOp: another draw, dispatch or side-effect packet (event, wait, DMA, ...) in between;
	// - StaleInstances: a later draw that took its instance count from m_num_instances while a GPU
	//   packet had left it at the pre-packet value (the host path stores the last record's count);
	// - StaleSgpr: a later draw whose shader can read a GS user-data register that a GPU packet left
	//   at its pre-packet value (the host path would have patched it with the last record's offset).
	//   "Can read" = the slot is below the bound shader's user SGPR count (an upper bound).
	enum GpuIndirect : uint32_t { GpuDraws = 0, FallbackIndexOffset, FallbackOther, FallbackMesh, FallbackSrt,
	                              MeshGpuDraws, FallbackMeshDenied, FallbackMeshCountAddr, FallbackMeshPrim,
	                              FallbackDenied, MeshClampedRecords, MeshEmptyRecords, ArgsWaitSkipped,
	                              ArgsUnbounded, ArgsGpuBound, ArgsCpuOnly, ArgsPageReadback, MergeSame,
	                              MergeUserData, MergeBreakCtx, MergeBreakShader, MergeBreakOp,
	                              StaleInstances, StaleSgpr, GpuIndirectCount };
	// Short names, one per enumerator in order; the size is deduced, so adding an enumerator without
	// a name (or the reverse) fails to compile.
	static constexpr auto kGpuIndirectNames = std::to_array<const char*>(
	    {"gpu", "index_offset", "other", "mesh", "srt", "mesh-gpu", "mesh-denied", "mesh-count-addr",
	     "mesh-prim", "denied", "mesh-clamped", "mesh-empty", "args-wait-skipped",
	     "args-unbounded", "args-gpu-bound", "args-cpu", "args-page-readback", "merge-same", "merge-userdata",
	     "merge-break-ctx", "merge-break-shader", "merge-break-op",
	     "stale-instances", "stale-sgpr"});
	static_assert(kGpuIndirectNames.size() == GpuIndirectCount, "GpuIndirect enum / name table mismatch");
	struct Snapshot {
		std::array<std::pair<uint64_t, uint64_t>, SiteCount> sites {}; // calls, nanoseconds
		std::array<uint64_t, GpuIndirectCount>               gpu_indirect {};
		uint64_t                                             gpu_indirect_wait_ns = 0;
		uint64_t                                             gpu_indirect_args_wait_ns = 0;
	};
	inline static std::array<HostReadCounters, SiteCount> s_sites {};
	inline static std::array<std::atomic<uint64_t>, GpuIndirectCount> s_gpu_indirect {};

	// Time the GPU-indirect path spends in PrepareHostRead (args, count, index range): it blocks when
	// the range intersects pending unbounded writes, which the `cp reads` counters no longer show.
	inline static std::atomic<uint64_t> s_gpu_indirect_wait_ns {0};
	static void AddGpuIndirectWait(uint64_t ns) noexcept {
		s_gpu_indirect_wait_ns.fetch_add(ns, std::memory_order_relaxed);
	}
	// The part of it spent at the argument-record check in ExecutePreparedDraw (also in the total).
	inline static std::atomic<uint64_t> s_gpu_indirect_args_wait_ns {0};
	static void AddGpuIndirectArgsWait(uint64_t ns) noexcept {
		s_gpu_indirect_wait_ns.fetch_add(ns, std::memory_order_relaxed);
		s_gpu_indirect_args_wait_ns.fetch_add(ns, std::memory_order_relaxed);
	}

	static void CountGpuIndirect(GpuIndirect kind, uint64_t count = 1) noexcept {
		s_gpu_indirect[kind].fetch_add(count, std::memory_order_relaxed);
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
		snapshot.gpu_indirect_args_wait_ns =
		    s_gpu_indirect_args_wait_ns.exchange(0, std::memory_order_relaxed);
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

// Why the GPU thread waits for the GPU (KYTY_PROBE_FAULT_HISTOGRAM=1, printed as "wait reasons" on the FaultHistogram
// line): every MasterSemaphore::Wait on the GPU thread is charged to the reason the OUTERMOST enclosing
// WaitReasonScope declared, so a download wait reached from an indirect-argument read counts as draw-args rather
// than download. "other" = a wait below no declared reason. The budget's wait category is the total.
// BufferCache::ReadMemory runs on the GPU thread (SendCommandSync) and carries its caller's reason
// there: "download" is now only a ReadMemory caller that declares nothing. Readback callers:
// read-fault-guest / read-fault-gpu (a host read of a GPU-dirty page, HandleFault, by thread),
// write-fault-guest / write-fault-gpu (a host write into a GPU-dirty page: readback before the
// write), invalidate-guest / invalidate-gpu (explicit range invalidation: file reads into guest
// memory, unmap, texture slices), texture-readback (texture cache host check), backing-read
// (TryReadBufferBacking), dispatch-args-lds (indirect dispatch sizing its LDS buffer from GPU-built
// arguments). Flush waits: index-range (DrawIndex publishing an index range), args-check
// (GPU-indirect argument check, lazy modes 0/2).
struct WaitReasonProbe {
	enum Reason : uint32_t {
		Other = 0, DrawArgs, DrawCount, DispatchArgs, Predication, // the HostReadProbe sites, in the same order
		Download, WriteFaults, FaultAreas, StreamWrap, Gc, Finish, DescriptorPool,
		ReadFaultGuest, ReadFaultGpu, WriteFaultGuest, WriteFaultGpu, InvalidateGuest, InvalidateGpu,
		TextureReadback, BackingRead, DispatchArgsLds, IndexRange, ArgsCheck, ReasonCount
	};
	struct Snapshot {
		std::array<std::pair<uint64_t, uint64_t>, ReasonCount> reasons {}; // calls, nanoseconds
	};
	inline static thread_local uint32_t s_current = Other;
	inline static std::array<HostReadCounters, ReasonCount> s_reasons {};
	static void Add(uint32_t reason, uint64_t ns) noexcept {
		s_reasons[reason].calls.fetch_add(1, std::memory_order_relaxed);
		s_reasons[reason].ns.fetch_add(ns, std::memory_order_relaxed);
	}
	static Snapshot Take() noexcept {
		Snapshot snapshot;
		for (uint32_t i = 0; i < ReasonCount; ++i) {
			snapshot.reasons[i] = {s_reasons[i].calls.exchange(0, std::memory_order_relaxed),
			                       s_reasons[i].ns.exchange(0, std::memory_order_relaxed)};
		}
		return snapshot;
	}
	static constexpr auto kNames = std::to_array<const char*>(
	    {"other", "draw-args", "draw-count", "dispatch-args", "predication", "download", "write-faults",
	     "fault-areas", "stream-wrap", "gc", "finish", "descriptor-pool", "read-fault-guest",
	     "read-fault-gpu", "write-fault-guest", "write-fault-gpu", "invalidate-guest", "invalidate-gpu",
	     "texture-readback", "backing-read", "dispatch-args-lds", "index-range", "args-check"});
	static_assert(kNames.size() == ReasonCount, "WaitReasonProbe reason / name table mismatch");
	static const char* Name(uint32_t reason) noexcept { return kNames[reason]; }
	// The reason a wait below this point on the calling thread would be charged to, or `fallback`.
	static Reason CurrentOr(Reason fallback) noexcept {
		return s_current != Other ? static_cast<Reason>(s_current) : fallback;
	}
};

// KYTY_PROBE_FAULT_HISTOGRAM: synchronous readbacks (BufferCache::DownloadBufferMemory<false>) by the
// wait reason in force: served over BAR (completed GPU data, no drain), or by a copy that waits
// because the data was written into the open command buffer (CopyOpen: only an earlier submit could
// help), is in flight in a submitted one (CopyInFlight: GPU latency / backlog), or other (unmapped
// buffer, page without a write tick).
struct ReadbackProbe {
	enum Kind : uint32_t { Bar = 0, CopyOpen, CopyInFlight, CopyOther, KindCount };
	inline static std::array<std::array<std::atomic<uint64_t>, KindCount>, WaitReasonProbe::ReasonCount> s_counts {};
	static void Count(uint32_t reason, Kind kind) noexcept {
		s_counts[reason][kind].fetch_add(1, std::memory_order_relaxed);
	}
	using Snapshot = std::array<std::array<uint64_t, KindCount>, WaitReasonProbe::ReasonCount>;
	static Snapshot Take() noexcept {
		Snapshot snapshot {};
		for (uint32_t r = 0; r < WaitReasonProbe::ReasonCount; ++r) {
			for (uint32_t k = 0; k < KindCount; ++k) {
				snapshot[r][k] = s_counts[r][k].exchange(0, std::memory_order_relaxed);
			}
		}
		return snapshot;
	}
};

// KYTY_SUBMIT_AHEAD (submitAhead.h): early submits by trigger.
struct SubmitAheadProbe {
	enum Counter : uint32_t { TimerSubmits = 0, DispatchTimerSubmits, CounterCount };
	inline static std::array<std::atomic<uint64_t>, CounterCount> s_counts {};
	static void Count(Counter counter) noexcept { s_counts[counter].fetch_add(1, std::memory_order_relaxed); }
	static std::array<uint64_t, CounterCount> Take() noexcept {
		std::array<uint64_t, CounterCount> snapshot {};
		for (uint32_t i = 0; i < CounterCount; ++i) {
			snapshot[i] = s_counts[i].exchange(0, std::memory_order_relaxed);
		}
		return snapshot;
	}
};

// Declares why waits below this point happen; an enclosing scope's reason wins. Cheap enough for any path (one
// thread-local read and write), independent of the probe switch.
class WaitReasonScope {
public:
	explicit WaitReasonScope(WaitReasonProbe::Reason reason) noexcept: m_previous(WaitReasonProbe::s_current) {
		if (m_previous == WaitReasonProbe::Other) {
			WaitReasonProbe::s_current = reason;
		}
	}
	~WaitReasonScope() { WaitReasonProbe::s_current = m_previous; }
	WaitReasonScope(const WaitReasonScope&)            = delete;
	WaitReasonScope& operator=(const WaitReasonScope&) = delete;

private:
	uint32_t m_previous;
};

// Times one MasterSemaphore::Wait on the GPU thread and charges it to the current reason.
class WaitReasonTimer {
public:
	explicit WaitReasonTimer(bool active) noexcept
	    : m_enabled(active && ProbeEnabled::Get()), m_reason(WaitReasonProbe::s_current),
	      m_begin(m_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {}) {}
	~WaitReasonTimer() {
		if (!m_enabled) {
			return;
		}
		const auto elapsed = std::chrono::steady_clock::now() - m_begin;
		WaitReasonProbe::Add(m_reason, static_cast<uint64_t>(
		                                   std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
	}
	WaitReasonTimer(const WaitReasonTimer&)            = delete;
	WaitReasonTimer& operator=(const WaitReasonTimer&) = delete;

private:
	bool                                  m_enabled;
	uint32_t                              m_reason;
	std::chrono::steady_clock::time_point m_begin;
};

class HostReadScope {
public:
	explicit HostReadScope(HostReadProbe::Site site) noexcept
	    : m_reason(static_cast<WaitReasonProbe::Reason>(uint32_t {WaitReasonProbe::DrawArgs} + uint32_t {site})), m_site(site),
	      m_enabled(ProbeEnabled::Get()),
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
	WaitReasonScope                       m_reason; // waits inside a host read are charged to its site
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
	inline static thread_local uint32_t s_depth     = 0; // enabled scopes open on this thread
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
			BudgetProbe::s_depth++;
		}
	}
	~BudgetScope() {
		if (!m_enabled) {
			return;
		}
		BudgetProbe::s_depth--;
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

// KYTY_PROBE_OP_SPLIT=1 (with KYTY_PROBE_FAULT_HISTOGRAM=1): splits the draw and dispatch budget
// scopes' exclusive time into phases, printed as "| op-split ms (us/call)" on the FaultHistogram line.
//   state      register/state snapshot: render-state and stage input info, shader parameters
//   program    shader program lookup and SRT walk (ProgramCache::Get: permutation match, resource
//              materialization, specialization reads)
//   pipeline   Vulkan pipeline lookup / creation (GetGraphicsPipeline / GetComputePipeline)
//   resources  buffer/texture obtain and binding preparation (PrepareBindings, vertex/index/argument
//              buffers, render targets, LDS buffer)
//   barrier    hazard tracking and barriers (PrepareForDraw, indirect-argument and lazy barriers,
//              post-draw/dispatch tracking)
//   record     command recording (CommitBindings, push constants, dynamic state, BeginRendering,
//              bind, draw/dispatch)
//   walk       inside program: the SRT walk itself (MaterializeResources); "program" keeps the key
//              lookup, memo check and permutation match
//   compile    inside program: translating / compiling a new permutation (hitches)
//   textures   inside resources: texture/image resolution (ResolveTexture) and samplers
//   buffers    inside resources: FindBuffers / RebindBuffers, vertex, index, argument and LDS buffers
//   targets    inside resources: render-target and depth resolution and acquisition
//   other      everything else inside the op (special-case clears/copies, diagnostics, cleanup)
// One phase is current per GPU-thread op; OpSplitProbe::Enter switches it and OpPhaseScope switches for a
// region and back. Each switch charges the elapsed time minus the nested budget scopes' time (sync,
// wait, protect, submit: BudgetProbe::s_nested_ns) to the phase it leaves, so the phases partition the
// op's exclusive time. Switches inside a nested budget scope are ignored (their time is excluded).
// Off: one thread-local test per switch site.
struct OpSplitProbe {
	enum Op : uint32_t { Draw = 0, Dispatch, OpCount };
	enum Phase : uint32_t { Other = 0, State, Program, Pipeline, Resources, Barrier, Record, Walk, Compile,
		                    Textures, Buffers, Targets, PhaseCount };
	static constexpr auto kPhaseNames = std::to_array<const char*>(
	    {"other", "state", "program", "pipeline", "resources", "barrier", "record", "walk", "compile",
	     "textures", "buffers", "targets"});
	static_assert(kPhaseNames.size() == PhaseCount, "OpSplitProbe phase / name table mismatch");
	struct Snapshot {
		std::array<uint64_t, OpCount>                                calls {};
		std::array<std::array<uint64_t, PhaseCount>, OpCount>       ns {};
	};
	inline static std::array<std::atomic<uint64_t>, OpCount>                          s_calls {};
	inline static std::array<std::array<std::atomic<uint64_t>, PhaseCount>, OpCount> s_ns {};

	// The open op on this thread (one at a time; ops do not nest).
	inline static thread_local bool                                  t_active      = false;
	inline static thread_local uint32_t                              t_op          = 0;
	inline static thread_local uint32_t                              t_phase       = 0;
	inline static thread_local uint32_t                              t_depth       = 0;
	inline static thread_local uint64_t                              t_last_nested = 0;
	inline static thread_local std::chrono::steady_clock::time_point t_last {};

	static bool Enabled() noexcept {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_PROBE_OP_SPLIT");
			return value != nullptr && value[0] == '1' && ProbeEnabled::Get();
		}();
		return enabled;
	}
	// True when a switch here belongs to the open op (not inside a nested budget scope).
	static bool Here() noexcept { return t_active && BudgetProbe::s_depth == t_depth; }
	static void Charge() noexcept {
		const auto now     = std::chrono::steady_clock::now();
		const auto elapsed = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(now - t_last).count());
		const auto nested = BudgetProbe::s_nested_ns - t_last_nested;
		s_ns[t_op][t_phase].fetch_add(elapsed > nested ? elapsed - nested : 0, std::memory_order_relaxed);
		t_last        = now;
		t_last_nested = BudgetProbe::s_nested_ns;
	}
	static void Enter(Phase phase) noexcept {
		if (Here() && t_phase != phase) {
			Charge();
			t_phase = phase;
		}
	}
	static Snapshot Take() noexcept {
		Snapshot snapshot;
		for (uint32_t op = 0; op < OpCount; ++op) {
			snapshot.calls[op] = s_calls[op].exchange(0, std::memory_order_relaxed);
			for (uint32_t phase = 0; phase < PhaseCount; ++phase) {
				snapshot.ns[op][phase] = s_ns[op][phase].exchange(0, std::memory_order_relaxed);
			}
		}
		return snapshot;
	}
};

// Opens a draw or dispatch for OpSplitProbe; declare it right after the op's BudgetScope.
class OpSplitScope {
public:
	explicit OpSplitScope(OpSplitProbe::Op op) noexcept
	    : m_enabled(OpSplitProbe::Enabled() && !OpSplitProbe::t_active && BudgetProbe::s_depth != 0) {
		if (m_enabled) {
			OpSplitProbe::t_active      = true;
			OpSplitProbe::t_op          = op;
			OpSplitProbe::t_phase       = OpSplitProbe::Other;
			OpSplitProbe::t_depth       = BudgetProbe::s_depth;
			OpSplitProbe::t_last_nested = BudgetProbe::s_nested_ns;
			OpSplitProbe::t_last        = std::chrono::steady_clock::now();
		}
	}
	~OpSplitScope() {
		if (m_enabled) {
			OpSplitProbe::Charge();
			OpSplitProbe::s_calls[OpSplitProbe::t_op].fetch_add(1, std::memory_order_relaxed);
			OpSplitProbe::t_active = false;
		}
	}
	OpSplitScope(const OpSplitScope&)            = delete;
	OpSplitScope& operator=(const OpSplitScope&) = delete;

private:
	bool m_enabled;
};

// Charges a region of an open op to `phase`, then returns to the phase current before it.
class OpPhaseScope {
public:
	explicit OpPhaseScope(OpSplitProbe::Phase phase) noexcept: m_switched(OpSplitProbe::Here()) {
		if (m_switched) {
			m_previous = static_cast<OpSplitProbe::Phase>(OpSplitProbe::t_phase);
			OpSplitProbe::Enter(phase);
		}
	}
	~OpPhaseScope() {
		if (m_switched) {
			OpSplitProbe::Enter(m_previous);
		}
	}
	OpPhaseScope(const OpPhaseScope&)            = delete;
	OpPhaseScope& operator=(const OpPhaseScope&) = delete;

private:
	bool                m_switched;
	OpSplitProbe::Phase m_previous = OpSplitProbe::Other;
};

// CPU time (user, system) of the calling thread and of the whole process, in milliseconds, read
// from /proc; zero on platforms without it.
struct CpuTimes {
	uint64_t thread_user_ms = 0, thread_sys_ms = 0, process_user_ms = 0, process_sys_ms = 0;
	static CpuTimes Read() noexcept;
};

struct SyncProbe {
	// wait_pass_*: KYTY_SYNC_ON_WAIT_PASS accounting. passes = guest-memory observations that did
	// not suspend (WAIT_REG_MEM that passed with func != 0, conditional-IB compare, COND_EXEC,
	// SET_PREDICATION) while the stream's sync epoch was already consumed. Not counted: func 0
	// ("always") waits, and waits re-executed after a resume (fresh, unconsumed epoch);
	// consumed = a BDA/unbounded consumer sync was requested later in that same epoch (with the
	// switch off this is a sync the guest's acquire would have required, but that was skipped);
	// extra = such syncs that actually ran (switch on) and uploaded > 0 bytes; bytes = their bytes.
	struct Snapshot {
		uint64_t calls = 0, regions = 0;
		uint64_t wait_passes = 0, wait_consumed = 0, wait_extra = 0, wait_bytes = 0;
	};
	inline static std::atomic<uint64_t> s_calls {0};
	inline static std::atomic<uint64_t> s_regions {0};
	inline static std::atomic<uint64_t> s_wait_passes {0};
	inline static std::atomic<uint64_t> s_wait_consumed {0};
	inline static std::atomic<uint64_t> s_wait_extra {0};
	inline static std::atomic<uint64_t> s_wait_bytes {0};
	static void     NoteCall() noexcept { s_calls.fetch_add(1, std::memory_order_relaxed); }
	static void     NoteRegion() noexcept { s_regions.fetch_add(1, std::memory_order_relaxed); }
	static void     NoteWaitPass() noexcept { s_wait_passes.fetch_add(1, std::memory_order_relaxed); }
	static void     NoteWaitConsumed() noexcept { s_wait_consumed.fetch_add(1, std::memory_order_relaxed); }
	static void     NoteWaitExtra(uint64_t bytes) noexcept {
        s_wait_extra.fetch_add(1, std::memory_order_relaxed);
        s_wait_bytes.fetch_add(bytes, std::memory_order_relaxed);
	}
	static Snapshot Take() noexcept {
		return {s_calls.exchange(0, std::memory_order_relaxed),
		        s_regions.exchange(0, std::memory_order_relaxed),
		        s_wait_passes.exchange(0, std::memory_order_relaxed),
		        s_wait_consumed.exchange(0, std::memory_order_relaxed),
		        s_wait_extra.exchange(0, std::memory_order_relaxed),
		        s_wait_bytes.exchange(0, std::memory_order_relaxed)};
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
