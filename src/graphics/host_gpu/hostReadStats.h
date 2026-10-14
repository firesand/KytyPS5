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
	struct Snapshot {
		std::array<std::pair<uint64_t, uint64_t>, SiteCount> sites {}; // calls, nanoseconds
	};
	inline static std::array<HostReadCounters, SiteCount> s_sites {};

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
struct BudgetProbe {
	enum Category : uint32_t { Wait = 0, Sync, Draw, Dispatch, Protect, Submit, CategoryCount };
	struct Snapshot {
		std::array<std::pair<uint64_t, uint64_t>, CategoryCount> categories {}; // calls, nanoseconds
	};
	inline static std::array<HostReadCounters, CategoryCount> s_categories {};
	static void Add(Category category, uint64_t ns) noexcept {
		s_categories[category].calls.fetch_add(1, std::memory_order_relaxed);
		s_categories[category].ns.fetch_add(ns, std::memory_order_relaxed);
	}
	static Snapshot Take() noexcept {
		Snapshot snapshot;
		for (uint32_t i = 0; i < CategoryCount; ++i) {
			snapshot.categories[i] = {s_categories[i].calls.exchange(0, std::memory_order_relaxed),
			                          s_categories[i].ns.exchange(0, std::memory_order_relaxed)};
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
	                        : std::chrono::steady_clock::time_point {}) {}
	~BudgetScope() {
		if (!m_enabled) {
			return;
		}
		const auto elapsed = std::chrono::steady_clock::now() - m_begin;
		BudgetProbe::Add(m_category, static_cast<uint64_t>(
		                                 std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed)
		                                     .count()));
	}
	BudgetScope(const BudgetScope&)            = delete;
	BudgetScope& operator=(const BudgetScope&) = delete;

private:
	BudgetProbe::Category                 m_category;
	bool                                  m_enabled;
	std::chrono::steady_clock::time_point m_begin;
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

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_HOSTREADSTATS_H_
