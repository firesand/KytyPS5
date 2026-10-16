#include "graphics/host_gpu/renderer/renderContext.h"

#include "graphics/host_gpu/hostReadStats.h"
#include "graphics/host_gpu/renderer/hazardState.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fmt/format.h>
#include <vector>
#include <string>
#include <atomic>
#include <unordered_map>
#include <utility>

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

// KYTY_PROBE_FAULT_HISTOGRAM=1: lock-free sample of CPU page faults handled by the caches, dumped
// as a per-MiB histogram from RunGarbageCollector (signal-handler safe: no allocation, no locks).
CpuTimes CpuTimes::Read() noexcept {
	CpuTimes   times;
	const auto read = [](const char* path, uint64_t& user_ms, uint64_t& sys_ms) {
		FILE* file = std::fopen(path, "r");
		if (file == nullptr) return;
		char   buffer[1024];
		size_t length = std::fread(buffer, 1, sizeof(buffer) - 1, file);
		std::fclose(file);
		buffer[length] = 0;
		// Fields after the parenthesised comm: state(3) ... utime(14) stime(15).
		const char* close = std::strrchr(buffer, ')');
		if (close == nullptr) return;
		unsigned long utime = 0, stime = 0;
		if (std::sscanf(close + 1, " %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu", &utime,
		                &stime) != 2) {
			return;
		}
		static const long ticks = sysconf(_SC_CLK_TCK);
		user_ms = ticks > 0 ? utime * 1000ull / static_cast<unsigned long>(ticks) : 0;
		sys_ms  = ticks > 0 ? stime * 1000ull / static_cast<unsigned long>(ticks) : 0;
	};
	read("/proc/thread-self/stat", times.thread_user_ms, times.thread_sys_ms);
	read("/proc/self/stat", times.process_user_ms, times.process_sys_ms);
	return times;
}

namespace {
struct FaultSampler {
	static constexpr size_t   Slots = 1u << 16;
	std::atomic<uint64_t>     entries[Slots] {};
	std::atomic<uint32_t>     cursor {0};
	std::atomic<uint64_t>     total {0};
	std::atomic<uint64_t>     reads {0};
	bool                      enabled = [] {
        const char* value = std::getenv("KYTY_PROBE_FAULT_HISTOGRAM");
        return value != nullptr && value[0] == '1';
	}();
	void Record(uint64_t vaddr, bool write) noexcept {
		if (!enabled) return;
		total.fetch_add(1, std::memory_order_relaxed);
		if (!write) reads.fetch_add(1, std::memory_order_relaxed);
		const auto slot = cursor.fetch_add(1, std::memory_order_relaxed) & (Slots - 1);
		entries[slot].store((vaddr & ~uint64_t {0xfff}) | (write ? 1u : 0u), std::memory_order_relaxed);
	}
	template <typename Describe>
	void Dump(Describe&& describe, const BufferCache::SyncStats& sync,
	          std::pair<uint64_t, uint64_t> protects, HotPageStats hot,
	          const HostReadProbe::Snapshot& host_reads, SyncProbe::Snapshot syncs,
	          const BudgetProbe::Snapshot& budget, const HostResidentProbe::Snapshot& resident) {
		if (!enabled) return;
		static uint64_t last_total = 0;
		static int      calls      = 0;
		// The callers hand over counters taken since the previous call; add them up so the line
		// printed every 64th call covers the whole interval (Codex review: 63/64 were dropped).
		static BufferCache::SyncStats         sync_sum {};
		static std::pair<uint64_t, uint64_t>  protect_sum {};
		static HotPageStats                   hot_sum {};
		static HostReadProbe::Snapshot        reads_sum {};
		static SyncProbe::Snapshot            syncs_sum {};
		static BudgetProbe::Snapshot          budget_sum {};
		static HostResidentProbe::Snapshot    resident_sum {};
		static CpuTimes                       last_cpu = CpuTimes::Read();
		static auto                           last_wall = std::chrono::steady_clock::now();
		sync_sum.obtain_calls += sync.obtain_calls;
		sync_sum.obtain_bytes += sync.obtain_bytes;
		sync_sum.obtain_pages += sync.obtain_pages;
		sync_sum.range_calls += sync.range_calls;
		sync_sum.range_bytes += sync.range_bytes;
		sync_sum.range_pages += sync.range_pages;
		sync_sum.range_repasses += sync.range_repasses;
		sync_sum.range_repass_pages += sync.range_repass_pages;
		sync_sum.download_calls += sync.download_calls;
		sync_sum.download_wait_ns += sync.download_wait_ns;
		sync_sum.read_memory_calls += sync.read_memory_calls;
		sync_sum.copy_batches += sync.copy_batches;
		sync_sum.copy_buffers += sync.copy_buffers;
		protect_sum.first += protects.first;
		protect_sum.second += protects.second;
		hot_sum.hashed += hot.hashed;
		hot_sum.skipped += hot.skipped;
		hot_sum.demoted += hot.demoted;
		hot_sum.invalidated += hot.invalidated;
		for (size_t i = 0; i < host_reads.sites.size(); ++i) {
			reads_sum.sites[i].first += host_reads.sites[i].first;
			reads_sum.sites[i].second += host_reads.sites[i].second;
		}
		resident_sum.binds += resident.binds;
		resident_sum.bytes += resident.bytes;
		resident_sum.fallbacks += resident.fallbacks;
		for (size_t i = 0; i < host_reads.gpu_indirect.size(); ++i) {
			reads_sum.gpu_indirect[i] += host_reads.gpu_indirect[i];
		}
		reads_sum.gpu_indirect_wait_ns += host_reads.gpu_indirect_wait_ns;
		syncs_sum.calls += syncs.calls;
		syncs_sum.regions += syncs.regions;
		syncs_sum.wait_passes += syncs.wait_passes;
		syncs_sum.wait_consumed += syncs.wait_consumed;
		syncs_sum.wait_extra += syncs.wait_extra;
		syncs_sum.wait_bytes += syncs.wait_bytes;
		for (size_t i = 0; i < budget.categories.size(); ++i) {
			budget_sum.categories[i].calls += budget.categories[i].calls;
			budget_sum.categories[i].inclusive_ns += budget.categories[i].inclusive_ns;
			budget_sum.categories[i].exclusive_ns += budget.categories[i].exclusive_ns;
		}
		if (++calls % 64 != 0) return;
		const auto sync_all = std::exchange(sync_sum, {});
		const auto protects_all = std::exchange(protect_sum, {});
		const auto hot_all = std::exchange(hot_sum, {});
		const auto reads_all = std::exchange(reads_sum, {});
		const auto syncs_all = std::exchange(syncs_sum, {});
		const auto budget_all = std::exchange(budget_sum, {});
		const auto resident_all = std::exchange(resident_sum, {});
		const auto cpu_now    = CpuTimes::Read();
		const auto wall_now   = std::chrono::steady_clock::now();
		const auto wall_ms    = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::milliseconds>(wall_now - last_wall).count());
		const CpuTimes cpu_delta {cpu_now.thread_user_ms - last_cpu.thread_user_ms,
		                          cpu_now.thread_sys_ms - last_cpu.thread_sys_ms,
		                          cpu_now.process_user_ms - last_cpu.process_user_ms,
		                          cpu_now.process_sys_ms - last_cpu.process_sys_ms};
		last_cpu  = cpu_now;
		last_wall = wall_now;
		const auto now = total.load(std::memory_order_relaxed);
		std::unordered_map<uint64_t, uint32_t> pages;
		uint32_t sampled = 0;
		for (auto& entry: entries) {
			const auto value = entry.exchange(0, std::memory_order_relaxed);
			if (value == 0) continue;
			++sampled;
			++pages[value & ~uint64_t {0xfff}];
		}
		// Aggregate per MiB but describe each MiB's hottest page (the describer is page-based).
		std::unordered_map<uint64_t, std::pair<uint32_t, std::pair<uint64_t, uint32_t>>> buckets;
		for (const auto& [page, count]: pages) {
			auto& bucket = buckets[(page >> 20u) << 20u];
			bucket.first += count;
			if (count > bucket.second.second) bucket.second = {page, count};
		}
		std::vector<std::pair<uint64_t, std::pair<uint32_t, std::pair<uint64_t, uint32_t>>>> top(
		    buckets.begin(), buckets.end());
		std::sort(top.begin(), top.end(),
		          [](auto& a, auto& b) { return a.second.first > b.second.first; });
		std::string line = fmt::format("FaultHistogram: total={} (+{}) reads={} sampled={} pages={} top MiB:",
		                               now, now - last_total, reads.load(), sampled, pages.size());
		for (size_t i = 0; i < top.size() && i < 8; ++i) {
			const auto [page, page_count] = top[i].second.second;
			line += fmt::format(" 0x{:x}:{} p0x{:x}:{} {}", top[i].first, top[i].second.first, page,
			                    page_count, describe(page));
		}
		line += fmt::format(" | syncs: obtain {} calls {} ranges {} KiB, range {} calls {} ranges {} KiB"
		                    " (repass {} passes {} pages) | downloads {} ({} ms waited), ReadMemory {}",
		                    sync_all.obtain_calls, sync_all.obtain_pages, sync_all.obtain_bytes >> 10,
		                    sync_all.range_calls, sync_all.range_pages, sync_all.range_bytes >> 10,
		                    sync_all.range_repasses, sync_all.range_repass_pages,
		                    sync_all.download_calls, sync_all.download_wait_ns / 1000000,
		                    sync_all.read_memory_calls);
		line += fmt::format(" | protect {} calls {} KiB | hot hashed {} skipped {} demoted {} rehash-miss {}",
		                    protects_all.first, protects_all.second >> 10, hot_all.hashed,
		                    hot_all.skipped, hot_all.demoted, hot_all.invalidated);
		const auto& rs = reads_all.sites;
		line += fmt::format(" | cpu-sync {} calls {} regions {} copy-batches {} copy-buffers"
		                    " | cp reads: args {} ({} ms) count {} ({} ms) dispatch {} ({} ms) pred {} ({} ms)"
		                    " | gpu-indirect {} draws {} fallback(index_offset) {} fallback(other) ({} of which mesh) {} fallback(srt) {} ms waited",
		                    syncs_all.calls, syncs_all.regions, sync_all.copy_batches, sync_all.copy_buffers,
		                    rs[HostReadProbe::DrawArgs].first,
		                    rs[HostReadProbe::DrawArgs].second / 1000000,
		                    rs[HostReadProbe::DrawCount].first, rs[HostReadProbe::DrawCount].second / 1000000,
		                    rs[HostReadProbe::DispatchArgs].first,
		                    rs[HostReadProbe::DispatchArgs].second / 1000000,
		                    rs[HostReadProbe::Predication].first,
		                    rs[HostReadProbe::Predication].second / 1000000,
		                    reads_all.gpu_indirect[HostReadProbe::GpuDraws],
		                    reads_all.gpu_indirect[HostReadProbe::FallbackIndexOffset],
		                    reads_all.gpu_indirect[HostReadProbe::FallbackOther],
		                    reads_all.gpu_indirect[HostReadProbe::FallbackMesh],
		                    reads_all.gpu_indirect[HostReadProbe::FallbackSrt],
		                    reads_all.gpu_indirect_wait_ns / 1000000);
		line += fmt::format(" | wait-pass {} (consumed {}, extra-syncs {}, bytes {})", syncs_all.wait_passes,
		                    syncs_all.wait_consumed, syncs_all.wait_extra, syncs_all.wait_bytes);
		line += fmt::format(" | host-resident binds {} bytes {} KiB fallbacks {}", resident_all.binds,
		                    resident_all.bytes >> 10, resident_all.fallbacks);
		// Lazy hazard-tracked barriers (KYTY_LAZY_BARRIERS). Mode 1: per site `emitted/elided`
		// (elided = barriers the pre-tracker code recorded there minus those the tracker recorded;
		// negative means the tracker added one). The tracker does not record draw-post barriers
		// itself: they are replaced by flush-draw / flush-end-rendering, so draw-post is credited
		// with those two counts (an upper bound, they also serve dispatch hazards). Mode 2 keeps
		// the old barriers, nothing is elided: it prints `decisions (old N)` only. Then the places
		// where deferred hazards were emitted.
		{
			const auto bar  = BarrierProbe::Take();
			const auto mode = LazyBarrierMode();
			line += fmt::format(" | barriers mode {}:", mode);
			const auto replaced = static_cast<int64_t>(bar.emitted[BarrierProbe::FlushDraw]) +
			                      static_cast<int64_t>(bar.emitted[BarrierProbe::FlushEndRendering]);
			for (uint32_t k = 0; k < BarrierProbe::KindCount; ++k) {
				if (bar.old[k] != 0 || k < BarrierProbe::FlushEndRendering) {
					if (mode == 2) {
						line += fmt::format(" {} {}(old {})", BarrierProbe::Name(k), bar.emitted[k],
						                    bar.old[k]);
					} else {
						auto elided = static_cast<int64_t>(bar.old[k]) -
						              static_cast<int64_t>(bar.emitted[k]);
						if (k == BarrierProbe::DrawPost) {
							elided -= replaced;
						}
						line += fmt::format(" {} {}/{}", BarrierProbe::Name(k), bar.emitted[k], elided);
					}
				} else {
					line += fmt::format(" {} {}", BarrierProbe::Name(k), bar.emitted[k]);
				}
			}
		}
		const auto& bc = budget_all.categories;
		const auto  ms = [](uint64_t ns) { return ns / 1000000; };
		const auto cat = [&](BudgetProbe::Category c) {
			return fmt::format("{}/{} ({})", ms(bc[c].inclusive_ns), ms(bc[c].exclusive_ns), bc[c].calls);
		};
		line += fmt::format(" | wall {} ms | gpu-thread cpu user {} sys {} ms | process cpu user {} sys {} ms"
		                    " | gpu-thread budget incl/excl ms (calls): wait {} sync {} draw {} dispatch {}"
		                    " protect {} submit {}",
		                    wall_ms, cpu_delta.thread_user_ms, cpu_delta.thread_sys_ms,
		                    cpu_delta.process_user_ms, cpu_delta.process_sys_ms, cat(BudgetProbe::Wait),
		                    cat(BudgetProbe::Sync), cat(BudgetProbe::Draw), cat(BudgetProbe::Dispatch),
		                    cat(BudgetProbe::Protect), cat(BudgetProbe::Submit));
		LOGF("%s\n", line.c_str());
		last_total = now;
	}
};
FaultSampler g_fault_sampler;
} // namespace

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	g_fault_sampler.Record(fault_vaddr, access == PageFaultAccess::Write);
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateWriteFault(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations())) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	SyncCpuWrites();
	m_fault_process_pending = true;
}

static std::atomic<int>& SyncOnWaitPassMode() {
	static std::atomic<int> mode = [] {
		const char* value = std::getenv("KYTY_SYNC_ON_WAIT_PASS");
		return value == nullptr ? 0 : std::atoi(value);
	}();
	return mode;
}

void RenderContext::SetSyncOnWaitPassMode(int mode) {
	SyncOnWaitPassMode().store(mode);
}

void RenderContext::NoteGuestAcquire(uint64_t label_addr) {
	SyncProbe::NoteWaitPass();
	const int mode = SyncOnWaitPassMode().load();
	// Nothing to do when this epoch has not synced yet: the first consumer syncs anyway.
	if (m_bda_synced_epoch != m_bda_sync_epoch) {
		return;
	}
	// Mode 0 only measures: the consumer is counted, the sync is not made.
	if (mode == 2 && label_addr != 0 && !m_buffer_cache.IsRegionCpuModified(label_addr, 4)) {
		return;
	}
	m_wait_pass_pending = true;
	if (mode != 0) {
		++m_bda_sync_epoch;
	}
}

void RenderContext::SyncCpuWrites() {
	// Once per command stream (KYTY_BDA_SYNC_PER_DRAW=1 restores the per-draw sync): every sync
	// re-protects the pages it uploads, so the game's next CPU write to them faults again.
	static const bool per_draw = [] {
		const char* value = std::getenv("KYTY_BDA_SYNC_PER_DRAW");
		return value != nullptr && value[0] == '1';
	}();
	if (!per_draw) {
		if (m_bda_synced_epoch == m_bda_sync_epoch) {
			if (m_wait_pass_pending) { // switch off: the sync the passing wait would have caused
				m_wait_pass_pending = false;
				SyncProbe::NoteWaitConsumed();
			}
			return;
		}
		m_bda_synced_epoch = m_bda_sync_epoch;
	}
	const bool     wait_pass   = m_wait_pass_pending;
	const uint64_t bytes_begin = m_buffer_cache.RangeBytesTotal();
	if (wait_pass) {
		m_wait_pass_pending = false;
		SyncProbe::NoteWaitConsumed();
	}
	SyncProbe::NoteCall();
	BudgetScope budget(BudgetProbe::Sync, true);
	// KYTY_BDA_SYNC_ALL=1 restores the walk over every cached buffer.
	static const bool sync_all = [] {
		const char* value = std::getenv("KYTY_BDA_SYNC_ALL");
		return value != nullptr && value[0] == '1';
	}();
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_buffer_cache.BeginUploadBatch();
	if (sync_all) {
		m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
			SyncProbe::NoteRegion();
			m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
		});
	} else {
		// Only regions with new CPU-dirty pages (and regions holding hot pages), clipped to mapped
		// memory: cached buffers can extend past the mapping and unmapped pages must not be read.
		m_buffer_cache.ForEachBdaSyncPendingRegion([this](uint64_t vaddr, uint64_t size) {
			SyncProbe::NoteRegion();
			m_mapped_ranges.ForEachInRange(vaddr, size, [this](uint64_t start, uint64_t end) {
				m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
			});
		});
	}
	m_buffer_cache.EndUploadBatch();
	if (wait_pass) {
		const uint64_t bytes = m_buffer_cache.RangeBytesTotal() - bytes_begin;
		if (bytes != 0) {
			SyncProbe::NoteWaitExtra(bytes);
		}
	}
}

void RenderContext::RunGarbageCollector() {
	GetCommandScheduler().DumpGpuTimestamps();
	g_fault_sampler.Dump(
	    [this](uint64_t vaddr) {
		    auto text = m_buffer_cache.DescribeBufferAt(vaddr);
		    if (text == "none") text = m_texture_cache.DescribeImagesIn(vaddr, uint64_t {1} << 20);
		    return text;
	    },
	    m_buffer_cache.TakeSyncStats(), PageManager::TakeProtectStats(),
	    MemoryTracker::TakeHotStats(), HostReadProbe::Take(), SyncProbe::Take(),
	    BudgetProbe::Take(), HostResidentProbe::Take());
	m_buffer_cache.FlushWriteFaults(false);
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
	m_pipeline_cache.MaybeSave();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
