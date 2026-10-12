#include "graphics/host_gpu/renderer/renderContext.h"

#include <cstdlib>
#include <fmt/format.h>
#include <vector>
#include <string>
#include <atomic>
#include <unordered_map>

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
	void Dump(Describe&& describe, const BufferCache::SyncStats& sync) {
		if (!enabled) return;
		static uint64_t last_total = 0;
		static int      calls      = 0;
		if (++calls % 64 != 0) return;
		const auto now = total.load(std::memory_order_relaxed);
		std::unordered_map<uint64_t, uint32_t> buckets;
		uint32_t sampled = 0;
		for (auto& entry: entries) {
			const auto value = entry.exchange(0, std::memory_order_relaxed);
			if (value == 0) continue;
			++sampled;
			++buckets[(value >> 20u) << 20u];
		}
		std::vector<std::pair<uint64_t, uint32_t>> top(buckets.begin(), buckets.end());
		std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.second > b.second; });
		std::string line = fmt::format("FaultHistogram: total={} (+{}) reads={} sampled={} top MiB:",
		                               now, now - last_total, reads.load(), sampled);
		for (size_t i = 0; i < top.size() && i < 8; ++i) {
			line += fmt::format(" 0x{:x}:{} {}", top[i].first, top[i].second, describe(top[i].first));
		}
		line += fmt::format(" | syncs: obtain {} calls {} pages {} KiB, range {} calls {} pages {} KiB"
		                    " | downloads {} ({} ms waited), ReadMemory {}",
		                    sync.obtain_calls, sync.obtain_pages, sync.obtain_bytes >> 10,
		                    sync.range_calls, sync.range_pages, sync.range_bytes >> 10,
		                    sync.download_calls, sync.download_wait_ns / 1000000, sync.read_memory_calls);
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
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
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

void RenderContext::SyncCpuWrites() {
	// Once per command stream (KYTY_BDA_SYNC_PER_DRAW=1 restores the per-draw sync): every sync
	// re-protects the pages it uploads, so the game's next CPU write to them faults again.
	static const bool per_draw = [] {
		const char* value = std::getenv("KYTY_BDA_SYNC_PER_DRAW");
		return value != nullptr && value[0] == '1';
	}();
	if (!per_draw) {
		if (m_bda_synced_epoch == m_bda_sync_epoch) {
			return;
		}
		m_bda_synced_epoch = m_bda_sync_epoch;
	}
	// KYTY_BDA_SYNC_ALL=1 restores the walk over every cached buffer.
	static const bool sync_all = [] {
		const char* value = std::getenv("KYTY_BDA_SYNC_ALL");
		return value != nullptr && value[0] == '1';
	}();
	std::shared_lock lock(m_mapped_ranges_mutex);
	if (sync_all) {
		m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
			m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
		});
	} else {
		// Only regions with new CPU-dirty pages (and regions holding hot pages), clipped to mapped
		// memory: cached buffers can extend past the mapping and unmapped pages must not be read.
		m_buffer_cache.ForEachBdaSyncPendingRegion([this](uint64_t vaddr, uint64_t size) {
			m_mapped_ranges.ForEachInRange(vaddr, size, [this](uint64_t start, uint64_t end) {
				m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
			});
		});
	}
}

void RenderContext::RunGarbageCollector() {
	g_fault_sampler.Dump([this](uint64_t vaddr) { return m_buffer_cache.DescribeBufferAt(vaddr); },
	                     m_buffer_cache.TakeSyncStats());
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
