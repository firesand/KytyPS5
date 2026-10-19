#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/writeCombinedCopy.h"
#include "graphics/host_gpu/hostReadStats.h"
#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include "graphics/host_gpu/renderer/gpuDiagnostics.h"

#include "common/alignment.h"
#include <fmt/format.h>
#include <chrono>
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/writeWatch.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {
// Kernel write-watch diagnostics (BufferCache::TakeWriteWatchStats).
std::atomic<uint64_t> g_ww_ownership_unscanned {0};
std::atomic<uint64_t> g_ww_fault_pass_collects {0};
} // namespace

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, bool allow_wait,
                                        const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment, allow_wait);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		m_scheduler.Current().Hazards().transfer_wrote = true; // hazard tracker: staging copy
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		{ GpuTimestampScope ts_scope(command, GpuTimestamps::Kind::BarrierBuffer); command.pipelineBarrier2(dependency); }
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		{ GpuTimestampScope ts_scope(command, GpuTimestamps::Kind::CopyUploadStaging); ts_scope.SetBytes(chunk); command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy); }
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		{ GpuTimestampScope ts_scope(command, GpuTimestamps::Kind::BarrierBuffer); command.pipelineBarrier2(dependency); }
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	DrainPendingGpuModified();
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	m_last_download_bar = false;
	if constexpr (!async) {
		const bool bar = TryBarReadback(buffer, copies);
		if (ProbeEnabled::Get()) {
			ReadbackProbe::Count(WaitReasonProbe::s_current,
			                     bar                       ? ReadbackProbe::Bar
			                     : m_last_bar_reason == 4 ? ReadbackProbe::CopyOpen
			                     : m_last_bar_reason == 3 ? ReadbackProbe::CopyInFlight
			                                              : ReadbackProbe::CopyOther);
		}
		if (bar) {
			return true;
		}
	}

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	{ GpuTimestampScope ts_scope(native, GpuTimestamps::Kind::BarrierBuffer); native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr); }
	{ GpuTimestampScope ts_scope(native, GpuTimestamps::Kind::CopyUploadDownload); if (ts_scope.Active()) ts_scope.SetBytes(SumCopySizes(copies)); native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data()); }
	if (GpuDiagnostics::Enabled()) {
		for (const auto& copy : copies) {
			GpuDiagnosticEvent event;
			event.kind = "readback-copy";
			event.args = {GpuDiagnostics::HandleValue(buffer.Handle()),
			              GpuDiagnostics::HandleValue(download.Handle()), copy.srcOffset,
			              copy.dstOffset, copy.size, buffer_address + copy.srcOffset};
			m_scheduler.RecordGpuOperation(event);
		}
	}

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	{ GpuTimestampScope ts_scope(native, GpuTimestamps::Kind::BarrierBuffer); native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr); }
	auto publish = [this, mapped, offset, total_size, buffer_address,
	                copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto      tick  = m_scheduler.CurrentTick();
		const auto      start = std::chrono::steady_clock::now();
		WaitReasonScope reason(WaitReasonProbe::Download);
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		m_sync_stats.download_calls++;
		m_sync_stats.download_wait_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
		        .count());
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	if (HostResidentRequested()) {
		m_host_resident = std::make_unique<HostResidentMemory>(graphics);
	}
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	const auto reason = GuestGpu::IsGpuThread() ? WaitReasonProbe::InvalidateGpu : WaitReasonProbe::InvalidateGuest;
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size, reason] { ReadMemory(vaddr, size, true, reason); });
}

void BufferCache::InvalidateWriteFault(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid write-fault range\n");
	}
	const auto reason = GuestGpu::IsGpuThread() ? WaitReasonProbe::WriteFaultGpu : WaitReasonProbe::WriteFaultGuest;
	m_memory_tracker.InvalidateWriteFault(
	    vaddr, size, [this, vaddr, size, reason] { ReadMemory(vaddr, size, true, reason); });
}

void BufferCache::NoteGpuIndirectArgs(uint64_t vaddr, uint64_t size) {
	m_gpu_indirect_args[m_gpu_indirect_args_next] = {vaddr >> 12u, ((vaddr + size - 1u) >> 12u) + 1u};
	m_gpu_indirect_args_next = (m_gpu_indirect_args_next + 1u) % kGpuIndirectArgsRing;
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write,
                             WaitReasonProbe::Reason reason) {
	m_sync_stats.read_memory_calls++; // counted on the calling thread; approximate
	// The calling thread's declared reason wins (a host read inside HostReadScope stays draw-args);
	// the GPU thread runs the readback below with it.
	const auto wait_reason = WaitReasonProbe::CurrentOr(reason);
	// KYTY_PROBE_READBACK_LOG=1: one line per synchronous readback request (first 4000, then 1/100).
	static const bool readback_log = [] {
		const char* value = std::getenv("KYTY_PROBE_READBACK_LOG");
		return value != nullptr && value[0] == '1';
	}();
	static std::atomic<uint64_t> readback_count {0};
	const auto readback_index = readback_log ? readback_count.fetch_add(1) : 0;
	const auto readback_start = std::chrono::steady_clock::now();
	struct ReadbackReport {
		bool enabled; uint64_t index; uint64_t vaddr, size; bool is_write; bool gpu_thread;
		std::chrono::steady_clock::time_point start; const bool* bar; const int* reason;
		uint32_t wait_reason;
		~ReadbackReport() {
			if (!enabled || (index >= 4000 && index % 100 != 0)) return;
			const auto ms = std::chrono::duration_cast<std::chrono::microseconds>(
			                    std::chrono::steady_clock::now() - start).count() / 1000.0;
			LOGF("Readback: #%llu %s 0x%016" PRIx64 "+0x%" PRIx64 " from %s %.2f ms %s reason=%s\n",
			     static_cast<unsigned long long>(index), is_write ? "write-fault" : "read",
			     vaddr, size, gpu_thread ? "gpu-thread" : "game-thread", ms,
			     *bar ? "bar" : (*reason == 1 ? "unmapped" : *reason == 2 ? "notick" :
			                     *reason == 3 ? "inflight" : *reason == 4 ? "open" : "copy"),
			     WaitReasonProbe::Name(wait_reason));
		}
	} readback_report {readback_log, readback_index, vaddr, size, is_write,
	                   GuestGpu::IsGpuThread(), readback_start, &m_last_download_bar,
	                   &m_last_bar_reason, wait_reason};
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write, wait_reason] {
		WaitReasonScope reason_scope(wait_reason);
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		// Diagnostics: a readback of a page that recent GPU-indirect draws took arguments from.
		if (const auto first = vaddr >> 12u, last = ((vaddr + size - 1u) >> 12u) + 1u;
		    std::ranges::any_of(m_gpu_indirect_args, [&](const auto& range) {
			    return range.second != 0 && first < range.second && range.first < last;
		    })) {
			HostReadProbe::CountGpuIndirect(HostReadProbe::ArgsPageReadback);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				LOGF("GpuIndirect(args-page-readback): %s 0x%016" PRIx64 "+0x%" PRIx64 " reason=%s\n",
				     is_write ? "write" : "read", vaddr, size, WaitReasonProbe::Name(wait_reason));
			}
		}
		uint64_t publish_vaddr = vaddr;
		uint64_t publish_size  = size;
		const auto current_buffer = [&]() -> Buffer& {
			return m_slot_buffers[FindBuffer(publish_vaddr, publish_size)];
		};
		bool sync_images = false;
		for (;;) {
			// Publishing an unbounded write can preserve an overlapping image, extend its
			// buffer, and retire the previous owner. Resolve only after those mutations.
			PrepareHostRead(publish_vaddr, publish_size);
			PublishPendingTextureInvalidations();
			uint64_t expanded_vaddr = publish_vaddr;
			uint64_t expanded_size  = publish_size;
			sync_images = m_texture_cache.SyncGpuImagesForHostRead(
			    current_buffer(), expanded_vaddr, expanded_size, /*record=*/false);
			if (!sync_images ||
			    (expanded_vaddr == publish_vaddr && expanded_size == publish_size)) {
				break;
			}
			// A page-sharing image widened the publication hull. Drain writes over the full
			// new range before selecting its authoritative images, then reacquire its owner.
			publish_vaddr = expanded_vaddr;
			publish_size  = expanded_size;
		}
		if (sync_images && m_texture_cache.SyncGpuImagesForHostRead(
			                   current_buffer(), publish_vaddr, publish_size, /*record=*/true)) {
			// The copy is in flight in this tick: keep TryBarReadback off the host-visible path.
			NoteGpuWrite(publish_vaddr, publish_size, m_scheduler.CurrentTick());
		}
		// Image-copy preparation may wait for stream space. Do not carry a cached Buffer
		// reference across it; download from the current owner of the complete hull.
		if (DownloadBufferMemory<false>(current_buffer(), publish_vaddr, publish_size)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(publish_vaddr, publish_size);
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
	});
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

// A replaced buffer whose device bytes are all complete can be copied on the host through the
// BAR mappings right now; only an overlap with a GPU write still in flight needs a recorded GPU
// copy (which then gates BAR readbacks of the new buffer through ready_tick). Without this every
// new buffer would wait for the current tick before any BAR readback (ng-39: menu 7 vs 12 fps).
bool BufferCache::HostJoinPossible(const Buffer& new_buffer, const Buffer& overlap) {
	// Experimental (KYTY_HOST_JOIN=1): off by default until delayed-upload->join and A->B->C join
	// regressions exist (Codex P1 2026-10-13). The default records GPU joins as before; a new
	// buffer only gets a ready_tick when such a copy was recorded.
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HOST_JOIN");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	if (!enabled) {
		return false;
	}
	if (new_buffer.Mapped().empty() || overlap.Mapped().empty() || !overlap.IsCoherent() ||
	    !new_buffer.IsCoherent()) {
		return false;
	}
	if (m_bar_block_tick != 0 && !m_scheduler.IsFree(m_bar_block_tick)) {
		return false;
	}
	// Every producer of the overlap's device bytes must have executed: its own join copies
	// (ready_tick) and any recorded upload / written binding / fill / copy into it
	// (last_gpu_write_tick), not only the per-page write stamps below.
	if ((overlap.ready_tick != 0 && !m_scheduler.IsFree(overlap.ready_tick)) ||
	    (overlap.last_gpu_write_tick != 0 && !m_scheduler.IsFree(overlap.last_gpu_write_tick))) {
		return false;
	}
	const auto address = overlap.CpuAddress();
	const auto first   = address >> CACHING_PAGEBITS;
	const auto last    = (address + overlap.Size() - 1) >> CACHING_PAGEBITS;
	for (auto page = first; page <= last; ++page) {
		if (const auto* stamp = m_gpu_write_ticks.Find(page); stamp != nullptr) {
			if (*stamp != 0 && !m_scheduler.IsFree(*stamp)) {
				return false; // a GPU write to the old buffer is still in flight
			}
			continue;
		}
		// No stamp: a GPU-dirty page marked by a path without a tick has unknown completion.
		if (m_gpu_modified_ranges.Intersects(page << CACHING_PAGEBITS, CACHING_PAGESIZE)) {
			return false;
		}
	}
	return true;
}

bool BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	const auto offset   = overlap.CpuAddress() - new_buffer.CpuAddress();
	bool       gpu_copy = true;
	if (HostJoinPossible(new_buffer, overlap)) {
		std::memcpy(new_buffer.Mapped().data() + offset, overlap.Mapped().data(), overlap.Size());
		m_sync_stats.host_joins++;
		gpu_copy = false;
	} else {
		new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0, offset, overlap.Size());
		new_buffer.last_gpu_write_tick = m_scheduler.CurrentTick();
		m_sync_stats.gpu_joins++;
	}
	DeleteBuffer(overlap_id);
	return gpu_copy;
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	bool gpu_copy = false;
	const bool has_overlap = overlap.first != overlap.last;
	uint64_t old_begin = overlap.end;
	uint64_t old_end   = overlap.begin;
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		const auto& old_buffer = m_slot_buffers[old_id];
		old_begin = std::min(old_begin, old_buffer.CpuAddress());
		old_end   = std::max(old_end, old_buffer.CpuAddress() + old_buffer.Size());
		gpu_copy |= JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	if (gpu_copy) {
		m_slot_buffers[id].ready_tick = m_scheduler.CurrentTick(); // the recorded join copies
	}
	Register(id);
	if (!has_overlap) {
		m_memory_tracker.MarkBdaSyncPending(overlap.begin, overlap.end - overlap.begin);
	} else {
		if (overlap.begin < old_begin) {
			m_memory_tracker.MarkBdaSyncPending(overlap.begin, old_begin - overlap.begin);
		}
		if (old_end < overlap.end) {
			m_memory_tracker.MarkBdaSyncPending(old_end, overlap.end - old_end);
		}
	}
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	if (is_written && WriteWatch::Get() != nullptr) {
		// Kernel write-watch: no scan per written binding (ng-154: tens of thousands per second).
		// CPU writes up to this stream's (or acquire's) collection are applied; a later one races
		// the GPU's ownership and is merged at the next collection, where the download copies only
		// GPU-written bytes, so CPU bytes outside this binding survive.
		g_ww_ownership_unscanned.fetch_add(1, std::memory_order_relaxed);
	}
	if (!is_written && !is_texel_buffer) {
		const uint64_t current_epoch  = m_scheduler.Context().BdaSyncEpoch();
		const uint64_t current_faults = m_memory_tracker.CpuWriteFaultCount();
		if (buffer.last_sync_epoch == current_epoch &&
		    buffer.last_sync_fault_count == current_faults &&
		    buffer.synced_vaddr <= vaddr &&
		    (vaddr + size) <= (buffer.synced_vaddr + buffer.synced_size)) {
			m_last_upload_bytes = 0;
			m_last_upload_pages = 0;
			return false;
		}
	}

	std::vector<vk::BufferCopy> copies;
	std::vector<const void*>    sources; // per copy: hot-page snapshot, or null = guest memory
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes, const void* snapshot) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    sources.push_back(snapshot);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, sources, total_size); });
	m_last_upload_bytes = total_size;
	m_last_upload_pages = copies.size();

	if (!is_written && !is_texel_buffer) {
		const uint64_t current_epoch  = m_scheduler.Context().BdaSyncEpoch();
		const uint64_t current_faults = m_memory_tracker.CpuWriteFaultCount();
		if (buffer.last_sync_epoch == current_epoch &&
		    buffer.last_sync_fault_count == current_faults &&
		    buffer.synced_size > 0 &&
		    vaddr <= (buffer.synced_vaddr + buffer.synced_size) &&
		    (vaddr + size) >= buffer.synced_vaddr) {
			const uint64_t new_start = std::min(buffer.synced_vaddr, vaddr);
			const uint64_t new_end   = std::max(buffer.synced_vaddr + buffer.synced_size, vaddr + size);
			buffer.synced_vaddr      = new_start;
			buffer.synced_size       = new_end - new_start;
		} else {
			buffer.last_sync_epoch       = current_epoch;
			buffer.last_sync_fault_count = current_faults;
			buffer.synced_vaddr          = vaddr;
			buffer.synced_size           = size;
		}
	} else if (is_written) {
		buffer.last_sync_epoch       = 0;
		buffer.last_sync_fault_count = 0;
		buffer.synced_vaddr          = 0;
		buffer.synced_size           = 0;
	}

	if (source) {
		m_sync_stats.copy_buffers++;
		buffer.last_gpu_write_tick = m_scheduler.CurrentTick(); // the staging copy below
		if (m_upload_batch_open && !is_texel_buffer) {
			// Queued: EndUploadBatch records the copy behind one shared barrier pair. Texel buffers
			// stay immediate: their image refresh below must follow the copy.
			m_pending_uploads.push_back({source, buffer.Handle(), std::move(copies)});
			return false;
		}
		m_sync_stats.copy_batches++;
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		command.Hazards().transfer_wrote = true; // hazard tracker: copy into a cache buffer
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		{ GpuTimestampScope ts_scope(native, GpuTimestamps::Kind::BarrierBuffer); native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr); }
		{ GpuTimestampScope ts_scope(native, GpuTimestamps::Kind::CopyUploadStaging); if (ts_scope.Active()) ts_scope.SetBytes(SumCopySizes(copies)); native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data()); }
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		{ GpuTimestampScope ts_scope(native, GpuTimestamps::Kind::BarrierBuffer); native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr); }
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     std::span<const void* const> sources, uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, /*allow_wait=*/!m_upload_batch_open,
	    [&](uint8_t* mapped, uint64_t base_offset) {
		for (size_t i = 0; i < copies.size(); i++) {
			auto& copy = copies[i];
			if (i < sources.size() && sources[i] != nullptr) {
				std::memcpy(mapped + copy.srcOffset, sources[i], copy.size);
			} else {
				const auto address = buffer.CpuAddress() + copy.dstOffset;
				if (const void* ptr = LibKernel::Memory::TryGetBackingPointer(address, copy.size)) {
					std::memcpy(mapped + copy.srcOffset, ptr, copy.size);
				} else if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
					std::memcpy(mapped + copy.srcOffset, reinterpret_cast<const void*>(address), copy.size);
				}
			}
			copy.srcOffset += base_offset;
		}
	});
	return source.first->Handle();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id, bool allow_stream) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (allow_stream && !is_written && size <= CACHING_PAGESIZE &&
	    m_memory_tracker.IsRegionCpuModifiedNotGpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (const void* ptr = LibKernel::Memory::TryGetBackingPointer(vaddr, size)) {
				std::memcpy(mapped, ptr, size);
			} else if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				std::memcpy(mapped, reinterpret_cast<const void*>(vaddr), size);
			}
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	m_sync_stats.obtain_calls++;
	m_sync_stats.obtain_bytes += m_last_upload_bytes;
	m_sync_stats.obtain_pages += m_last_upload_pages;
	if (is_written) {
		m_gpu_modified_ranges.Add(vaddr, size);
		NoteGpuWrite(vaddr, size, m_scheduler.CurrentTick());
		buffer.last_gpu_write_tick  = m_scheduler.CurrentTick();
		buffer.last_sync_epoch       = 0;
		buffer.last_sync_fault_count = 0;
		buffer.synced_vaddr          = 0;
		buffer.synced_size           = 0;
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    /*allow_wait=*/true,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	static const bool fill_log = [] {
		const char* value = std::getenv("KYTY_PROBE_READBACK_LOG");
		return value != nullptr && value[0] == '1';
	}();
	if (fill_log) {
		static uint64_t fills = 0;
		if (fills++ < 6000 || fills % 100 == 0) {
			LOGF("FillBuffer: 0x%016" PRIx64 "+0x%" PRIx64 " value=0x%08x gpu_dirty=%d\n", vaddr,
			     size, value, IsRegionGpuModified(vaddr, size) ? 1 : 0);
		}
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	// 2026-10-13: a DMA fill that covers a cached image exactly is a clear of that image. Clear the
	// VkImage directly and keep it GPU-owned, like the compute-fill shortcut does, instead of
	// invalidating it: the old path re-uploaded the whole surface from guest memory at the next bind
	// (ng-59: 5000 uploads of the 2180x1228 scene colour in 150 s, each over GPU pixels). The buffer
	// copy still receives the value for CPU readers. OPT-IN (KYTY_DMA_FILL_IMAGE_CLEAR=1): the shortcut
	// clears only the exact-size image and leaves other images overlapping the range (e.g. a smaller
	// stencil allocation at the same base) stale (Codex review 2026-10-13), and ng-62 showed it does not
	// reduce the uploads over GPU-owned images; alias coherence must be handled before it can default on.
	static const bool clear_images = [] {
		const char* value = std::getenv("KYTY_DMA_FILL_IMAGE_CLEAR");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	if (clear_images && m_texture_cache.ClearImageFromBuffer(m_scheduler.Current(), vaddr, size, value,
	                                                         /*decline_if_aliased=*/true)) {
		auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
		dst->Fill(dst_offset, size, value);
		return;
	}
	// A CPU fill into a GPU-owned image's memory would fault into InvalidateCpuAliases and lose the
	// image's pixels; route it through the GPU so the merge below applies (review 2026-10-13, D5).
	if (!IsRegionGpuModified(vaddr, size) && !m_texture_cache.HasGpuOwnedImageIn(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size, /*write_pending=*/true, {},
	    TextureCache::GpuWriteCoverage::Exact);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size) &&
	    !m_texture_cache.HasGpuOwnedImageIn(dst_vaddr, size)) {
		// RenderContext::StoreGuestMemory: the tracker learns of the store at once, as from a write
		// fault, and kernel write-watch does not mistake it for a guest CPU write.
		m_scheduler.Context().StoreGuestMemory(dst_vaddr, reinterpret_cast<const void*>(src_vaddr), size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size, /*write_pending=*/true, {},
		    TextureCache::GpuWriteCoverage::Exact);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	PrepareHostRead(vaddr, size);
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	PrepareHostRead(vaddr, size);
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

void BufferCache::QueueGpuModified(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_pending_gpu_modified_mutex);
	m_pending_gpu_modified.emplace_back(vaddr, size);
	m_has_pending_gpu_modified.store(true, std::memory_order_release);
}

void BufferCache::DrainPendingGpuModified() {
	if (!m_has_pending_gpu_modified.load(std::memory_order_acquire)) {
		return;
	}
	std::vector<std::pair<uint64_t, uint64_t>> pending;
	{
		std::lock_guard lock(m_pending_gpu_modified_mutex);
		pending.swap(m_pending_gpu_modified);
		m_has_pending_gpu_modified.store(false, std::memory_order_release);
	}
	for (const auto [vaddr, size]: pending) {
		m_gpu_modified_ranges.Add(vaddr, size);
		NoteGpuWrite(vaddr, size, 0); // marked by a priority operation after the GPU finished
		m_pending_texture_invalidations.emplace_back(vaddr, size);
	}
}

void BufferCache::NoteGpuWrite(uint64_t vaddr, uint64_t size, uint64_t tick) {
	const auto first = vaddr >> CACHING_PAGEBITS;
	const auto last  = (vaddr + size - 1) >> CACHING_PAGEBITS;
	if (last - first >= 8192) {
		m_bar_block_tick = std::max(m_bar_block_tick, tick); // too many pages: block until done
		return;
	}
	for (auto page = first; page <= last; ++page) {
		// Monotonic: a later marking with an older (or 0 = complete) tick must not hide a write
		// still in flight on the same 16 KiB entry (stale-buffer workflow 2026-10-13, fix 2).
		if (!m_gpu_write_ticks.Note(page, tick)) {
			m_bar_block_tick = std::max(m_bar_block_tick, tick); // outside the guest range: block until done
		}
	}
}

bool BufferCache::TryBarReadback(const Buffer& buffer, std::span<const vk::BufferCopy> copies) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BAR_READBACK");
		return value == nullptr || !(value[0] == '0' && value[1] == '\0');
	}();
	m_last_bar_reason = 0;
	if (!enabled || buffer.Mapped().empty() || copies.empty()) {
		m_last_bar_reason = 1;
		return false;
	}
	const auto address = buffer.CpuAddress();
	// The join/initial copies into this buffer must have executed, not only the page's writes.
	// KYTY_BAR_READY_GATE=0 drops that condition (A/B of its cost only: it re-opens the lost
	// download of the stale-buffer workflow, so it must never be the play default).
	static const bool ready_gate = [] {
		const char* value = std::getenv("KYTY_BAR_READY_GATE");
		return value == nullptr || !(value[0] == '0' && value[1] == '\0');
	}();
	auto latest = std::max(m_bar_block_tick, ready_gate ? buffer.ready_tick : 0);
	for (const auto& copy: copies) {
		const auto first = (address + copy.srcOffset) >> CACHING_PAGEBITS;
		const auto last  = (address + copy.srcOffset + copy.size - 1) >> CACHING_PAGEBITS;
		for (auto page = first; page <= last; ++page) {
			const auto* stamp = m_gpu_write_ticks.Find(page);
			if (stamp == nullptr) {
				m_last_bar_reason = 2;
				return false; // marked by a path without a tick: keep the copy path
			}
			latest = std::max(latest, *stamp);
		}
	}
	if (latest != 0 && !m_scheduler.IsFree(latest)) {
		// Still in flight, or recorded into the open command buffer (not even submitted).
		m_last_bar_reason = latest >= m_scheduler.CurrentTick() ? 4 : 3;
		return false;
	}
	// The mapping is uncached for CPU reads (no HOST_CACHED): stream it into a cached bounce
	// buffer (writeCombinedCopy.h), then write the guest backing from there.
	const bool streaming = WriteCombinedCopyEnabled();
	const auto copy_begin = ProbeEnabled::Get() ? std::chrono::steady_clock::now()
	                                            : std::chrono::steady_clock::time_point {};
	constexpr uint64_t kBounceBytes = 256 * 1024;
	for (const auto& copy: copies) {
		if (!buffer.IsCoherent()) {
			const_cast<Buffer&>(buffer).Invalidate(copy.srcOffset, copy.size);
		}
		for (uint64_t done = 0; done < copy.size;) {
			const auto bytes = std::min<uint64_t>(copy.size - done, kBounceBytes);
			if (m_bar_bounce.size() < bytes) {
				m_bar_bounce.resize(kBounceBytes);
			}
			CopyFromWriteCombined(m_bar_bounce.data(), buffer.Mapped().data() + copy.srcOffset + done, bytes,
			                      streaming);
			Libs::LibKernel::Memory::WriteBacking(address + copy.srcOffset + done, m_bar_bounce.data(), bytes);
			done += bytes;
		}
		m_sync_stats.bar_readback_bytes += copy.size;
	}
	if (ProbeEnabled::Get()) {
		m_sync_stats.bar_readback_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - copy_begin)
		        .count());
	}
	m_sync_stats.bar_readbacks++;
	m_last_download_bar = true;
	return true;
}

void BufferCache::PublishPendingTextureInvalidations() {
	if (m_pending_texture_invalidations.empty()) {
		return;
	}
	std::vector<std::pair<uint64_t, uint64_t>> pending;
	pending.swap(m_pending_texture_invalidations);
	// These writes already executed; every pending page of an image holds new data, so the seed
	// must skip all of them, not just the entry being published (review 2026-10-13, D3).
	for (const auto [vaddr, size]: pending) {
		m_texture_cache.InvalidateMemoryFromGPU(vaddr, size, /*write_pending=*/false, pending);
	}
}

void BufferCache::OnCommandBoundary() {
	FlushWriteFaults(false);
	DrainPendingGpuModified();
	PublishPendingTextureInvalidations();
	if (HasPendingCpuWrites()) {
		m_scheduler.Context().ApplyPendingCpuWrites();
	}
}

BufferCache::WriteWatchStats BufferCache::TakeWriteWatchStats() {
	WriteWatchStats stats;
	stats.ownership_unscanned = g_ww_ownership_unscanned.exchange(0, std::memory_order_relaxed);
	stats.fault_pass_collects = g_ww_fault_pass_collects.exchange(0, std::memory_order_relaxed);
	return stats;
}

void BufferCache::MarkPagesWritten(const uint64_t* encoded, uint32_t count) {
	if (WriteWatch::Get() == nullptr) {
		for (uint32_t index = 0; index < count; ++index) {
			MarkPageWritten(GuestAddress(encoded[index]));
		}
		return;
	}
	RangeSet pages;
	for (uint32_t index = 0; index < count; ++index) {
		pages.Add(GuestAddress(encoded[index]), WRITE_FAULT_PAGESIZE);
	}
	// A CPU write the kernel recorded but nobody collected yet wins, as a faulted one does.
	pages.ForEach([&](uint64_t start, uint64_t end) {
		g_ww_fault_pass_collects.fetch_add(1, std::memory_order_relaxed);
		CollectCpuWritesInline(start, end - start);
	});
	RangeSet marked;
	pages.ForEach([&](uint64_t start, uint64_t end) {
		for (auto page = start; page < end; page += WRITE_FAULT_PAGESIZE) {
			if (m_memory_tracker.TryMarkRegionAsGpuModified(page, WRITE_FAULT_PAGESIZE)) {
				marked.Add(page, WRITE_FAULT_PAGESIZE);
				continue;
			}
			static std::atomic<uint32_t> reported {0};
			if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("BufferCache: unbounded GPU write to CPU-dirty page 0x%016" PRIx64 " kept CPU data\n",
				     page);
			}
		}
	});
	// A CPU write between that collection and the read protection the marks applied predates the
	// GPU's ownership: the CPU wins the pages it wrote, the GPU keeps the rest.
	RangeSet written;
	marked.ForEach([&](uint64_t start, uint64_t end) {
		g_ww_fault_pass_collects.fetch_add(1, std::memory_order_relaxed);
		m_memory_tracker.CollectWrittenPages(start, end - start,
		                                     [&](uint64_t run, uint64_t bytes) { written.Add(run, bytes); });
	});
	written.ForEach([&](uint64_t start, uint64_t end) {
		m_memory_tracker.UnmarkRegionAsGpuModified(start, end - start);
		CollectCpuWritesApply(start, end - start);
		marked.Subtract(start, end - start);
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("BufferCache: CPU write in the marking window of 0x%016" PRIx64 " kept CPU data\n",
			     start);
		}
	});
	marked.ForEach([&](uint64_t start, uint64_t end) { QueueGpuModified(start, end - start); });
}

void BufferCache::CollectCpuWritesInline(uint64_t vaddr, uint64_t size) {
	if (WriteWatch::Get() == nullptr) {
		return;
	}
	m_memory_tracker.CollectWrittenPages(
	    vaddr, size, [this](uint64_t start, uint64_t bytes) { CollectCpuWritesApply(start, bytes); });
}

void BufferCache::CollectCpuWritesApply(uint64_t vaddr, uint64_t size) {
	const auto queue = [this](uint64_t address, uint64_t length, bool texture_only) {
		std::lock_guard lock(m_pending_cpu_writes_mutex);
		m_pending_cpu_writes.push_back({address, length, texture_only});
		m_has_pending_cpu_writes.store(true, std::memory_order_release);
	};
	m_memory_tracker.MarkCpuWrittenUnlessGpuModified(
	    vaddr, size, [&](uint64_t gpu_start, uint64_t gpu_bytes) { queue(gpu_start, gpu_bytes, false); });
	queue(vaddr, size, true);
}

std::vector<BufferCache::PendingCpuWrite> BufferCache::TakePendingCpuWrites() {
	std::vector<PendingCpuWrite> pending;
	std::lock_guard              lock(m_pending_cpu_writes_mutex);
	pending.swap(m_pending_cpu_writes);
	m_has_pending_cpu_writes.store(false, std::memory_order_release);
	return pending;
}

void BufferCache::MarkPageWritten(uint64_t vaddr) {
	// A CPU write the kernel recorded but nobody collected yet must win, as a faulted one does.
	CollectCpuWritesInline(vaddr, WRITE_FAULT_PAGESIZE);
	if (m_memory_tracker.TryMarkRegionAsGpuModified(vaddr, WRITE_FAULT_PAGESIZE)) {
		// Kernel write-watch: a CPU write between that collection and the read protection the mark
		// just applied predates the GPU's ownership; it must win too, not be downloaded over.
		if (WriteWatch::Get() != nullptr) {
			RangeSet written;
			m_memory_tracker.CollectWrittenPages(
			    vaddr, WRITE_FAULT_PAGESIZE,
			    [&](uint64_t start, uint64_t bytes) { written.Add(start, bytes); });
			if (!written.Empty()) {
				// The CPU wins the pages it wrote; the GPU keeps the rest of the fault page.
				RangeSet gpu;
				gpu.Add(vaddr, WRITE_FAULT_PAGESIZE);
				written.ForEach([&](uint64_t start, uint64_t end) {
					m_memory_tracker.UnmarkRegionAsGpuModified(start, end - start);
					CollectCpuWritesApply(start, end - start);
					gpu.Subtract(start, end - start);
				});
				gpu.ForEach([&](uint64_t start, uint64_t end) { QueueGpuModified(start, end - start); });
				static std::atomic<uint32_t> reported {0};
				if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
					LOGF("BufferCache: CPU write in the marking window of 0x%016" PRIx64 " kept CPU data\n",
					     vaddr);
				}
				return;
			}
		}
		QueueGpuModified(vaddr, WRITE_FAULT_PAGESIZE);
		return;
	}
	// The CPU wrote the page after the GPU did; the CPU data wins for that page.
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
		LOGF("BufferCache: unbounded GPU write to CPU-dirty page 0x%016" PRIx64 " kept CPU data\n",
		     vaddr);
	}
}

void BufferCache::MarkWriteFaultOverflow() {
	// More pages than the compress pass can list: fall back to marking every unbounded written
	// range, region by region (4 MiB), skipping regions with CPU-dirty pages.
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
		LOGF("BufferCache: write-fault list overflow, marking whole unbounded ranges\n");
	}
	m_write_fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
		for (auto address = start; address < end;) {
			const auto region_end = std::min(end, Common::AlignDown(address, TRACKER_REGION_SIZE) +
			                                          TRACKER_REGION_SIZE);
			CollectCpuWritesInline(address, region_end - address);
			if (m_memory_tracker.TryMarkRegionAsGpuModified(address, region_end - address)) {
				QueueGpuModified(address, region_end - address);
			}
			address = region_end;
		}
	});
}

void BufferCache::FlushWriteFaults(bool wait) {
	if (m_write_faults_dirty && m_scheduler.Active()) {
		// KYTY_WRITE_FAULT_TICK_AFTER=0 restores the pre-58c4a83 order (tick before the pass) for
		// the boot-freeze A/B.
		static const bool tick_after = [] {
			const char* env = std::getenv("KYTY_WRITE_FAULT_TICK_AFTER");
			return env == nullptr || env[0] != '0';
		}();
		m_write_faults_dirty = false;
		if (!tick_after) {
			m_write_fault_tick = m_scheduler.CurrentTick();
		}
		m_fault_manager.ProcessWriteFaultBuffer();
		// Taken after Process(): reusing a slot can submit the open buffer, which moves the
		// compress pass and its mark operation to the next tick (audit 2026-10-12: a host read
		// then waited one tick too early and parsed pages before they were protected).
		if (tick_after) {
			m_write_fault_tick = m_scheduler.CurrentTick();
		}
	}
	if (!wait || m_write_fault_tick == 0) {
		return;
	}
	const auto tick    = m_write_fault_tick;
	m_write_fault_tick = 0;
	WaitReasonScope reason(WaitReasonProbe::WriteFaults);
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	DrainPendingGpuModified();
}

void BufferCache::BeginUploadBatch() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_UPLOAD_BATCH");
		const bool  on    = value == nullptr || value[0] != '0';
		LOGF("BufferCache: upload batching %s\n", on ? "on" : "off");
		return on;
	}();
	// Serialized diagnostics flush after every operation; keep the per-buffer barriers there so a
	// flush can never split a batch from its staging data.
	if (!enabled || m_upload_batch_open || GpuDiagnostics::Serialize()) {
		return;
	}
	m_upload_batch_open = true;
	m_upload_batch_tick = m_scheduler.CurrentTick();
}

void BufferCache::EndUploadBatch() {
	if (!m_upload_batch_open) {
		return;
	}
	m_upload_batch_open = false;
	if (m_pending_uploads.empty()) {
		return;
	}
	// The queued copies' staging data is watched under the tick they were committed in; the
	// copies must execute in that same tick (UploadCopies never waits inside a batch).
	EXIT_IF(m_scheduler.CurrentTick() != m_upload_batch_tick);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto        native = command.Handle();
	command.Hazards().transfer_wrote = true; // hazard tracker: batched copies into cache buffers
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                       vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	{ GpuTimestampScope ts_scope(native, GpuTimestamps::Kind::BarrierGlobal); native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                       vk::DependencyFlagBits::eByRegion, 1, &before, 0, nullptr, 0, nullptr); }
	for (const auto& upload: m_pending_uploads) {
		{ GpuTimestampScope ts_scope(native, GpuTimestamps::Kind::CopyUploadStaging); if (ts_scope.Active()) ts_scope.SetBytes(SumCopySizes(upload.copies)); native.copyBuffer(upload.source, upload.destination, static_cast<uint32_t>(upload.copies.size()),
		                  upload.copies.data()); }
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	{ GpuTimestampScope ts_scope(native, GpuTimestamps::Kind::BarrierGlobal); native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 1, &after, 0, nullptr, 0, nullptr); }
	m_sync_stats.copy_batches++;
	m_pending_uploads.clear();
}

void BufferCache::PrepareHostRead(uint64_t vaddr, uint64_t size) {
	if (!m_has_pending_gpu_modified.load(std::memory_order_acquire) &&
	    !m_write_faults_dirty && m_write_fault_tick == 0) {
		return;
	}
	if ((m_write_faults_dirty || m_write_fault_tick != 0) &&
	    m_write_fault_ranges.Intersects(vaddr, size)) {
		FlushWriteFaults(true);
	}
	DrainPendingGpuModified();
}

bool BufferCache::PublishGpuWritesNoWait(uint64_t vaddr, uint64_t size) {
	const bool pending = HasPendingUnboundedWrites() && m_write_fault_ranges.Intersects(vaddr, size);
	if (pending) {
		FlushWriteFaults(false);
	}
	DrainPendingGpuModified();
	return pending;
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	OnCommandBoundary();
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			// An enrolled image over this buffer still holds the pixels: sync it in before the publish
			// drops the read protection (review 2026-10-14, C4). OnCommandBoundary ran above.
			uint64_t sync_vaddr = buffer.CpuAddress();
			uint64_t sync_size  = buffer.Size();
			if (m_texture_cache.SyncGpuImagesForHostRead(buffer, sync_vaddr, sync_size, /*record=*/true)) {
				NoteGpuWrite(sync_vaddr, sync_size, m_scheduler.CurrentTick());
			}
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto      completion_tick = m_scheduler.CurrentTick();
	WaitReasonScope reason(WaitReasonProbe::Gc);
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

bool BufferCache::ReadSyncRepassEnv() {
	const char* value = std::getenv("KYTY_SYNC_REPASS");
	const bool on = value == nullptr || value[0] != '0';
	LOGF("BufferCache: sync repass %s\n", on ? "on" : "off");
	return on;
}

// Calls func(owner, start, finish) for every cached buffer piece inside [from, to).
template <typename Func>
void BufferCache::ForEachSyncOwnerPiece(uint64_t from, uint64_t to, Func&& func) {
	uint64_t cursor = from;
	while (cursor < to) {
		const auto* owner = m_page_table.Find(cursor >> PageTable::kPageBits);
		if (owner == nullptr || !*owner) {
			cursor = Common::AlignDown(cursor, CACHING_PAGESIZE) + CACHING_PAGESIZE;
			continue;
		}
		auto&      buffer = m_slot_buffers[*owner];
		const auto start  = std::max(buffer.CpuAddress(), cursor);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), to);
		if (start < finish) {
			func(*owner, start, finish);
		}
		cursor = std::max(buffer.CpuAddress() + buffer.Size(),
		                  Common::AlignDown(cursor, CACHING_PAGESIZE) + CACHING_PAGESIZE);
	}
}

void BufferCache::CollectSyncOwners(uint64_t vaddr, uint64_t size) {
	m_sync_vaddr = vaddr;
	m_sync_size  = size;
	m_sync_candidates.clear();
	m_memory_tracker.ForEachSyncCandidateRun(
	    vaddr, size, [this](uint64_t address, uint64_t bytes) noexcept {
		    m_sync_candidates.emplace_back(address, bytes);
	    });
	// One synchronization per owning buffer: the hull of its candidate runs. ForEachUploadRange
	// only emits the dirty/hot pages inside it, so the hull costs a bit-scan, not uploads.
	m_sync_owners.clear();
	for (const auto [run_address, run_size]: m_sync_candidates) {
		ForEachSyncOwnerPiece(run_address, run_address + run_size,
		                      [this](BufferId id, uint64_t start, uint64_t finish) {
			                      auto found = m_sync_owners.find(id.index);
			                      if (found == m_sync_owners.end()) {
				                      m_sync_owners.emplace(id.index, SyncOwner {id, start, finish});
			                      } else {
				                      found->second.start  = std::min(found->second.start, start);
				                      found->second.finish = std::max(found->second.finish, finish);
			                      }
		                      });
	}
}

void BufferCache::SyncCollectedOwners() {
	const auto sync_one = [this](BufferId id, uint64_t start, uint64_t finish) {
		auto& buffer = m_slot_buffers[id];
		(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		m_sync_stats.range_calls++;
		m_sync_stats.range_bytes += m_last_upload_bytes;
		m_sync_stats.range_pages += m_last_upload_pages;
		m_range_bytes_total += m_last_upload_bytes;
	};
	m_sync_done.clear();
	for (const auto& [index, hull]: m_sync_owners) {
		sync_one(hull.id, hull.start, hull.finish);
		m_sync_done.emplace_back(hull.start, hull.finish);
	}
	if (!m_sync_repass) {
		return;
	}
	std::sort(m_sync_done.begin(), m_sync_done.end());
	// Up to two more passes. A page that became CPU-dirty after the first snapshot, outside every
	// hull already synced, would wait for the next sync epoch; the buffer walk (which samples each
	// buffer when it visits it) would have uploaded it. Hot pages stay dirty and reappear inside
	// the hulls already synced, where they are subtracted: stop when nothing remains, not when
	// there are no candidates. One SynchronizeBuffer per (owner, segment), never per merged hull,
	// so a hot page is visited (and its unchanged counter advanced) once per sync.
	for (int pass = 2; pass <= 3; ++pass) {
		m_sync_candidates.clear();
		m_memory_tracker.ForEachSyncCandidateRun(
		    m_sync_vaddr, m_sync_size, [this](uint64_t address, uint64_t bytes) noexcept {
			    m_sync_candidates.emplace_back(address, bytes);
		    });
		m_sync_pieces.clear();
		for (const auto [run_address, run_size]: m_sync_candidates) {
			uint64_t       cursor  = run_address;
			const uint64_t run_end = run_address + run_size;
			auto           done    = std::upper_bound(
                m_sync_done.begin(), m_sync_done.end(), std::make_pair(cursor, UINT64_MAX));
			if (done != m_sync_done.begin()) {
				--done; // the hull starting at or before cursor may cover it
			}
			const auto add_gap = [this](uint64_t from, uint64_t to) {
				ForEachSyncOwnerPiece(from, to, [this](BufferId id, uint64_t start, uint64_t finish) {
					m_sync_pieces.push_back({id, start, finish});
				});
			};
			for (; done != m_sync_done.end() && cursor < run_end; ++done) {
				if (done->second <= cursor) {
					continue;
				}
				if (done->first > cursor) {
					add_gap(cursor, std::min(done->first, run_end));
				}
				cursor = std::max(cursor, done->second);
			}
			if (cursor < run_end) {
				add_gap(cursor, run_end);
			}
		}
		if (m_sync_pieces.empty()) {
			break;
		}
		m_sync_stats.range_repasses++;
		for (const auto& piece: m_sync_pieces) {
			sync_one(piece.id, piece.start, piece.finish);
			m_sync_stats.range_repass_ranges += m_last_upload_pages;
			m_sync_done.emplace_back(piece.start, piece.finish);
		}
		std::sort(m_sync_done.begin(), m_sync_done.end());
	}
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	// Page-driven (default): the tracker lists the CPU-dirty and hot page runs of the range and only
	// their owning buffers are synchronized. The historical loop visited every cached buffer in the
	// range: in the first mission that was ~29,000 buffer visits per sync, 13 syncs per second, for
	// ~2,200 non-empty uploads (Codex perf-sync-30 counters; ng-68 profile: tree walk + slot lookups
	// ~8.5% of CPU). KYTY_SYNC_PAGE_DRIVEN=0 restores the buffer walk for A/B.
	static const bool page_driven = [] {
		const char* value = std::getenv("KYTY_SYNC_PAGE_DRIVEN");
		const bool  on    = value == nullptr || value[0] != '0';
		LOGF("BufferCache: %s range synchronization\n", on ? "page-driven" : "buffer-walk");
		return on;
	}();
	if (page_driven) {
		CollectSyncOwners(vaddr, size);
		SyncCollectedOwners();
		return;
	}
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
			m_sync_stats.range_calls++;
			m_sync_stats.range_bytes += m_last_upload_bytes;
			m_sync_stats.range_pages += m_last_upload_pages;
			m_range_bytes_total += m_last_upload_bytes;
		}
	}
}

bool BufferCache::HasGpuWriteStamp(uint64_t vaddr) const {
	return m_gpu_write_ticks.Contains(vaddr >> CACHING_PAGEBITS);
}

uint64_t BufferCache::OwnerReadyTick(uint64_t vaddr) const {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) {
		return 0;
	}
	return m_slot_buffers[*owner].ready_tick;
}

bool BufferCache::PeekDeviceCopy(uint64_t vaddr, void* out, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) {
		return false;
	}
	auto& buffer = m_slot_buffers[*owner];
	if (buffer.Mapped().empty() || !buffer.IsInBounds(vaddr, size)) {
		return false;
	}
	const auto offset = buffer.Offset(vaddr);
	if (!buffer.IsCoherent()) {
		buffer.Invalidate(offset, size);
	}
	std::memcpy(out, buffer.Mapped().data() + offset, size);
	return true;
}

std::string BufferCache::DescribeBufferAt(uint64_t vaddr) {
	const auto  page  = vaddr & ~uint64_t {0xfff};
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	std::string text;
	if (owner == nullptr || !*owner) {
		text = "none";
	} else {
		const auto& buffer = m_slot_buffers[*owner];
		text = fmt::format("{}buf[0x{:x}+0x{:x} score={}]", IsBufferInvalid(*owner) ? "invalid-" : "",
		                   buffer.CpuAddress(), buffer.Size(), buffer.StreamScore());
	}
	// Tracker state of the faulting page helps tell stale protection from live traffic.
	if (m_memory_tracker.IsRegionCpuModified(page, 0x1000)) text += " cpu-dirty";
	if (m_memory_tracker.IsRegionGpuModified(page, 0x1000)) text += " gpu-dirty";
	return text;
}

} // namespace Libs::Graphics
