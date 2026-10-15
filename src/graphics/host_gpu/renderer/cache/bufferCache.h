#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/hostResidentMemory.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <atomic>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	// The fault buffer holds two page bitmaps: BDA accesses to uncached pages (16 KiB pages), then
	// pages written through unbounded (whole-heap) buffer descriptors at tracker granularity
	// (4 KiB) so counters do not falsely dirty neighbouring metadata and CPU data.
	static constexpr uint32_t WRITE_FAULT_PAGEBITS   = 12;
	static constexpr uint64_t WRITE_FAULT_PAGESIZE   = uint64_t {1} << WRITE_FAULT_PAGEBITS;
	static constexpr uint64_t WRITE_FAULT_NUMPAGES   = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> WRITE_FAULT_PAGEBITS;
	static constexpr uint32_t WRITE_FAULT_WORD_OFFSET = static_cast<uint32_t>(CACHING_NUMPAGES / 32);
	static constexpr uint64_t FAULT_BUFFER_BYTES     = CACHING_NUMPAGES / 8 + WRITE_FAULT_NUMPAGES / 8;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// A host-side label store from a non-GPU thread: tracker-only CPU-modified mark (no readback,
	// no scheduler access), safe from the priority thread.
	void MarkLabelWritten(uint64_t vaddr, uint64_t size) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
	// Companion query for the same path. BufferCache::IsRegionGpuModified first runs
	// PrepareHostRead (write-fault flush, pending-GPU-modified drain into the RangeSet maps): GPU
	// thread only. Calling it from the priority thread corrupted m_gpu_modified_ranges (ng-33,
	// std::_Rb_tree_rebalance_for_erase on a null node). The tracker itself locks per region and
	// the write-fault pass marks it before any later priority operation runs, so it is enough.
	[[nodiscard]] bool IsLabelRegionGpuModified(uint64_t vaddr, uint64_t size) {
		return m_memory_tracker.IsRegionGpuModified(vaddr, size);
	}

	// Write-fault entry: InvalidateMemory with the fault-ahead window (see MemoryTracker).

	void InvalidateWriteFault(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	// Non-null only under KYTY_HOST_RESIDENT=1 (prototype); see HostResidentMemory.
	[[nodiscard]] HostResidentMemory* GetHostResident() noexcept { return m_host_resident.get(); }
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	// Unbounded-descriptor write tracking. Shaders set a bit per written guest page in the fault
	// buffer's write half; FlushWriteFaults records the compress pass and a priority operation
	// marks those pages GPU-modified once the GPU is done (instead of protecting the whole
	// 528 MiB clamped range at bind time, which made every CPU access to the heap fault).
	void NoteUnboundedWrite(uint64_t vaddr, uint64_t size) { m_write_fault_ranges.Add(vaddr, size); }
	void MarkWriteFaultsDirty() { m_write_faults_dirty = true; }
	// Diagnostics (GPU thread): declared unbounded-write range membership and the last tick a
	// tracked GPU write touched the page (0 = none recorded).
	[[nodiscard]] bool InUnboundedWriteRange(uint64_t vaddr, uint64_t size) const {
		return m_write_fault_ranges.Intersects(vaddr, size);
	}
	[[nodiscard]] uint64_t LastGpuWriteTick(uint64_t vaddr) const {
		const auto it = m_gpu_write_ticks.find(vaddr >> CACHING_PAGEBITS);
		return it != m_gpu_write_ticks.end() ? it->second : 0;
	}
	// Unbounded-heap GPU writes recorded or in flight whose pages are not yet marked GPU-dirty.
	// Host-resident eligibility (review 2026-10-16 F1): tracker-only ownership query; a range inside a
	// declared unbounded-write heap with records pending falls back, because only the drain in
	// PrepareHostRead would make its GPU ownership current, and that drain must not run per bind.
	[[nodiscard]] bool HostResidentEligible(uint64_t vaddr, uint64_t size) {
		if (HasPendingUnboundedWrites() && InUnboundedWriteRange(vaddr, size)) {
			return false;
		}
		return !IsLabelRegionGpuModified(vaddr, size);
	}
	[[nodiscard]] bool HasPendingUnboundedWrites() const noexcept {
		return m_write_faults_dirty || m_write_fault_tick != 0;
	}
	void FlushWriteFaults(bool wait);
	// Before the host reads guest memory that unbounded writes may have touched.
	void PrepareHostRead(uint64_t vaddr, uint64_t size);
	void MarkPageWritten(uint64_t vaddr);      // any thread (fault-processing operation)
	void MarkWriteFaultOverflow();             // any thread
	// Safe point between PM4 packets / at garbage collection: records the pending write-fault
	// pass, publishes marked pages and deferred texture invalidations.
	void OnCommandBoundary();
	// Diagnostics (GPU thread): the cached buffer covering vaddr, and upload statistics by path.
	[[nodiscard]] std::string DescribeBufferAt(uint64_t vaddr);
	// Stale-buffer probe (GPU thread): whether the 16 KiB entry has a write stamp at all, the
	// owning buffer's ready tick, and the owner's current device bytes (BAR-mapped buffers only;
	// the caller waits for the GPU first).
	// Per-16 KiB-page tick of the last recorded GPU write (BAR readback gate); callable by the
	// renderer when it commits written bindings.
	void NoteGpuWrite(uint64_t vaddr, uint64_t size, uint64_t tick);
	[[nodiscard]] bool     HasGpuWriteStamp(uint64_t vaddr) const;
	[[nodiscard]] uint64_t OwnerReadyTick(uint64_t vaddr) const;
	bool                   PeekDeviceCopy(uint64_t vaddr, void* out, uint64_t size);
	struct SyncStats {
		uint64_t obtain_calls = 0, obtain_bytes = 0, obtain_pages = 0;
		uint64_t range_calls = 0, range_bytes = 0, range_pages = 0;
		uint64_t download_calls = 0, download_wait_ns = 0, read_memory_calls = 0;
		uint64_t copy_batches = 0; // barrier pairs recorded around staging copies
		uint64_t copy_buffers = 0; // SynchronizeBuffer calls that recorded or queued a staging copy
		uint64_t bar_readbacks = 0;
		uint64_t host_joins = 0; // replaced buffers copied on the host (complete)
		uint64_t gpu_joins  = 0; // replaced buffers copied on the GPU (in flight)
	};
	[[nodiscard]] SyncStats TakeSyncStats() { return std::exchange(m_sync_stats, {}); }
	void SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Upload batching (CPU-write synchronization): between Begin and End, SynchronizeBuffer queues
	// its staging copies instead of recording each with its own full-pipeline barrier pair; End
	// records one barrier pair around all of them. Nothing may read a queued buffer on the GPU
	// before End (SyncCpuWrites only synchronizes, so it is the only user). KYTY_UPLOAD_BATCH=0
	// restores the per-buffer barriers.
	void BeginUploadBatch();
	void EndUploadBatch();
	// Tracker regions whose CPU-dirty state changed since the last call: func(address, size).
	// Callers clip them to mapped guest memory before synchronizing (buffers may extend past it).
	template <typename Func>
	void ForEachBdaSyncPendingRegion(Func&& func) {
		m_memory_tracker.ForEachBdaSyncPendingRegion(std::forward<Func>(func));
	}
	void               RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	// Returns true when the overlap's content had to be copied on the GPU (recorded, not yet
	// executed); false when it was copied on the host through the BAR mappings because every byte
	// of it was already complete (no GPU write in flight, see HostJoinPossible).
	bool JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] bool HostJoinPossible(const Buffer& new_buffer, const Buffer& overlap);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      std::span<const void* const> sources, uint64_t total_size);
	void QueueGpuModified(uint64_t vaddr, uint64_t size); // any thread
	// Last GPU-write tick per 16 KiB page (0 = known complete), for TryBarReadback.
	// Copies completed GPU data straight from host-visible device memory (no queue drain).
	[[nodiscard]] bool TryBarReadback(const Buffer& buffer, std::span<const vk::BufferCopy> copies);
	// GPU thread. Only touches m_gpu_modified_ranges: it can run under the texture cache lock
	// (TextureCache::FindImage asks HasGpuDirtyBytes), so texture invalidations are deferred.
	void DrainPendingGpuModified();
	void PublishPendingTextureInvalidations(); // GPU thread, never under the texture cache lock
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	// Scratch for page-driven range synchronization (GPU thread): candidate page runs and the
	// per-buffer hull of the runs it owns.
	std::vector<std::pair<uint64_t, uint64_t>>                     m_sync_candidates;
	struct PendingUpload {
		vk::Buffer                  source;
		vk::Buffer                  destination;
		std::vector<vk::BufferCopy> copies;
	};
	std::vector<PendingUpload> m_pending_uploads;
	bool                       m_upload_batch_open = false;
	uint64_t                   m_upload_batch_tick = 0;
	struct SyncOwner {
		BufferId id;
		uint64_t start  = 0;
		uint64_t finish = 0;
	};
	std::unordered_map<uint32_t, SyncOwner>                        m_sync_owners; // keyed by slot index
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	std::unique_ptr<HostResidentMemory>               m_host_resident;
	TextureCache&                                     m_texture_cache;
	RangeSet                                          m_write_fault_ranges;
	bool                                              m_write_faults_dirty = false;
	uint64_t                                          m_write_fault_tick   = 0;
	std::mutex                                        m_pending_gpu_modified_mutex;
	std::vector<std::pair<uint64_t, uint64_t>>        m_pending_gpu_modified;
	std::vector<std::pair<uint64_t, uint64_t>>        m_pending_texture_invalidations;
	SyncStats                                         m_sync_stats;
	std::unordered_map<uint64_t, uint64_t>            m_gpu_write_ticks;
	uint64_t                                          m_bar_block_tick = 0;
	bool                                              m_last_download_bar = false;
	int                                               m_last_bar_reason   = 0; // 1 unmapped, 2 no tick, 3 in flight
	uint64_t                                          m_last_upload_bytes = 0;
	uint64_t                                          m_last_upload_pages = 0;
	std::atomic<bool>                                 m_has_pending_gpu_modified {false};
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
