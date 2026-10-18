#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	// False before InitializeGpu and after shutdown (unit tests that never start the GPU thread):
	// paths that need SendCommandSync, such as CPU readback enrolment, must stay inactive then.
	[[nodiscard]] bool HasGpu() const noexcept { return m_gpu != nullptr; }
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	Common::Mutex&      GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	void               PrepareBda();
	// Uploads CPU writes (pending regions + hot pages) once per command stream; used before
	// pointer-using draws and before whole-heap descriptor reads.
	void               SyncCpuWrites();
	// Called when the command processor starts (or resumes) a guest command stream. Under kernel
	// write-watch, collects the CPU writes first when a guest submission arrived since the last
	// collection (its CPU writes precede the submission).
	void               BeginCommandStream();
	// Kernel write-watch, GPU thread at a safe point: collects when a guest submission arrived since
	// the last collection, else applies queued writes (BeginCommandStream, CPU flip preparation).
	void               CollectSubmittedCpuWrites();
	// Game threads: a guest submission was queued (GuestGpu::Enqueue).
	void               NoteGuestSubmit() noexcept {
		m_guest_submits.fetch_add(1, std::memory_order_acq_rel);
	}
	// Kernel write-watch (writeWatch.h; no-op without it). Applies the CPU writes the kernel
	// recorded since their last collection, as the write faults they replace would have (buffer
	// tracker with download of GPU-owned pages, texture cache), plus the writes queued by
	// BufferCache::CollectCpuWritesInline. size == 0: every span holding write-watched pages,
	// label pages first. GPU thread at a safe point (no cache lock held). Returns pages applied.
	uint64_t           CollectCpuWrites(uint64_t vaddr = 0, uint64_t size = 0);
	uint64_t           ApplyPendingCpuWrites();
	// Command-processor stores to guest memory (EOP labels, flips, clock values; GPU thread).
	// Under kernel write-watch they go through the unregistered backing alias: through a guest
	// view the kernel would report them as guest CPU writes, every wait on such a label would
	// collect, and the tracker would hear of them only at the next collection. The tracker is told
	// first, as the write fault of a direct store would have told it.
	void               StoreGuestMemory(uint64_t vaddr, const void* data, uint64_t size);
	[[nodiscard]] uint64_t BdaSyncEpoch() const noexcept { return m_bda_sync_epoch; }
	// Signals the start of a new frame / boundary (FlipPreparation, SuspendPoint).
	// Throttles unflagged hot-region scanning in SyncCpuWrites to once per frame.
	void               NoteFrameBoundary() noexcept {
		m_sync_hot_regions_pending = true;
	}
	// The command processor observed guest memory without suspending (a WAIT_REG_MEM that passed,
	// a conditional-IB compare, COND_EXEC, SET_PREDICATION): CPU writes made before the value it
	// saw must be visible to later BDA / unbounded consumers in this stream. Starts a new sync
	// epoch when the current one was already consumed. KYTY_SYNC_ON_WAIT_PASS: 0 (default) only
	// counts, 1 bumps the epoch. label_vaddr/label_size: the guest memory the command processor
	// read (0 for ACQUIRE_MEM, which orders nothing against the CPU). Under kernel write-watch a
	// label page written since the last collection began triggers a collection (and a new epoch
	// when it found CPU writes), so data released by that label is visible like faulted writes.
	void               NoteGuestAcquire(uint64_t label_vaddr = 0, uint64_t label_size = 0);
	// Test hook: overrides the KYTY_SYNC_ON_WAIT_PASS mode read from the environment.
	static void        SetSyncOnWaitPassMode(int mode);
	void               RunGarbageCollector();

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	GraphicContext&           m_graphics;
	Common::Mutex             m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	// How a CPU write reaches the caches: a collected run (page granular: somewhere in each page),
	// its texture half only (the tracker half was applied inline), or exact bytes before a store.
	enum class CpuWrite { Pages, PagesTextureOnly, Exact };
	uint64_t                  ApplyCpuWrite(uint64_t vaddr, uint64_t size, CpuWrite kind);
	void                      NoteWaitPass();
	std::atomic<uint64_t>     m_guest_submits {1};
	uint64_t                  m_collected_submits = 0;
	uint64_t                  m_bda_sync_epoch    = 1;
	uint64_t                  m_bda_synced_epoch  = 0;
	bool                      m_wait_pass_pending = false; // a passing wait bumped a consumed epoch
	bool                      m_sync_hot_regions_pending = true;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	bool                      m_fault_process_pending = false;
	bool                      m_bda_logged = false;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
