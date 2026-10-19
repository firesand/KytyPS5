#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/hostReadStats.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <array>
#include <atomic>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;

struct DrawIndirectArgs {
	uint32_t vertex_count_per_instance;
	uint32_t instance_count;
	uint32_t start_vertex_location;
	uint32_t start_instance_location;
};

struct DrawIndexedIndirectArgs {
	uint32_t index_count_per_instance;
	uint32_t instance_count;
	uint32_t start_index_location;
	uint32_t base_vertex_location;
	uint32_t start_instance_location;
};

// KYTY_CP_GPU_INDIRECT=1: indirect draws read their argument records on the GPU
// (vkCmdDraw[Indexed]Indirect[Count]) instead of through host reads that wait for the producing
// dispatch. Default off.
static bool GpuIndirectEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CP_GPU_INDIRECT");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->Reset();
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		command();
	}
}

void GuestGpu::Wake() {
	Common::LockGuard lock(m_queue_mutex);
	m_work_available.Signal();
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	Submission submission;
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	m_suspend_point_ready->acquire();
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

void GuestGpu::ReportBlockedQueues(double seconds) {
	// m_queue_mutex is held by the caller (the GPU thread's scheduling loop).
	for (uint32_t id = 0; id < QueueCount; id++) {
		const auto& queue = m_queues[id];
		if (queue.empty()) {
			continue;
		}
		const auto& submission = queue.front();
		const auto& why        = GetProcessor(id).LastSuspend();
		uint64_t    value      = 0;
		if (why.addr != 0 && (why.size == sizeof(uint32_t) || why.size == sizeof(uint64_t))) {
			// Backing read: a page-fault download must not run under m_queue_mutex (diagnostic).
			(void)LibKernel::Memory::TryReadBacking(why.addr, &value, why.size);
		}
		const auto where = why.addr != 0 ? m_renderer.GetBufferCache().DescribeBufferAt(why.addr)
		                                 : std::string {};
		LOGF("GPU blocked: queue %u type %d (%zu queued, started %d) for %.0f s: %s addr=0x%016" PRIx64
		     " func=%u ref=0x%" PRIx64 " mask=0x%" PRIx64 " value=0x%" PRIx64 " %s\n",
		     id, static_cast<int>(submission.type), queue.size(), submission.started ? 1 : 0, seconds,
		     why.kind != nullptr ? why.kind : "?", why.addr, why.func, why.ref, why.mask, value,
		     where.c_str());
	}
	std::printf("GPU blocked: every queue blocked for %.0f s (details in the log)\n", seconds);
	std::fflush(stdout);
	Log::Flush();
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances                    = 1;
	m_sgpr_poison                      = {};
	m_merge_probe                      = {};
	m_submit_ahead                     = {};
	m_any_sgpr_poison                  = false;
	m_num_instances_stale              = false;
	m_predicate_skip                   = false;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

void CommandProcessor::BufferInit() {
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	GetScheduler().Flush();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4("wait_ce", 0, m_de_count, m_ce_count);
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4("wait_de_diff", 0, diff, m_ce_count - m_de_count);
	}
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4("wait_rewind");
	}
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	memcpy(dst, m_const_ram + offset / 4, static_cast<size_t>(dw_num) * 4);
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	if (!TestWaitRegMemValue(*addr, ref, mask, func)) {
		SuspendPm4("wait_reg_mem", reinterpret_cast<uint64_t>(addr), ref, mask, func, sizeof(T));
	} else if (func != 0) {
		// The resume path of a suspended wait starts a new epoch; a wait that passes does not.
		// func 0 ("always") observes no value, so it is not an acquire.
		NoteGuestAcquire(reinterpret_cast<uint64_t>(addr), sizeof(T));
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	// Through RenderContext::StoreGuestMemory: the tracker learns of the store at once, as from the
	// write fault of a direct store, and kernel write-watch does not mistake it for a guest write.
	if (write_one_address) {
		for (uint32_t i = 0; i < dw_num; i++) {
			m_renderer.StoreGuestMemory(reinterpret_cast<uint64_t>(dst), &src[i], sizeof(uint32_t));
		}
	} else {
		m_renderer.StoreGuestMemory(reinterpret_cast<uint64_t>(dst), src,
		                            static_cast<uint64_t>(dw_num) * sizeof(uint32_t));
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	m_renderer.StoreGuestMemory(dst_address, &value, num_bytes); // little-endian low bytes
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::PrepareHostRead(uint64_t vaddr, uint64_t size) {
	// KYTY_CP_HOST_READ_PREPARE=0 restores the pre-9d3eaab behaviour (no write-fault flush before
	// the CP's own guest reads) for bisecting the title-screen hang.
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CP_HOST_READ_PREPARE");
		return value == nullptr || value[0] != '0';
	}();
	if (!enabled) {
		return;
	}
	m_renderer.GetBufferCache().PrepareHostRead(vaddr, size);
}

void CommandProcessor::ResyncStaleGuestBytes(const char* what, uint64_t vaddr, uint64_t size) {
	auto& cache = m_renderer.GetBufferCache();
	GetScheduler().Current().EndRendering();
	// A fresh write-fault pass over everything written so far, waited to completion including the
	// priority operation that marks the pages; the caller's re-read then faults and downloads —
	// for pages inside a declared unbounded range only. A page the tracker already calls clean
	// (e.g. after a premature readback) is not repaired here; see the device-copy probe in
	// ProcessIndirectBuffer.
	cache.MarkWriteFaultsDirty();
	cache.FlushWriteFaults(true);
	cache.OnCommandBoundary();
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 50) {
		LOGF("CommandProcessor: %s at 0x%016" PRIx64 "/%" PRIu64
		     " bytes looked stale: resynced GPU writes\n",
		     what, vaddr, size);
	}
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	// The guest's CPU writes made before this submission must be collected before it executes
	// (kernel write-watch; RenderContext::BeginCommandStream).
	m_renderer.NoteGuestSubmit();
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = true;
	GpuThreadMark::on = true; // host_gpu probes (budget scopes) identify the GPU thread by this
	g_gpu_state  = gpu;

	// Every queue blocked (WaitRegMem/CE/rewind) with nothing else to do: report after 5 s.
	std::chrono::steady_clock::time_point blocked_since {};
	std::chrono::steady_clock::time_point blocked_reported {};
	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				// Wake once a second so an incomplete submission is reported even when nobody
				// waits on it (the game may be waiting for its end-of-pipe interrupt instead).
				gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 1000000);
				gpu->m_renderer.GetCommandScheduler().GetMasterSemaphore().CheckStall(
				    "gpu thread idle");
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					const auto now = std::chrono::steady_clock::now();
					if (blocked_since == std::chrono::steady_clock::time_point {}) {
						blocked_since = now;
					} else if (now - blocked_since > std::chrono::seconds(5) &&
					           now - blocked_reported > std::chrono::seconds(30)) {
						blocked_reported = now;
						gpu->ReportBlockedQueues(
						    std::chrono::duration<double>(now - blocked_since).count());
					}
					gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					gpu->m_renderer.GetCommandScheduler().GetMasterSemaphore().CheckStall(
					    "gpu thread blocked queues");
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
				has_submission    = true;
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			command();

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		const bool complete = gpu->Process(submission);
		// Progress of this slice only: a finished constant-engine stream keeps its old flag on
		// later slices (it is no longer processed), which reset the timer forever (review).
		if (complete || gpu->m_last_slice_progress) {
			blocked_since = {}; // the blocked-queue timer counts only retries without progress
		}

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

bool GuestGpu::Process(Submission& submission) {
	const bool first_slice = !submission.started;
	auto&      cp          = GetProcessor(submission.queue_id);

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
	}

	cp.BufferInit();
	bool complete         = true;
	m_last_slice_progress = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			m_last_slice_progress = progressed;
			if (progressed) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::Compute: {
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			m_last_slice_progress = submission.command_execution.MadeProgress();
			if (submission.command_execution.MadeProgress()) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			// A CPU flip presents guest memory without a command stream: CPU writes to the surface
			// must reach the texture cache first (kernel write-watch).
			m_renderer.CollectSubmittedCpuWrites();
			m_renderer.NoteFrameBoundary();
			m_renderer.RunGarbageCollector();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint: {
			m_renderer.NoteFrameBoundary();
			// AgcSuspendPoint is called once per frame by UE5's RHI submission thread and blocks
			// until the previous point is answered. By default the answer waits for the GPU to
			// COMPLETE everything before the point (priority operation), which serializes CPU
			// recording of frame N+1 behind GPU execution of frame N (hot-2 thread snapshots).
			// KYTY_SUSPEND_POINT_REACHED=1 (probe, A/B) answers as soon as the GPU thread has
			// REACHED the point, i.e. recorded and submitted everything before it.
			static const bool reached = [] {
				const char* value = std::getenv("KYTY_SUSPEND_POINT_REACHED");
				return value != nullptr && value[0] == '1' && value[1] == '\0';
			}();
			cp.EmitGlobalBarrier();
			m_renderer.GetBufferCache().FlushWriteFaults(false);
			if (!reached) {
				m_renderer.GetCommandScheduler().DeferPriorityOperation(
				    [ready = m_suspend_point_ready] { ready->release(); });
			}
			cp.BufferFlush();
			cp.Reset();
			if (reached) {
				m_suspend_point_ready->release();
			}
			break;
		}
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;
	m_renderer.BeginCommandStream();

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	ProcessPm4(execution);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

// Type-3 packet with a known opcode, or a type-2 NOP filler. GPU-built command buffers use
// nothing else; anything else at the start of a buffer means the host read stale bytes.
static bool LooksLikePm4Header(uint32_t header) {
	switch (header >> 30u) {
		case 3u: return g_cp_op_func[(header >> 8u) & 0xffu] != nullptr;
		case 2u: return true;
		default: return false;
	}
}

// Whole-buffer check: every packet is a known type-3 packet or a type-2 filler, the lengths
// chain exactly to the end, and at least two real packets exist. A single header can be matched
// by a negative float (0xc0100000 = -2.25 reads as a NOP), a whole stream cannot (review).
static bool LooksLikePm4Stream(std::span<const uint32_t> words) {
	size_t   i       = 0;
	uint32_t packets = 0;
	while (i < words.size()) {
		const auto header = words[i];
		if ((header >> 30u) == 2u) {
			i++;
			continue;
		}
		if (!LooksLikePm4Header(header)) {
			return false;
		}
		const auto length = static_cast<size_t>(KYTY_PM4_LEN(header));
		if (length == 0 || i + length > words.size()) {
			return false;
		}
		i += length;
		packets++;
	}
	return packets >= 2;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	// GPU-generated command buffers (Zero Company builds some through the unbounded heap) are
	// only known to be GPU-dirty once the write-fault records are processed; prepare the range
	// so the host read below faults into a download instead of parsing stale CPU bytes
	// ("unknown op" with float data at a predicated branch target, run ng-2).
	PrepareHostRead(reinterpret_cast<uint64_t>(commands.data()), commands.size_bytes());
	// The prepare above only flushes writes inside declared unbounded ranges. A buffer whose
	// first dword is not a PM4 header is almost certainly stale CPU memory (ng-2 and ng-21 died
	// on float data there): land every pending GPU write, let the tracker protect the pages and
	// read again before parsing.
	if (!commands.empty() && !LooksLikePm4Header(commands[0])) {
		const auto first = commands[0];
		ResyncStaleGuestBytes("indirect buffer", reinterpret_cast<uint64_t>(commands.data()),
		                      commands.size_bytes());
		static std::atomic<uint32_t> reports {0};
		const bool still_invalid = !LooksLikePm4Header(commands[0]);
		if (reports.fetch_add(1, std::memory_order_relaxed) < 50 || still_invalid) {
			LOGF("CommandProcessor: indirect buffer 0x%016" PRIx64 "/%zu started with 0x%08" PRIx32
			     " (not a PM4 header): resynced GPU writes, now 0x%08" PRIx32 " (%s)\n",
			     reinterpret_cast<uint64_t>(commands.data()), commands.size(), first,
			     *reinterpret_cast<const volatile uint32_t*>(commands.data()),
			     still_invalid ? "still invalid" : "valid");
		}
		if (still_invalid) {
			// ng-35 (intro, t=498): float transform data at the buffer start survived the resync.
			// Record what the caches know about the page and where the packet came from, so the
			// next occurrence tells a stale parent packet from an untracked GPU writer.
			auto&      cache   = m_renderer.GetBufferCache();
			const auto address = reinterpret_cast<uint64_t>(commands.data());
			uint32_t   backing = 0;
			(void)LibKernel::Memory::TryReadBacking(address, &backing, sizeof(backing));
			LOGF("CommandProcessor: stale indirect buffer 0x%016" PRIx64 ": %s unbounded-range=%s "
			     "last-gpu-write-tick=%" PRIu64 " now-tick=%" PRIu64 " pending-unbounded=%s "
			     "backing=0x%08" PRIx32 "\n",
			     address, cache.DescribeBufferAt(address).c_str(),
			     cache.InUnboundedWriteRange(address, commands.size_bytes()) ? "yes" : "no",
			     cache.LastGpuWriteTick(address), GetScheduler().CurrentTick(),
			     cache.HasPendingUnboundedWrites() ? "yes" : "no", backing);
			if (!g_current_execution->m_buffer_stack.empty()) {
				const auto& parent = g_current_execution->m_buffer_stack.back();
				const auto  base   = reinterpret_cast<uint64_t>(parent.commands.data());
				const auto  at     = base + uint64_t {parent.offset_dw} * sizeof(uint32_t);
				LOGF("CommandProcessor: stale indirect buffer referenced from 0x%016" PRIx64
				     " (parent 0x%016" PRIx64 "/%zu +0x%x, %s, unbounded-range=%s)\n",
				     at, base, parent.commands.size(), parent.offset_dw,
				     cache.DescribeBufferAt(at).c_str(),
				     cache.InUnboundedWriteRange(base, parent.commands.size_bytes()) ? "yes" : "no");
				const auto* words = parent.commands.data() + parent.offset_dw;
				const auto  avail = parent.commands.size() - parent.offset_dw;
				for (size_t i = 0; i < std::min<size_t>(avail, 16); i++) {
					LOGF("\t parent[%zu] = 0x%08" PRIx32 "\n", i, words[i]);
				}
			}
			// Stale-buffer workflow (2026-10-13) mechanism #1: a BAR readback copied the owner's
			// device memory before its join copy had executed and un-dirtied the page; the write-
			// fault resync cannot repair a page the tracker already calls clean. Decide from the
			// device copy itself: wait for the GPU, then look at what the owner buffer holds.
			const auto ready = cache.OwnerReadyTick(address);
			LOGF("CommandProcessor: stale indirect buffer: write-stamp=%s owner-ready-tick=%" PRIu64
			     " (%s)\n",
			     cache.HasGpuWriteStamp(address) ? "yes" : "no", ready,
			     GetScheduler().IsFree(ready) ? "free" : "IN FLIGHT");
			GetScheduler().FlushAndWait();
			std::array<uint32_t, 8> device {};
			if (cache.PeekDeviceCopy(address, device.data(), sizeof(device))) {
				LOGF("CommandProcessor: stale indirect buffer: device copy = %08x %08x %08x %08x %08x %08x %08x %08x\n",
				     device[0], device[1], device[2], device[3], device[4], device[5], device[6], device[7]);
				if (LooksLikePm4Header(device[0]) && device[0] != commands[0]) {
					// Restore the host bytes from the device copy only when the WHOLE device copy
					// parses as a PM4 stream, the host did not write the page after the lost
					// download (not CPU-dirty) and the GPU is idle (FlushAndWait above). Loud on
					// purpose: a mitigation for a lost download, not the fix for its cause.
					std::vector<uint32_t> restored(commands.size());
					const bool have_copy =
					    cache.PeekDeviceCopy(address, restored.data(), commands.size_bytes());
					// Kernel write-watch: CPU writes to the page reach the tracker only when collected.
					(void)m_renderer.CollectCpuWrites(address, commands.size_bytes());
					const bool cpu_dirty = cache.IsRegionCpuModified(address, commands.size_bytes());
					const bool stream    = have_copy && LooksLikePm4Stream(restored);
					// Experimental (KYTY_STALE_IB_RECOVER=1): a valid-looking stream in VRAM proves
					// neither allocation identity nor freshness (an old stream stays valid after the
					// GPU moved on), so by default this only reports; the run then fails as before.
					static const bool recover = [] {
						const char* value = std::getenv("KYTY_STALE_IB_RECOVER");
						return value != nullptr && value[0] == '1' && value[1] == '\0';
					}();
					if (!recover) {
						LOGF("CommandProcessor: stale indirect buffer: device copy %s (pm4-stream=%d "
						     "cpu-dirty=%d); recovery is opt-in (KYTY_STALE_IB_RECOVER=1)\n",
						     have_copy ? "captured" : "unavailable", stream ? 1 : 0, cpu_dirty ? 1 : 0);
					} else if (have_copy && stream && !cpu_dirty) {
						LibKernel::Memory::WriteBacking(address, restored.data(), commands.size_bytes());
						LOGF("CommandProcessor: stale indirect buffer 0x%016" PRIx64 "/%zu RECOVERED from "
						     "the device copy (complete PM4 stream, page not CPU-dirty; lost download: BAR "
						     "readback before the owner was ready?)\n",
						     address, commands.size());
					} else {
						LOGF("CommandProcessor: stale indirect buffer: device copy NOT used (copy=%d "
						     "pm4-stream=%d cpu-dirty=%d)\n",
						     have_copy ? 1 : 0, stream ? 1 : 0, cpu_dirty ? 1 : 0);
					}
				}
			} else {
				LOGF("CommandProcessor: stale indirect buffer: no mapped owner to peek\n");
			}
		}
	}
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

void CommandProcessor::SuspendPm4(const char* kind, uint64_t addr, uint64_t ref, uint64_t mask,
                                  uint32_t func, uint32_t size) {
	m_last_suspend = {kind, addr, ref, mask, func, size};
	SuspendPm4();
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	while (!execution.m_buffer_stack.empty()) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (GraphicsRunDebugDumpEnabled() && skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (GraphicsRunDebugDumpEnabled() && log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		if (GpuIndirectEnabled()) {
			NoteMergeOpcode(opcode);
		}
		if (SubmitAheadConfig::Get().Enabled()) {
			SubmitAheadBeforePacket(opcode);
		}
		const auto packet_dw =
		    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		// Safe point (no cache locks held): record pending write-fault passes, publish pages.
		m_renderer.GetBufferCache().OnCommandBoundary();
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances       = num_instances;
	m_num_instances_stale = false;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			PrepareHostRead(reinterpret_cast<uint64_t>(address), 16u * 2u * sizeof(uint64_t));
			HostReadScope host_read(HostReadProbe::Predication);
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4("predication_results", reinterpret_cast<uint64_t>(address), 0, 0,
						           0, 16u * 2u * sizeof(uint64_t));
					} else {
						m_predicate_skip = false;
						NoteGuestAcquire(reinterpret_cast<uint64_t>(address),
						                 16u * 2u * sizeof(uint64_t));
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			PrepareHostRead(reinterpret_cast<uint64_t>(address), sizeof(uint64_t));
			{
				HostReadScope host_read(HostReadProbe::Predication);
				value = *reinterpret_cast<const volatile uint64_t*>(address);
			}
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	// The command processor read the result: acquire point for later BDA / unbounded consumers.
	NoteGuestAcquire(reinterpret_cast<uint64_t>(address),
	                 op == 0x01 ? 16u * 2u * sizeof(uint64_t) : sizeof(uint64_t));
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}


enum class PacketClass { ShRegs, CtxRegs, Draw, Dispatch, Benign, Other };

// Classifies PM4 packets for the (b1) merge probe and KYTY_SUBMIT_AHEAD.
static PacketClass ClassifyPacket(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_SH_REG_INDIRECT: return PacketClass::ShRegs;
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
		case Pm4::IT_CLEAR_STATE: return PacketClass::CtxRegs;
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE: return PacketClass::Draw;
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT: return PacketClass::Dispatch;
		// No GPU work, no side effect: per-packet state the records carry, control flow, CE/DE sync.
		case Pm4::IT_NOP:
		case Pm4::IT_INDIRECT_BUFFER: // control flow only; its packets are classified one by one
		case Pm4::IT_SET_BASE:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_PFP_SYNC_ME:
		case Pm4::IT_INCREMENT_DE_COUNTER:
		case Pm4::IT_INCREMENT_CE_COUNTER:
		case Pm4::IT_WAIT_ON_CE_COUNTER:
		case Pm4::IT_WAIT_ON_DE_COUNTER_DIFF:
		case Pm4::IT_WRITE_CONST_RAM: return PacketClass::Benign;
		default: return PacketClass::Other;
	}
}

// (b1) diagnostics: classify every PM4 packet so consecutive GPU mesh packets can be compared.
void CommandProcessor::NoteMergeOpcode(uint32_t opcode) {
	auto& probe = m_merge_probe;
	switch (ClassifyPacket(opcode)) {
		case PacketClass::ShRegs: probe.sh++; break;
		case PacketClass::CtxRegs: probe.ctx++; break;
		case PacketClass::Draw:
		case PacketClass::Dispatch: probe.draw++; break;
		case PacketClass::Benign: break;
		case PacketClass::Other: probe.other++; break;
	}
}

// KYTY_SUBMIT_AHEAD (submitAhead.h): before a work or side-effect packet, submit the open command
// buffer once it has been recording longer than the threshold. Same operation as BufferFlush
// (submit, no wait).
void CommandProcessor::SubmitAheadBeforePacket(uint32_t opcode) {
	auto& scheduler = GetScheduler();
	if (!scheduler.Active() || CurrentBuffer().IsInvalid()) {
		return;
	}
	using Packet    = SubmitAheadTrigger::Packet;
	const auto kind = ClassifyPacket(opcode);
	const auto packet = kind == PacketClass::ShRegs || kind == PacketClass::CtxRegs ? Packet::Register
	                    : kind == PacketClass::Benign                               ? Packet::Benign
	                    : kind == PacketClass::Draw                                 ? Packet::Draw
	                    : kind == PacketClass::Dispatch                             ? Packet::Dispatch
	                                                                                : Packet::Other;
	const bool work   = packet != Packet::Register && packet != Packet::Benign;
	const auto reason = m_submit_ahead.Before(packet, scheduler.CurrentTick(),
	                                          work ? scheduler.OpenCommandAgeNs() : 0,
	                                          CurrentBuffer().IsRendering(), SubmitAheadConfig::Get());
	if (reason == SubmitAheadTrigger::Reason::None) {
		return;
	}
	const auto tick = scheduler.CurrentTick();
	BufferFlush();
	m_submit_ahead.Submitted(packet, scheduler.CurrentTick());
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
		LOGF("SubmitAhead: submit tick %" PRIu64 " before opcode 0x%02x (%s)\n", tick, opcode,
		     reason == SubmitAheadTrigger::Reason::DispatchTimer ? "dispatch-timer" : "timer");
	}
	SubmitAheadProbe::Count(reason == SubmitAheadTrigger::Reason::DispatchTimer
	                            ? SubmitAheadProbe::DispatchTimerSubmits
	                            : SubmitAheadProbe::TimerSubmits);
}

// Called for each GPU mesh packet after it was recorded (the packet itself is one probe.draw).
void CommandProcessor::CountMergeRun(uint64_t mesh_hash) {
	auto&       probe = m_merge_probe;
	const auto& vs    = m_sh_ctx.GetVs();
	const auto& ps    = m_sh_ctx.GetPs();
	uint64_t    user_data = 0xcbf29ce484222325ull; // FNV-1a over the GS and PS user SGPR values
	const auto  mix = [&](uint64_t value) {
		 user_data = (user_data ^ value) * 0x100000001b3ull;
	};
	for (uint32_t i = 0; i < vs.gs_user_sgpr.count && i < HW::UserSgprInfo::SGPRS_MAX; ++i) {
		mix(vs.gs_user_sgpr.value[i]);
	}
	mix(UINT64_MAX);
	for (uint32_t i = 0; i < ps.ps_user_sgpr.count && i < HW::UserSgprInfo::SGPRS_MAX; ++i) {
		mix(ps.ps_user_sgpr.value[i]);
	}
	const uint64_t ps_addr = ps.ps_regs.data_addr;
	if (probe.valid) {
		if (probe.other != probe.at_other || probe.draw != probe.at_draw + 1u) {
			HostReadProbe::CountGpuIndirect(HostReadProbe::MergeBreakOp);
		} else if (mesh_hash != probe.mesh_hash || ps_addr != probe.ps_addr) {
			HostReadProbe::CountGpuIndirect(HostReadProbe::MergeBreakShader);
		} else if (probe.ctx != probe.at_ctx) {
			HostReadProbe::CountGpuIndirect(HostReadProbe::MergeBreakCtx);
		} else if (user_data != probe.user_data) {
			HostReadProbe::CountGpuIndirect(HostReadProbe::MergeUserData);
		} else {
			HostReadProbe::CountGpuIndirect(HostReadProbe::MergeSame);
		}
	}
	probe.valid     = true;
	probe.mesh_hash = mesh_hash;
	probe.ps_addr   = ps_addr;
	probe.user_data = user_data;
	probe.at_ctx    = probe.ctx;
	probe.at_draw   = probe.draw;
	probe.at_other  = probe.other;
}

void CommandProcessor::PoisonGpuIndirectSlots(const ShaderDrawParameters& parameters,
                                              uint64_t last_record_addr, bool indexed) {
	const auto poison = [&](uint32_t slot, uint32_t field) {
		if (slot == ShaderDrawParameters::kNoSlot || slot >= m_sgpr_poison.size()) {
			return;
		}
		m_sgpr_poison[slot] = {.active      = true,
		                       .writes      = m_sh_ctx.GsUserSgprWrites(slot),
		                       .record_addr = last_record_addr,
		                       .field       = field};
		m_any_sgpr_poison   = true;
	};
	// DrawIndirectArgs: start vertex = dword 2, start instance = dword 3; DrawIndexedIndirectArgs:
	// base vertex = dword 3, start instance = dword 4.
	poison(parameters.base_vertex_slot, indexed ? 3u : 2u);
	poison(parameters.base_instance_slot, indexed ? 4u : 3u);
	m_num_instances_stale = true;
}

// Counts draws that read state a GPU packet left unpatched. `instances_from_state`: the draw took
// its instance count from m_num_instances. Slots the current GPU packet names are served by it.
void CommandProcessor::CountStaleDrawState(bool instances_from_state,
                                           const GpuIndirectDraw* gpu_indirect) {
	if (instances_from_state && m_num_instances_stale) {
		HostReadProbe::CountGpuIndirect(HostReadProbe::StaleInstances);
	}
	if (!m_any_sgpr_poison) {
		return;
	}
	const auto user_sgprs = static_cast<uint32_t>(m_sh_ctx.GetVs().gs_regs.rsrc2.user_sgpr);
	bool       any        = false;
	bool       stale      = false;
	uint32_t   stale_slot = 0;
	for (uint32_t slot = 0; slot < m_sgpr_poison.size(); ++slot) {
		auto& poison = m_sgpr_poison[slot];
		if (!poison.active) {
			continue;
		}
		if (poison.writes != m_sh_ctx.GsUserSgprWrites(slot)) {
			poison.active = false; // rewritten since the packet
			continue;
		}
		any = true;
		if (slot >= user_sgprs || (gpu_indirect != nullptr &&
		                           (slot == gpu_indirect->parameters.base_vertex_slot ||
		                            slot == gpu_indirect->parameters.base_instance_slot))) {
			continue;
		}
		if (!stale) {
			stale      = true;
			stale_slot = slot;
		}
	}
	m_any_sgpr_poison = any;
	if (!stale) {
		return;
	}
	HostReadProbe::CountGpuIndirect(HostReadProbe::StaleSgpr);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		const auto& poison = m_sgpr_poison[stale_slot];
		LOGF("GpuIndirect(stale-sgpr): draw reads GS slot %u (user_sgprs=%u) left unpatched by a GPU"
		     " packet; host path value = dword %u of record 0x%016" PRIx64 "\n",
		     stale_slot, user_sgprs, poison.field, poison.record_addr);
	}
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	args.index_type_and_size = m_index_type_and_size;
	const bool instances_from_state = args.instance_count == 0;
	if (args.instance_count == 0) {
		args.instance_count = m_num_instances;
	}
	if (GpuIndirectEnabled()) {
		CountStaleDrawState(instances_from_state, args.gpu_indirect);
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	DrawIndex({.index_count = index_count, .index_addr = index_addr});
}

static void PatchIndirectDrawOffsets(CommandProcessor& cp, IndirectDrawRegisters registers,
                                     uint32_t vertex_offset, uint32_t instance_offset,
                                     uint32_t first_index) {
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location == Pm4::SH_NOP) {
			return;
		}
		EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
		                     g_hw_sh_indirect_func[location] == nullptr);
		g_hw_sh_indirect_func[location](cp, location, value);
	};
	write(registers.vertex_offset, vertex_offset);
	write(registers.instance_offset, instance_offset);
	write(registers.index_offset, first_index);
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
                                    uint32_t draw_initiator, bool indexed) {
	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	DrawIndirectMulti(data_offset, 1, nullptr, args_size, registers, draw_initiator, indexed);
}

// Logs the first 32 distinct mesh-packet tuples with the decision (KYTY_CP_GPU_INDIRECT only).
// Registers are printed as GS user-data slots (-1: not named, r<n>: not a GS user-data register).
static void LogMeshTuple(const RenderExecutor::GpuIndirectShaderInfo& shader, bool indexed,
                         uint32_t stride, uint32_t max_count, bool has_count_addr,
                         IndirectDrawRegisters registers, uint32_t decision_id, const char* decision) {
	using Tuple = std::array<uint64_t, 9>;
	static std::mutex         mutex;
	static std::vector<Tuple> seen;
	const Tuple tuple {shader.hash, shader.fast_launch, shader.primitive, shader.ge_primitive_group_size,
	                   (indexed ? 1u : 0u) | (has_count_addr ? 2u : 0u), stride,
	                   registers.vertex_offset, registers.instance_offset, decision_id};
	std::lock_guard lock(mutex);
	if (seen.size() >= 32 || std::find(seen.begin(), seen.end(), tuple) != seen.end()) {
		return;
	}
	seen.push_back(tuple);
	const auto slot = [](uint32_t location) {
		if (location == Pm4::SH_NOP) {
			return std::string("-1");
		}
		if (location >= Pm4::SPI_SHADER_USER_DATA_GS_0 && location <= Pm4::SPI_SHADER_USER_DATA_GS_31) {
			return std::to_string(location - Pm4::SPI_SHADER_USER_DATA_GS_0);
		}
		return "r" + std::to_string(location);
	};
	LOGF("GpuIndirect(mesh-tuple): ms=0x%016" PRIx64 " fl=%u prim=%u ge_ppg=%u indexed=%d stride=%u"
	     " max_count=%u count_addr=%d vertex_offset_slot=%s instance_offset_slot=%s -> %s\n",
	     shader.hash, shader.fast_launch, shader.primitive, shader.ge_primitive_group_size,
	     indexed ? 1 : 0, stride, max_count, has_count_addr ? 1 : 0,
	     slot(registers.vertex_offset).c_str(), slot(registers.instance_offset).c_str(), decision);
}

// Returns true when the draw was recorded with GPU-resident arguments. False keeps the host path
// (and has counted why).
bool CommandProcessor::TryDrawIndirectOnGpu(uint32_t data_offset, uint32_t max_count_or_count,
                                            const volatile uint32_t* count_addr,
                                            uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                            bool indexed) {
	// The first-index SGPR has no draw-parameter built-in.
	if (registers.index_offset != Pm4::SH_NOP) {
		HostReadProbe::CountGpuIndirect(HostReadProbe::FallbackIndexOffset);
		return false;
	}
	const auto fallback = [](HostReadProbe::GpuIndirect kind = HostReadProbe::FallbackOther) {
		HostReadProbe::CountGpuIndirect(kind);
		return false;
	};
	ShaderDrawParameters parameters;
	parameters.active = true;
	const auto slot_of = [](uint32_t location, uint32_t& slot) {
		if (location == Pm4::SH_NOP) {
			slot = ShaderDrawParameters::kNoSlot;
			return true;
		}
		if (location < Pm4::SPI_SHADER_USER_DATA_GS_0 || location > Pm4::SPI_SHADER_USER_DATA_GS_31) {
			return false; // not a vertex-stage user-data register
		}
		slot = location - Pm4::SPI_SHADER_USER_DATA_GS_0;
		return true;
	};
	if (!slot_of(registers.vertex_offset, parameters.base_vertex_slot) ||
	    !slot_of(registers.instance_offset, parameters.base_instance_slot)) {
		return fallback();
	}

	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	if (max_count_or_count == 0 || stride_in_bytes < args_size || (stride_in_bytes & 3u) != 0 ||
	    ((m_draw_indirect_args_base_addr + data_offset) & 3u) != 0 ||
	    (count_addr != nullptr && (reinterpret_cast<uint64_t>(count_addr) & 3u) != 0)) {
		return fallback();
	}
	if (indexed && (m_index_base_addr == 0 || m_index_buffer_size == 0)) {
		return fallback();
	}

	auto& executor = m_renderer.GetRenderExecutor();
	using Block = RenderExecutor::GpuIndirectBlock;
	Block                                   block = Block::Other;
	RenderExecutor::GpuIndirectShaderInfo shader;
	const bool eligible = executor.CanDrawIndirectOnGpu(CurrentBuffer(), indexed, m_index_type_and_size,
	                                                    max_count_or_count, parameters, &block, &shader,
	                                                    count_addr != nullptr);
	if (shader.known && shader.mesh) {
		const char* decision = "gpu";
		switch (block) {
			case Block::None: break;
			case Block::Mesh: decision = "fallback(mesh)"; break;
			case Block::MeshDenied: decision = "fallback(mesh-denied)"; break;
			case Block::MeshCountAddr: decision = "fallback(mesh-count-addr)"; break;
			case Block::MeshPrim: decision = "fallback(mesh-prim)"; break;
			case Block::Denied: decision = "fallback(denied)"; break;
			case Block::Srt: decision = "fallback(srt)"; break;
			case Block::Other: decision = "fallback(other)"; break;
		}
		LogMeshTuple(shader, indexed, stride_in_bytes, max_count_or_count, count_addr != nullptr,
		             registers, static_cast<uint32_t>(block), decision);
	}
	if (!eligible) {
		switch (block) {
			case Block::Srt:
				// The vertex shader's resource plan reads an offset SGPR (or is not compiled yet).
				HostReadProbe::CountGpuIndirect(HostReadProbe::FallbackSrt);
				return false;
			case Block::Mesh: HostReadProbe::CountGpuIndirect(HostReadProbe::FallbackMesh); break;
			case Block::MeshDenied:
			case Block::MeshCountAddr:
			case Block::MeshPrim:
				HostReadProbe::CountGpuIndirect(HostReadProbe::FallbackMesh);
				HostReadProbe::CountGpuIndirect(block == Block::MeshDenied      ? HostReadProbe::FallbackMeshDenied
				                                : block == Block::MeshCountAddr ? HostReadProbe::FallbackMeshCountAddr
				                                                                : HostReadProbe::FallbackMeshPrim);
				break;
			case Block::Denied: HostReadProbe::CountGpuIndirect(HostReadProbe::FallbackDenied); break;
			default: break;
		}
		return fallback();
	}

	GpuIndirectDraw draw;
	draw.args_addr         = m_draw_indirect_args_base_addr + data_offset;
	draw.count_addr        = reinterpret_cast<uint64_t>(count_addr);
	draw.draw_count        = max_count_or_count;
	draw.stride            = stride_in_bytes;
	draw.index_base        = m_index_base_addr;
	draw.index_buffer_size = m_index_buffer_size;
	draw.parameters        = parameters;

	// The renderer needs a non-empty placeholder: the real counts live in the argument records.
	if (indexed) {
		DrawIndex({.index_count    = m_index_buffer_size,
		           .index_addr     = reinterpret_cast<const void*>(m_index_base_addr),
		           .instance_count = 1,
		           .offset_source  = DrawOffsetSource::IndirectArgs,
		           .gpu_indirect   = &draw});
	} else {
		DrawIndexAuto({.vertex_count   = 1,
		               .instance_count = 1,
		               .offset_source  = DrawOffsetSource::IndirectArgs,
		               .gpu_indirect   = &draw});
	}
	HostReadProbe::CountGpuIndirect(HostReadProbe::GpuDraws);
	if (shader.mesh) {
		HostReadProbe::CountGpuIndirect(HostReadProbe::MeshGpuDraws);
		CountMergeRun(shader.hash);
	}
	PoisonGpuIndirectSlots(parameters,
	                       draw.args_addr + static_cast<uint64_t>(max_count_or_count - 1u) * stride_in_bytes,
	                       indexed);
	return true;
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	if (GpuIndirectEnabled() &&
	    TryDrawIndirectOnGpu(data_offset, max_count_or_count, count_addr, stride_in_bytes,
	                         registers, indexed)) {
		return;
	}

	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		PrepareHostRead(reinterpret_cast<uint64_t>(count_addr), sizeof(uint32_t));
		{
			HostReadScope host_read(HostReadProbe::DrawCount);
			draw_count = *count_addr;
		}
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}

	if (draw_count == 0) {
		return;
	}

	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);

	uint64_t index_size = 0;
	if (indexed) {
		switch (m_index_type_and_size) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
		}
	}

	// GPU-driven culling builds these records: publish pending unbounded writes before the host
	// reads them (audit 2026-10-12: only count_addr was prepared; a stale instance count here
	// becomes a runaway draw).
	PrepareHostRead(m_draw_indirect_args_base_addr + data_offset,
	                static_cast<uint64_t>(draw_count - 1) * stride_in_bytes + args_size);
	{
		HostReadScope host_read(HostReadProbe::DrawArgs);
		HostReadProbe::Touch(m_draw_indirect_args_base_addr + data_offset,
		                     static_cast<uint64_t>(draw_count - 1) * stride_in_bytes + args_size);
	}

	for (uint32_t i = 0; i < draw_count; i++) {
		const auto args_addr = m_draw_indirect_args_base_addr + data_offset +
		                       static_cast<uint64_t>(i) * stride_in_bytes;

		if (!indexed) {
			auto* args = reinterpret_cast<const DrawIndirectArgs*>(args_addr);
			PatchIndirectDrawOffsets(*this, registers, args->start_vertex_location,
			                         args->start_instance_location, 0);
			if (args->vertex_count_per_instance == 0 || args->instance_count == 0) {
				continue;
			}
			m_num_instances       = args->instance_count;
			m_num_instances_stale = false;
			DrawIndexAuto({.vertex_count   = args->vertex_count_per_instance,
			               .instance_count = args->instance_count,
			               .first_vertex   = args->start_vertex_location,
			               .first_instance = args->start_instance_location,
			               .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		auto* args = reinterpret_cast<const DrawIndexedIndirectArgs*>(args_addr);
		PatchIndirectDrawOffsets(*this, registers, args->base_vertex_location,
		                         args->start_instance_location, args->start_index_location);

		const uint32_t index_count =
		    (m_index_buffer_size != 0
		         ? std::min(args->index_count_per_instance, m_index_buffer_size)
		         : args->index_count_per_instance);
		if (index_count == 0 || args->instance_count == 0) {
			continue;
		}
		if (GraphicsRunDebugDumpEnabled() && index_count != args->index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args->index_count_per_instance, index_count);
			}
		}

		auto* index_addr = reinterpret_cast<const void*>(
		    m_index_base_addr + static_cast<uint64_t>(args->start_index_location) * index_size);

		m_num_instances       = args->instance_count;
		m_num_instances_stale = false;
		DrawIndex({.index_count    = index_count,
		           .index_addr     = index_addr,
		           .instance_count = args->instance_count,
		           .base_vertex    = static_cast<int32_t>(args->base_vertex_location),
		           .first_instance = args->start_instance_location,
		           .offset_source  = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));

	uint32_t frame_num = 0;

	{
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
				const auto& cs = m_sh_ctx.GetCs().cs_regs;
				const auto& oa = m_ucfg.GetGdsOaCounter(m_ucfg.GetGdsOaState().GetIndex());
				LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
				     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
				     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
				     "\n",
				     frame_num, m_submit_id, thread_group_x, thread_group_y, thread_group_z,
				     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
				     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
				     cs.data_addr, m_ucfg.GetGdsOaState().GetIndex(),
				     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
				     oa.GetSpaceAvailable());
			}
		}

		m_renderer.GetRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
		                                              thread_group_y, thread_group_z, mode, IsAsyncComputeQueue());
	}
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		// Host-side read of GPU-built dispatch arguments (audit 2026-10-12).
		PrepareHostRead(args_addr, sizeof(vk::DispatchIndirectCommand));
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(args_addr);
		uint32_t    x = 0, y = 0, z = 0;
		{
			HostReadScope host_read(HostReadProbe::DispatchArgs);
			x = args->x;
			y = args->y;
			z = args->z;
		}
		DispatchDirect(x, y, z, mode);
		return;
	}
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	m_renderer.GetRenderExecutor().DispatchIndirect(m_submit_id, CurrentBuffer(), args_addr, mode,
	                                                IsAsyncComputeQueue());
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	const bool instances_from_state = args.instance_count == 0;
	if (args.instance_count == 0) {
		args.instance_count = m_num_instances;
	}
	if (GpuIndirectEnabled()) {
		CountStaleDrawState(instances_from_state, args.gpu_indirect);
	}
	m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	BufferFlush();

	m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
	                                      static_cast<int>(display_buffer_index));
}

// End-of-pipe labels used to be stored at record time. With lazily tracked unbounded-heap GPU
// writes in flight, a poller could then see "done" and read result pages before the write-fault
// pass marked them, i.e. stale CPU bytes (audit 2026-10-12; UE5 loaders wait on such results).
// While such writes are pending the store is deferred behind the write-fault pass (priority
// order) and performed on the GPU thread, so a protected page takes the tracker's merge path.
// KYTY_EOP_LABEL_DEFER=1 enables the deferred store (off by default until validated): v1 (58c4a83)
// hopped through the GPU thread's command queue and lost most of the frame rate; v2 (498d034)
// called BufferCache::InvalidateMemory from the priority thread (readback path) and crashed; v3
// writes through the backing mapping and only marks the tracker; v3 first used the buffer cache's
// IsRegionGpuModified, whose PrepareHostRead touches GPU-thread-only maps (ng-33 crash), so the
// query is now tracker-only (IsLabelRegionGpuModified).
// The interrupt step for a deferred label store (see WriteEopLabel); empty without an interrupt.
std::function<void()> CommandProcessor::EopInterrupt(bool with_interrupt, uint32_t context_id) {
	if (!with_interrupt) {
		return {};
	}
	return [this, event_id = m_interrupt_event_id, context_id] {
		m_renderer.TriggerInterrupt(event_id, context_id);
	};
}

template <typename T>
bool CommandProcessor::WriteEopLabel(T* dst, T value, std::function<void()> after_store) {
	static const bool defer = [] {
		const char* env = std::getenv("KYTY_EOP_LABEL_DEFER");
		return env != nullptr && env[0] == '1';
	}();
	auto& cache = m_renderer.GetBufferCache();
	if (!defer || !cache.HasPendingUnboundedWrites()) {
		m_renderer.StoreGuestMemory(reinterpret_cast<uint64_t>(dst), &value, sizeof(value));
		return false; // the caller schedules the interrupt as before
	}
	cache.FlushWriteFaults(false);
	auto* gpu = g_gpu_state;
	EXIT_IF(gpu == nullptr);
	// The interrupt (after_store) is delivered by the same step that performs the store, so it
	// can no longer overtake a store handed to the GPU thread (review 2026-10-12).
	GetScheduler().DeferPriorityOperation([gpu, &cache, dst, value, after_store = std::move(after_store)] {
		// Priority thread, after the tick completed and the write-fault marks ran. A label page
		// the GPU does not own is written directly (and marked CPU-modified for later uploads);
		// a GPU-dirty page goes through the GPU thread so the store takes the merge path. Either
		// way wake the GPU thread: a WAIT_REG_MEM on this label retries only every 100 ms.
		const auto address = reinterpret_cast<uint64_t>(dst);
		if (!cache.IsLabelRegionGpuModified(address, sizeof(T)) &&
		    LibKernel::Memory::TryWriteBacking(address, &value, sizeof(T))) {
			cache.MarkLabelWritten(address, sizeof(T)); // tracker only: no readback off the GPU thread
			if (after_store) {
				after_store();
			}
			gpu->Wake();
			return;
		}
		gpu->SendCommand([dst, value, after_store] {
			std::memcpy(dst, &value, sizeof(value));
			if (after_store) {
				after_store();
			}
		});
	});
	return true;
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				Sync::TriggerEopEventAtEndOfPipe(command, m_interrupt_event_id,
				                                 interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	auto write32 = [&](bool with_writeback) {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		const bool deferred = WriteEopLabel(dst, data, EopInterrupt(with_interrupt, interrupt_context_id));

		if (with_interrupt && !deferred) {
			if (with_writeback) {
				Sync::WriteAtEndOfPipeWithInterruptWriteBack32(m_submit_id, command, dst,
				                                               data, m_interrupt_event_id,
				                                               interrupt_context_id);
			} else {
				Sync::WriteAtEndOfPipeWithInterrupt32(m_submit_id, command, dst, data,
				                                      m_interrupt_event_id, interrupt_context_id);
			}
		} else if (with_writeback) {
			Sync::WriteAtEndOfPipeWithWriteBack32(m_submit_id, command, dst, data);
		} else {
			Sync::WriteAtEndOfPipe32(m_submit_id, command, dst, data);
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					SynchronizeGpu();
					Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst, value & 0xffffu,
					              value >> 16u);
					Sync::WriteAtEndOfPipeGds32(m_submit_id, command, dst, value & 0xffffu,
					                            value >> 16u);
					if (with_interrupt) {
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32(false);
				return;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00: write32(false); return;
						case 0x38: write32(true); return;
						default: break;
					}
				}
			} else {
				if (event_write_source == 0x04) {
					value = Sync::ReadReferenceClock();
				}
				auto write64 = [&](bool with_writeback) {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					const bool deferred =
					    WriteEopLabel(dst, value, EopInterrupt(with_interrupt, interrupt_context_id));

					if (with_interrupt && !deferred) {
						if (with_writeback) {
							Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
							    m_submit_id, command, dst, value, m_interrupt_event_id,
							    interrupt_context_id);
						} else {
							Sync::WriteAtEndOfPipeWithInterrupt64(m_submit_id, command, dst,
							                                      value, m_interrupt_event_id,
							                                      interrupt_context_id);
						}
					} else if (with_writeback) {
						Sync::WriteAtEndOfPipeWithWriteBack64(m_submit_id, command, dst,
						                                      value);
					} else {
						Sync::WriteAtEndOfPipe64(m_submit_id, command, dst, value);
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64(false);
									return;
								}
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) {
									write64(false);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64(false);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05) ||
								    (event_index == 0x00)) {
									write64(true);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64(true);
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}

	EXIT("unknown event type\n");
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::EmitGlobalBarrier() {
	Common::LockGuard lock(m_renderer.GetMutex());

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	GetScheduler().EndRendering();
	auto& buffer = CurrentBuffer();
	buffer.RawHandle().pipelineBarrier2(dependency);
	// An all-commands -> all-commands read|write barrier satisfies every pending tracker hazard.
	buffer.Hazards().Clear();
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id, interrupt_context_id);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: EmitGlobalBarrier(); break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			const uint64_t     results      = event_address;
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				m_renderer.StoreGuestMemory(results + db * 2u * sizeof(uint64_t), &value, sizeof(value));
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip() {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	auto& command = CurrentBuffer();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeOnlyFlip(m_submit_id, command, m_flip.handle, m_flip.index,
	                               m_flip.flip_mode, m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	m_renderer.StoreGuestMemory(reinterpret_cast<uint64_t>(dst_gpu_addr), &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithFlip32(m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr),
	                                 value, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                 m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	m_renderer.StoreGuestMemory(reinterpret_cast<uint64_t>(dst_gpu_addr), &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithInterruptWriteBackFlip32(
	    m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr), value, m_flip.handle,
	    m_flip.index, m_flip.flip_mode, m_flip.flip_arg, request, m_interrupt_event_id);
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	m_renderer.GetBufferCache().FlushWriteFaults(false);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	GetScheduler().Flush();
}

void CommandProcessor::SynchronizeGpu() {
	GetScheduler().Finish();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

} // namespace Libs::Graphics
