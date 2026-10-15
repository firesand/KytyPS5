#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/fault_buffer_process_4k_spv.h"
#include "gpu_tiler_shaders/fault_buffer_process_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <bit>
#include <cinttypes>
#include <cstring>
#include <limits>

namespace Libs::Graphics {

namespace {

constexpr size_t MaxPageFaults    = 65536; // matches -DMAX_PAGE_FAULTS for fault_buffer_process.comp
constexpr size_t PageFaultAreaSize = MaxPageFaults * sizeof(uint64_t);

} // namespace

FaultManager::FaultManager(GraphicContext& graphics, CommandScheduler& scheduler,
                           BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      m_fault_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                     BufferCache::FAULT_BUFFER_BYTES), // BDA bitmap + unbounded-write bitmap
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
                        2 * MaxPendingFaults * PageFaultAreaSize) {
	SetVulkanObjectNameF(m_graphics.device, m_fault_buffer.Handle(), "Fault Buffer");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                &m_fault_process_desc_layout),
	    "create fault-buffer descriptor layout");

	const auto module = CompileSPV(FAULT_BUFFER_PROCESS_SPV, m_graphics.device);

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &m_fault_process_desc_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                           &m_fault_process_pipeline_layout),
	    "create fault-buffer pipeline layout");

	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_fault_process_pipeline_layout;
	const auto result = m_graphics.device.createComputePipelines(
	    nullptr, 1, &pipeline_info, nullptr, &m_fault_process_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create fault-buffer pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_fault_process_pipeline, "Fault Buffer Parser");

	const auto module_4k = CompileSPV(FAULT_BUFFER_PROCESS_4K_SPV, m_graphics.device);
	stage.module         = module_4k;
	pipeline_info.stage  = stage;
	const auto result_4k = m_graphics.device.createComputePipelines(
	    nullptr, 1, &pipeline_info, nullptr, &m_write_process_pipeline);
	m_graphics.device.destroyShaderModule(module_4k, nullptr);
	RequireVulkanSuccess(result_4k, "create write-fault pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_write_process_pipeline, "Write Fault Parser");
}

FaultManager::~FaultManager() {
	m_graphics.device.destroyPipeline(m_fault_process_pipeline, nullptr);
	m_graphics.device.destroyPipeline(m_write_process_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fault_process_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_fault_process_desc_layout, nullptr);
}

void FaultManager::ProcessFaultBuffer() {
	Process(false);
}

void FaultManager::ProcessWriteFaultBuffer() {
	Process(true);
}

void FaultManager::Process(bool write_faults) {
	auto&      areas        = write_faults ? m_write_areas : m_fault_areas;
	auto&      current_area = write_faults ? m_current_write_area : m_current_area;
	if (const auto wait_tick = areas[current_area].load(std::memory_order_acquire); wait_tick != 0) {
		m_scheduler.Wait(wait_tick);
		if (write_faults) {
			m_scheduler.WaitPriorityOperations(wait_tick);
		} else {
			m_scheduler.PopPendingOperations();
		}
	}

	const auto offset = (current_area + (write_faults ? MaxPendingFaults : 0u)) * PageFaultAreaSize;
	auto*      mapped = m_download_buffer.Mapped().data() + offset;
	// The compress pass only reads the entry counter; entries are written before being read back.
	std::memset(mapped, 0, 64);
	m_download_buffer.Flush(offset, 64);
	constexpr vk::DeviceSize bda_bytes   = BufferCache::CACHING_NUMPAGES / 8;
	constexpr vk::DeviceSize write_bytes = BufferCache::WRITE_FAULT_NUMPAGES / 8;

	vk::BufferMemoryBarrier2 pre_barrier {};
	pre_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	pre_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	pre_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	pre_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	pre_barrier.buffer        = m_fault_buffer.Handle();
	pre_barrier.offset        = 0;
	pre_barrier.size           = m_fault_buffer.Size();
	auto post_barrier         = pre_barrier;
	post_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	post_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	post_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	post_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;

	const vk::DescriptorBufferInfo infos[] {
	    {m_fault_buffer.Handle(), write_faults ? bda_bytes : 0, write_faults ? write_bytes : bda_bytes},
	    {m_download_buffer.Handle(), offset, PageFaultAreaSize},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &pre_barrier;
	{ GpuTimestampScope ts_scope(command, GpuTimestamps::Kind::BarrierBuffer); command.pipelineBarrier2(dependency); }
	command.bindPipeline(vk::PipelineBindPoint::eCompute,
	                     write_faults ? m_write_process_pipeline : m_fault_process_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,
	                             m_fault_process_pipeline_layout, 0, writes);
	const auto num_threads    = (write_faults ? BufferCache::WRITE_FAULT_NUMPAGES
	                                          : BufferCache::CACHING_NUMPAGES) / 32;
	const auto num_workgroups = (num_threads + 63) / 64;
	{ GpuTimestampScope ts_scope(command, GpuTimestamps::Kind::CacheDispatch); command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1); }
	if (GpuDiagnostics::Enabled()) {
		GpuDiagnosticEvent event;
		event.kind = "utility-fault-process";
		event.pipeline = GpuDiagnostics::HandleValue(m_fault_process_pipeline);
		event.args = {num_workgroups, 1, 1, offset,
		              GpuDiagnostics::HandleValue(m_fault_buffer.Handle()),
		              GpuDiagnostics::HandleValue(m_download_buffer.Handle())};
		m_scheduler.RecordGpuOperation(event);
	}
	dependency.pBufferMemoryBarriers = &post_barrier;
	{ GpuTimestampScope ts_scope(command, GpuTimestamps::Kind::BarrierBuffer); command.pipelineBarrier2(dependency); }

	const auto area = current_area;
	if (write_faults) {
		// Priority operation: it must run before the end-of-pipe signals queued after it, so the
		// game's fence wait already finds the written pages marked and protected.
		m_scheduler.DeferPriorityOperation([this, mapped, offset, area] {
			m_download_buffer.Invalidate(offset, PageFaultAreaSize);
			const auto* faults = std::bit_cast<const uint64_t*>(mapped);
			const auto  count  = std::min<uint32_t>(static_cast<uint32_t>(faults[0]),
			                                        static_cast<uint32_t>(MaxPageFaults - 1));
			for (uint32_t index = 1; index <= count; ++index) {
				m_buffer_cache.MarkPageWritten(BufferCache::GuestAddress(faults[index]));
			}
			if (static_cast<uint32_t>(faults[0]) >= MaxPageFaults - 1) {
				m_buffer_cache.MarkWriteFaultOverflow();
			}
			m_write_areas[area].store(0, std::memory_order_release);
		});
	} else {
		m_scheduler.DeferOperation([this, mapped, offset, area] {
			m_download_buffer.Invalidate(offset, PageFaultAreaSize);
			RangeSet    fault_ranges;
			const auto* faults = std::bit_cast<const uint64_t*>(mapped);
			const auto  count  = static_cast<uint32_t>(faults[0]);
			for (uint32_t index = 1; index <= count; ++index) {
				const auto address = BufferCache::GuestAddress(faults[index]);
				fault_ranges.Add(address, BufferCache::CACHING_PAGESIZE);
				if (::Log::VerboseEnabled()) LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", address);
			}
			fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
				EXIT_IF(end - start > std::numeric_limits<uint32_t>::max());
				(void)m_buffer_cache.FindBuffer(start, end - start);
			});
			m_fault_areas[area].store(0, std::memory_order_release);
		});
	}

	areas[current_area++].store(m_scheduler.CurrentTick(), std::memory_order_release);
	current_area %= MaxPendingFaults;
}

} // namespace Libs::Graphics
