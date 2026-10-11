// Included inside Libs::Graphics by renderCompute.cpp. This opt-in diagnostic reads
// only completed GPU copies of the exact descriptors about to be dispatched.
namespace ZeroCompanyInputs {

struct Chunk {
	const char* name = "unknown";
	uint32_t resource = 0;
	uint64_t requested = 0;
	uint64_t destination = 0;
	uint64_t logical_offset = 0;
	uint32_t descriptor_index = 0;
	uint32_t packed_word = 0;
	uint32_t prefix = 0;
	uint64_t guest_address = 0;
	uint64_t guest_size = 0;
	vk::DescriptorBufferInfo view {};
	uint64_t native_offset = 0;
	uint64_t size = 0;
	std::array<uint8_t, 240> bytes {};
};

static void Log(const std::string& message) {
	::Log::WriteToConsoleAndLog(message.c_str());
	std::fflush(stdout);
}

static bool Resolve(Chunk& chunk, const PreparedBindings& bindings) {
	if (bindings.runtime == nullptr || !bindings.runtime->program) return false;
	const auto& program = *bindings.runtime->program;
	if (chunk.resource >= program.info.buffers.size()) return false;
	const auto index = program.info.buffers[chunk.resource].descriptor_index;
	if (index == UINT32_MAX || index >= bindings.buffers.size() ||
	    index >= bindings.buffer_sources.size()) return false;
	const uint64_t word_index = uint64_t {program.bindings.memory_offset_dword} + index / 4u;
	if (word_index >= bindings.shader_data.size()) return false;
	chunk.descriptor_index = index;
	chunk.packed_word = bindings.shader_data[word_index];
	chunk.prefix = (chunk.packed_word >> ((index % 4u) * 8u)) & 0xffu;
	chunk.view = bindings.buffers[index];
	chunk.guest_address = bindings.buffer_sources[index].address;
	chunk.guest_size = bindings.buffer_sources[index].size;
	// The shader's dword address arithmetic truncates an unaligned prefix. Do not
	// label such an ambiguous raw-byte capture as the logical guest descriptor.
	if (!chunk.view.buffer || chunk.view.range == VK_WHOLE_SIZE || chunk.prefix % 4u != 0 ||
	    chunk.logical_offset % 4u != 0 || chunk.destination % 4u != 0 ||
	    chunk.prefix > chunk.view.range || chunk.logical_offset > chunk.guest_size ||
	    chunk.logical_offset > chunk.view.range - chunk.prefix) return false;
	const auto prefix_and_offset = uint64_t {chunk.prefix} + chunk.logical_offset;
	if (chunk.view.offset > UINT64_MAX - prefix_and_offset) return false;
	chunk.native_offset = chunk.view.offset + prefix_and_offset;
	chunk.size = std::min({chunk.requested, uint64_t {chunk.bytes.size()},
	                       chunk.guest_size - chunk.logical_offset,
	                       uint64_t {chunk.view.range} - prefix_and_offset}) & ~uint64_t {3};
	return chunk.size != 0 && chunk.native_offset % 4u == 0 &&
	       chunk.native_offset <= UINT64_MAX - chunk.size;
}

static uint32_t Word(const Chunk& chunk, size_t index) {
	const auto* bytes = chunk.bytes.data() + index * 4u;
	return uint32_t {bytes[0]} | (uint32_t {bytes[1]} << 8u) |
	       (uint32_t {bytes[2]} << 16u) | (uint32_t {bytes[3]} << 24u);
}

static bool Download(CommandBuffer& buffer, std::span<Chunk> chunks, uint64_t total,
                     const GpuDiagnosticEvent& dispatch_event, const char* phase) {
	auto& scheduler = buffer.GetContext().GetCommandScheduler();
	const auto tick = scheduler.CurrentTick();
	for (const auto& chunk: chunks) {
		if (chunk.size == 0 || chunk.destination > total ||
		    chunk.size > total - chunk.destination) return false;
		Log(fmt::format(
		    "ZeroCompanyInputs: before_copy phase={} submit={} tick={} name={} "
		    "resource={} descriptor={} guest=0x{:x} source_size={} native=0x{:x} "
		    "view_offset={} view_range={} packed_word=0x{:08x} prefix={} "
		    "logical_offset={} native_offset={} requested={} bytes={} destination={}\n",
		    phase, dispatch_event.guest_submit, tick, chunk.name, chunk.resource,
		    chunk.descriptor_index, chunk.guest_address, chunk.guest_size,
		    GpuDiagnostics::HandleValue(chunk.view.buffer), chunk.view.offset, chunk.view.range,
		    chunk.packed_word, chunk.prefix, chunk.logical_offset, chunk.native_offset,
		    chunk.requested, chunk.size, chunk.destination));
	}
	// A dedicated allocation cannot wrap a shared download/upload stream or cause
	// BufferCache to clear GPU-dirty ranges or publish stale data to guest backing.
	Buffer download(buffer.GetGraphics(), scheduler, MemoryUsage::Download, 0,
	                vk::BufferUsageFlagBits::eTransferDst, total);
	if (download.Mapped().size() < total) {
		Log(fmt::format("ZeroCompanyInputs: skipped phase={} reason=download_unmapped\n", phase));
		return false;
	}
	buffer.EndRendering();
	const auto command = buffer.Handle();
	std::vector<vk::BufferMemoryBarrier2> before, after;
	for (const auto& chunk: chunks) {
		vk::BufferMemoryBarrier2 barrier {};
		barrier.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
		barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		barrier.dstStageMask = vk::PipelineStageFlagBits2::eCopy;
		barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer = chunk.view.buffer;
		barrier.offset = chunk.native_offset;
		barrier.size = chunk.size;
		before.push_back(barrier);
		barrier.srcStageMask = vk::PipelineStageFlagBits2::eCopy;
		barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
		barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
		barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		after.push_back(barrier);
	}
	vk::BufferMemoryBarrier2 host {};
	host.srcStageMask = vk::PipelineStageFlagBits2::eCopy;
	host.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	host.dstStageMask = vk::PipelineStageFlagBits2::eHost;
	host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
	host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host.buffer = download.Handle();
	host.offset = 0;
	host.size = total;
	after.push_back(host);
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(before.size());
	dependency.pBufferMemoryBarriers = before.data();
	command.pipelineBarrier2(dependency);
	for (const auto& chunk: chunks) {
		const vk::BufferCopy copy {chunk.native_offset, chunk.destination, chunk.size};
		command.copyBuffer(chunk.view.buffer, download.Handle(), 1, &copy);
		auto event = dispatch_event;
		event.kind = "persistent-input-copy";
		event.args = {chunk.resource, GpuDiagnostics::HandleValue(chunk.view.buffer),
		              chunk.native_offset, chunk.destination, chunk.size, chunk.logical_offset};
		scheduler.RecordGpuOperation(event);
	}
	dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(after.size());
	dependency.pBufferMemoryBarriers = after.data();
	command.pipelineBarrier2(dependency);
	Log(fmt::format("ZeroCompanyInputs: copy_submit phase={} submit={} tick={} "
	                "ordering_changed=true bytes={}\n", phase, dispatch_event.guest_submit, tick, total));
	// FlushAndWait keeps the CommandBuffer wrapper and prepared descriptors alive;
	// it does not drain deferred cache destruction. No stream uploads are allocated.
	scheduler.FlushAndWait();
	download.Invalidate(0, total);
	Log(fmt::format("ZeroCompanyInputs: copy_complete phase={} submit={} tick={} completed={} "
	                "next_tick={}\n", phase, dispatch_event.guest_submit, tick,
	                scheduler.GetMasterSemaphore().KnownGpuTick(), scheduler.CurrentTick()));
	for (auto& chunk: chunks) {
		std::memcpy(chunk.bytes.data(), download.Mapped().data() + chunk.destination, chunk.size);
		std::string words;
		for (size_t i = 0; i < chunk.size / 4u; ++i) {
			if (i != 0) words += ',';
			words += fmt::format("{:08x}", Word(chunk, i));
		}
		Log(fmt::format("ZeroCompanyInputs: data phase={} submit={} tick={} name={} "
		                "logical_offset={} bytes={} words_le32={}\n",
		                phase, dispatch_event.guest_submit, tick, chunk.name,
		                chunk.logical_offset, chunk.size, words));
	}
	return true;
}

static void Capture(CommandBuffer& buffer, const PreparedBindings& bindings,
                    const GpuDiagnosticEvent& dispatch_event) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PROBE_PERSISTENT_INPUTS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (!enabled || !GpuDiagnostics::SerializePreparation() || bindings.runtime == nullptr ||
	    !bindings.runtime->program ||
	    bindings.runtime->program->shader_hash != 0x2ef44268a059254full) return;
	static std::atomic_flag captured = ATOMIC_FLAG_INIT;
	if (captured.test_and_set(std::memory_order_relaxed)) return;
	// Called only after the preparation fence and before CommitBindings/dispatch.
	std::array<Chunk, 3> chunks {{{"constants", 0, 240, 0},
	                             {"header", 2, 52, 256},
	                             {"queue_counts", 3, 64, 320}}};
	for (auto& chunk: chunks) {
		if (!Resolve(chunk, bindings)) {
			Log(fmt::format("ZeroCompanyInputs: skipped submit={} name={} reason=unsafe_binding\n",
			                dispatch_event.guest_submit, chunk.name));
			return;
		}
	}
	if (!Download(buffer, chunks, 384, dispatch_event, "initial")) return;
	const auto& constants = chunks[0];
	const auto& header = chunks[1];
	if (constants.size < 228 || header.size < 16) {
		Log("ZeroCompanyInputs: records_skipped reason=short_constants_or_header\n");
		return;
	}
	const uint64_t bin_count = Word(constants, 4) >> 6u; // Native s59.
	const uint64_t claimed_index = Word(header, 3);     // Old counter at header byte 12.
	const uint32_t work_bound = Word(constants, 56);    // Native s54.
	// Both operands originate in u32 values, so the widened computation cannot wrap.
	const uint64_t record_offset = bin_count * 4u + claimed_index * 8u;
	const bool native_u32_wrap = record_offset > UINT32_MAX;
	Chunk records {"work_records", 3, 64, 0};
	records.logical_offset = record_offset;
	const bool valid = !native_u32_wrap && Resolve(records, bindings);
	Log(fmt::format("ZeroCompanyInputs: records_derived submit={} s59={} s54={} "
	                "header_counter12={} logical_offset={} valid={} bytes={} "
	                "counter_below_s54={} native_u32_wrap={}\n", dispatch_event.guest_submit, bin_count, work_bound,
	                claimed_index, record_offset, valid, valid ? records.size : 0,
	                claimed_index < work_bound, native_u32_wrap));
	if (valid) Download(buffer, std::span {&records, 1u}, 64, dispatch_event, "records");
}

} // namespace ZeroCompanyInputs
