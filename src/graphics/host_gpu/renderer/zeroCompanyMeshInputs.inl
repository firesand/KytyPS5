// Included inside Libs::Graphics by renderDraw.cpp. This diagnostic copies only
// actual prepared Vulkan views; it never reads or publishes guest backing.
namespace ZeroCompanyMeshInputs {

constexpr uint64_t MaxBytes = 768ull * 1024 * 1024;
constexpr uint32_t MaxCaptures = 2;

struct Chunk {
	vk::DescriptorBufferInfo view {};
	uint64_t destination = 0;
};

static void Log(const std::string& message) {
	::Log::WriteToConsoleAndLog(message.c_str());
	std::fflush(stdout);
}

static std::string Words(std::span<const uint32_t> words) {
	std::string result = "[";
	for (const auto word: words) {
		if (result.size() != 1) result += ',';
		result += fmt::format("{}", word);
	}
	return result + ']';
}

// Every string placed in the manifest is escaped, including environment values.
static std::string JsonString(std::string_view value) {
	std::string result = "\"";
	for (const unsigned char c: value) {
		if (c == '\\' || c == '"') {
			result += '\\';
			result += static_cast<char>(c);
		} else if (c < 32 || c >= 127) {
			result += fmt::format("\\u{:04x}", c);
		} else {
			result += static_cast<char>(c);
		}
	}
	return result + '"';
}

static bool WriteFile(const std::filesystem::path& path, const void* bytes, uint64_t size) {
	// Exclusive creation protects previous evidence. Only completed GPU data is
	// passed here; a failed disk write never becomes a completed .bin artifact.
	const auto partial = path.string() + ".partial";
	auto* file = std::fopen(partial.c_str(), "wbx");
	if (file == nullptr) return false;
	const auto* data = static_cast<const uint8_t*>(bytes);
	bool ok = true;
	for (uint64_t offset = 0; offset < size;) {
		const auto count = static_cast<size_t>(std::min<uint64_t>(1024 * 1024, size - offset));
		if (std::fwrite(data + offset, 1, count, file) != count) {
			ok = false;
			break;
		}
		offset += count;
	}
	if (std::fflush(file) != 0) ok = false;
	if (std::fclose(file) != 0) ok = false;
	std::error_code error;
	if (ok) {
		if (std::filesystem::exists(path, error) || error) ok = false;
		if (ok) {
			std::filesystem::rename(partial, path, error);
			ok = !error;
		}
	}
	if (!ok) {
		error.clear();
		std::filesystem::remove(partial, error);
	}
	return ok;
}

static bool BuildPlan(std::span<PreparedBindings* const> stages, std::vector<Chunk>& chunks,
                      uint64_t& total, std::string& metadata) {
	using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;
	metadata = "[";
	for (uint32_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
		const auto* prepared = stages[stage_index];
		if (!prepared || !prepared->runtime || !*prepared->runtime) return false;
		const auto& program = *prepared->runtime->program;
		const auto& snapshot = *prepared->runtime->resources;
		// These exact variants use push constants. Do not silently omit a new
		// variant's additional descriptor buffers from a purported complete capture.
		if (prepared->flattened_srt.buffer || prepared->shader_data_buffer.buffer ||
		    prepared->shared_memory.buffer || prepared->gds.buffer ||
		    program.info.buffers.size() > 64 || program.info.images.size() > 64 ||
		    prepared->shader_data.size() > 4096 || snapshot.user_data.size() > 256 ||
		    program.info.buffers.size() != snapshot.buffers.size() ||
		    program.info.images.size() != snapshot.images.size() ||
		    prepared->images.size() != program.info.images.size()) return false;
		if (stage_index != 0) metadata += ',';
		metadata += fmt::format(
		    "{{\"stage_index\":{},\"stage\":{},\"hash\":\"0x{:016x}\","
		    "\"push_start\":{},\"memory_offset_dword\":{},\"shader_data\":{},"
		    "\"user_data_base\":{},\"user_data\":{},\"buffers\":[",
		    stage_index, static_cast<uint32_t>(program.stage), program.shader_hash,
		    program.bindings.push_data_start_dword, program.bindings.memory_offset_dword,
		    Words(prepared->shader_data), program.user_data_base, Words(snapshot.user_data));
		for (uint32_t resource_index = 0; resource_index < program.info.buffers.size(); ++resource_index) {
			const auto& resource = program.info.buffers[resource_index];
			const auto& raw = snapshot.buffers[resource_index];
			const auto index = resource.descriptor_index;
			const uint64_t word = uint64_t {program.bindings.memory_offset_dword} + index / 4u;
			if (index == UINT32_MAX || index >= prepared->buffers.size() ||
			    index >= prepared->buffer_sources.size() || word >= prepared->shader_data.size() ||
			    raw.dword_count < 4 || raw.dword_count > raw.dwords.size()) return false;
			const auto& view = prepared->buffers[index];
			const auto& source = prepared->buffer_sources[index];
			const auto packed = prepared->shader_data[word];
			const auto prefix = (packed >> ((index % 4u) * 8u)) & 255u;
			if (!view.buffer || view.range == 0 || view.range == VK_WHOLE_SIZE ||
			    view.offset > UINT64_MAX - view.range || view.offset % 4u || view.range % 4u ||
			    prefix % 4u || prefix > view.range || source.size > view.range - prefix ||
			    source.address > UINT64_MAX - source.size) return false;
			const auto found = std::ranges::find_if(chunks, [&](const Chunk& chunk) {
				return chunk.view.buffer == view.buffer && chunk.view.offset == view.offset &&
				       chunk.view.range == view.range;
			});
			uint32_t chunk_index;
			if (found == chunks.end()) {
				if (view.range > MaxBytes - total) return false;
				chunk_index = static_cast<uint32_t>(chunks.size());
				chunks.push_back({view, total});
				total += view.range;
			} else {
				chunk_index = static_cast<uint32_t>(found - chunks.begin());
			}
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(raw);
			if (resource_index != 0) metadata += ',';
			metadata += fmt::format(
			    "{{\"resource\":{},\"source\":{},\"first_use_pc\":{},\"descriptor\":{},"
			    "\"binding\":{},\"chunk\":{},\"prefix\":{},\"packed_prefix_word\":{},"
			    "\"guest_address\":\"0x{:x}\",\"source_size\":{},\"raw_size\":{},"
			    "\"scalar\":{},\"read\":{},\"write\":{},\"image_alias\":{},\"raw\":{}}}",
			    resource_index, resource.source, resource.first_use_pc, index,
			    ShaderRecompiler::IR::NativeBinding(program.stage, BindingKind::Buffers),
			    chunk_index, prefix, packed, source.address, source.size, descriptor.GetSize(),
			    resource.scalar, resource.read, resource.written, resource.image_alias,
			    Words(std::span {raw.dwords}.first(raw.dword_count)));
		}
		metadata += "],\"images_uncaptured\":[";
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& raw = snapshot.images[i];
			if (raw.dword_count > raw.dwords.size()) return false;
			const auto& resource = program.info.images[i];
			const auto& image = prepared->images[i];
			const auto kind = ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			if (i != 0) metadata += ',';
			metadata += fmt::format(
			    "{{\"resource\":{},\"binding\":{},\"descriptor\":{},\"raw\":{},"
			    "\"view\":\"0x{:x}\",\"slot\":[{},{}],\"view_format\":{},"
			    "\"guest_address\":\"0x{:x}\",\"size\":{},\"extent\":[{},{},{}],"
			    "\"atomic64\":{},\"contents_captured\":false}}",
			    i, kind ? ShaderRecompiler::IR::NativeBinding(program.stage, *kind) : UINT32_MAX,
			    resource.descriptor_index, Words(std::span {raw.dwords}.first(raw.dword_count)),
			    GpuDiagnostics::HandleValue(image.image_view), image.image_id.index,
			    image.image_id.generation, JsonString(vk::to_string(image.desc.view_info.format)),
			    image.desc.info.data.address, image.desc.info.data.size, image.desc.info.extent.width,
			    image.desc.info.extent.height, image.desc.info.extent.depth, resource.atomic64);
		}
		metadata += "]}";
	}
	metadata += ']';
	return !chunks.empty() && total != 0;
}

static void Capture(CommandBuffer& buffer, uint64_t submit, uint64_t pipeline,
                    std::span<PreparedBindings* const> stages,
                    std::span<const uint32_t> draw_data, const ShaderMeshInputInfo& mesh,
                    uint32_t groups, uint32_t instances, bool indexed, bool restart) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PROBE_MESH_INPUTS");
		return value && std::strcmp(value, "1") == 0;
	}();
	if (!enabled) return;
	if (!GpuDiagnostics::Enabled() || !GpuDiagnostics::Serialize() ||
	    !GpuDiagnostics::SerializePreparation()) {
		static std::atomic_flag warned = ATOMIC_FLAG_INIT;
		if (!warned.test_and_set(std::memory_order_relaxed))
			Log("MeshInputs: disabled reason=requires_diagnostics_sync_and_sync_prep\n");
		return;
	}
	bool has_mesh = false, has_pixel = false;
	for (const auto* stage: stages) {
		if (!stage || !stage->runtime || !*stage->runtime) continue;
		const auto& program = *stage->runtime->program;
		has_mesh |= program.stage == ShaderType::Mesh && program.shader_hash == 0x67e610ccc37800edull;
		has_pixel |= program.stage == ShaderType::Pixel && program.shader_hash == 0x548ab529f34a1307ull;
	}
	if (!has_mesh || !has_pixel) return;
	static std::atomic<uint32_t> attempts {0};
	const auto ordinal = attempts.fetch_add(1, std::memory_order_relaxed);
	if (ordinal >= MaxCaptures) return;
	auto fail = [&](const char* reason) {
		Log(fmt::format("MeshInputs: skipped capture={} submit={} reason={}\n", ordinal, submit, reason));
	};
	// The only BDA reads in this exact MS are indexed-input loads. Their payload
	// is outside this buffer-only capture; do not advertise indexed draws as complete.
	if (stages.size() != 2 || indexed || restart || draw_data.size() != 6 ||
	    draw_data[3] != 0 || draw_data[4] != 0 || draw_data[5] != 0) {
		fail("unsupported_draw_shape");
		return;
	}
	std::vector<Chunk> chunks;
	uint64_t total = 0;
	std::string stage_metadata;
	if (!BuildPlan(stages, chunks, total, stage_metadata)) {
		fail("unsafe_or_oversize_bindings");
		return;
	}
	const auto* diagnostics_path = std::getenv("KYTY_GPU_DIAGNOSTICS_PATH");
	if (!diagnostics_path || !*diagnostics_path) {
		fail("missing_diagnostics_path");
		return;
	}
	std::error_code error;
	auto parent = std::filesystem::path(diagnostics_path).parent_path();
	if (parent.empty()) parent = std::filesystem::current_path(error);
	if (error) { fail("diagnostics_parent"); return; }
	const auto space = std::filesystem::space(parent, error);
	if (error || space.available < total + 1024 * 1024) {
		fail("insufficient_evidence_space");
		return;
	}
	// All attempts use one fresh evidence directory. Existing evidence is never
	// reused, including after a process restart with the same diagnostic path.
	static std::mutex directory_mutex;
	static std::optional<std::filesystem::path> output_root;
	static bool root_failed = false;
	std::filesystem::path directory;
	{
		const std::lock_guard lock(directory_mutex);
		if (root_failed) { fail("evidence_directory_unavailable"); return; }
		if (!output_root) {
			const auto root = parent / "mesh-inputs";
			if (!std::filesystem::create_directory(root, error) || error) {
				root_failed = true;
				fail("evidence_directory_exists_or_unwritable");
				return;
			}
			output_root = root;
		}
		directory = *output_root / fmt::format("draw-{}", ordinal);
		if (!std::filesystem::create_directory(directory, error) || error) {
			fail("draw_directory_exists_or_unwritable");
			return;
		}
	}
	std::string metadata = fmt::format(
	    "\"schema\":1,\"capture\":{},\"guest_submit\":{},\"pipeline\":\"0x{:x}\","
	    "\"ordering_changed\":true,\"buffer_only\":true,\"ps_images_captured\":false,"
	    "\"sync\":true,\"sync_prep\":true,\"bytes\":{},\"limit_bytes\":{},"
	    "\"draw_data\":{},\"mesh_groups\":[{},{},1],\"indexed\":false,"
	    "\"mesh\":{{\"threads\":{},\"wave_size\":{},\"host_subgroup_size\":{},"
	    "\"lds_dwords\":{},\"input_primitive\":{},\"fast_launch\":{},"
	    "\"primitives_per_group\":{},\"vertices_per_group\":{},\"max_vertices\":{},"
	    "\"max_primitives\":{},\"provoking_vertex\":{}}},\"stages\":{},\"chunks\":[",
	    ordinal, submit, pipeline, total, MaxBytes, Words(draw_data), groups, instances,
	    Words(mesh.threads_num), mesh.wave_size, mesh.host_subgroup_size, mesh.lds_size_dwords,
	    mesh.input_primitive, mesh.fast_launch, mesh.primitives_per_group, mesh.vertices_per_group,
	    mesh.max_vertices, mesh.max_primitives, mesh.provoking_vertex, stage_metadata);
	for (uint32_t i = 0; i < chunks.size(); ++i) {
		const auto& chunk = chunks[i];
		if (i != 0) metadata += ',';
		metadata += fmt::format(
		    "{{\"chunk\":{},\"file\":\"buffer-{:02}.bin\",\"handle\":\"0x{:x}\","
		    "\"offset\":{},\"range\":{},\"staging_offset\":{}}}", i, i,
		    GpuDiagnostics::HandleValue(chunk.view.buffer), chunk.view.offset, chunk.view.range,
		    chunk.destination);
	}
	metadata += "],\"probe_environment\":{";
	bool first = true;
	for (const char* name: {"KYTY_PROBE_MESH_INPUTS", "KYTY_PROBE_LDS_WAVE_ORDER",
	                       "KYTY_PROBE_LDS_WAVE_ORDER_ALL", "KYTY_PROBE_SKIP_PS_548A"}) {
		if (!first) metadata += ',';
		first = false;
		const auto* value = std::getenv(name);
		metadata += JsonString(name) + ':' + JsonString(value ? value : "");
	}
	metadata += '}';
	const auto pending = "{\"status\":\"pending_gpu_copy\"," + metadata + "}\n";
	if (!WriteFile(directory / "pending.json", pending.data(), pending.size())) {
		fail("pending_manifest_write_failed");
		return;
	}
	auto& scheduler = buffer.GetContext().GetCommandScheduler();
	const auto preparation_tick = scheduler.CurrentTick();
	Log(fmt::format("MeshInputs: preparation_submit capture={} submit={} tick={} chunks={} "
	                "bytes={} ordering_changed=true buffer_only=true ps_images_captured=false path={}\n",
	                ordinal, submit, preparation_tick, chunks.size(), total, directory.string()));
	buffer.EndRendering();
	scheduler.FlushAndWait();
	const auto preparation_completed = scheduler.GetMasterSemaphore().KnownGpuTick();
	Log(fmt::format("MeshInputs: preparation_complete capture={} submitted={} completed={} next={}\n",
	                ordinal, preparation_tick, preparation_completed, scheduler.CurrentTick()));
	Buffer download(buffer.GetGraphics(), scheduler, MemoryUsage::Download, 0,
	                vk::BufferUsageFlagBits::eTransferDst, total);
	if (download.Mapped().size() < total) { fail("download_unmapped"); return; }
	const auto copy_tick = scheduler.CurrentTick();
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
		barrier.offset = chunk.view.offset;
		barrier.size = chunk.view.range;
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
	for (uint32_t i = 0; i < chunks.size(); ++i) {
		const auto& chunk = chunks[i];
		const vk::BufferCopy copy {chunk.view.offset, chunk.destination, chunk.view.range};
		command.copyBuffer(chunk.view.buffer, download.Handle(), 1, &copy);
		GpuDiagnosticEvent event;
		event.kind = "mesh-input-copy";
		event.guest_submit = submit;
		event.pipeline = pipeline;
		event.args = {ordinal, i, GpuDiagnostics::HandleValue(chunk.view.buffer),
		              chunk.view.offset, chunk.view.range, chunk.destination};
		scheduler.RecordGpuOperation(event);
	}
	dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(after.size());
	dependency.pBufferMemoryBarriers = after.data();
	command.pipelineBarrier2(dependency);
	Log(fmt::format("MeshInputs: copy_submit capture={} submit={} tick={} bytes={}\n",
	                ordinal, submit, copy_tick, total));
	// Do not Finish/PopPendingOperations, rebind, or allocate from shared streams.
	// Serialize() also fences the resumed draw before a later draw can recycle a
	// stream reservation whose old preparation tick has now completed.
	scheduler.FlushAndWait();
	download.Invalidate(0, total);
	const auto copy_completed = scheduler.GetMasterSemaphore().KnownGpuTick();
	Log(fmt::format("MeshInputs: copy_complete capture={} submitted={} completed={} next={}\n",
	                ordinal, copy_tick, copy_completed, scheduler.CurrentTick()));
	for (uint32_t i = 0; i < chunks.size(); ++i) {
		const auto& chunk = chunks[i];
		if (!WriteFile(directory / fmt::format("buffer-{:02}.bin", i),
		               download.Mapped().data() + chunk.destination, chunk.view.range)) {
			fail("binary_write_failed_pending_manifest_is_incomplete");
			return;
		}
		Log(fmt::format("MeshInputs: file_complete capture={} chunk={} bytes={}\n",
		                ordinal, i, chunk.view.range));
	}
	const auto manifest = fmt::format(
	    "{{\"status\":\"complete\",{},\"preparation_tick\":{},\"preparation_completed\":{},"
	    "\"copy_tick\":{},\"copy_completed\":{},\"resume_tick\":{}}}\n",
	    metadata, preparation_tick, preparation_completed, copy_tick, copy_completed,
	    scheduler.CurrentTick());
	if (!WriteFile(directory / "manifest.json", manifest.data(), manifest.size())) {
		fail("complete_manifest_write_failed");
		return;
	}
	Log(fmt::format("MeshInputs: capture_complete capture={} submit={} resume_tick={} bytes={} "
	                "ordering_changed=true buffer_only=true ps_images_captured=false path={}\n",
	                ordinal, submit, scheduler.CurrentTick(), total, directory.string()));
}

} // namespace ZeroCompanyMeshInputs
