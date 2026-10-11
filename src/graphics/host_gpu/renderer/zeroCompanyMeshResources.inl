// Included inside Libs::Graphics by renderDraw.cpp. CPU metadata only: never
// resolve cache entries, read guest memory, submit, wait, or download resources.
namespace ZeroCompanyMeshResources {

static std::string Words(std::span<const uint32_t> words) {
	std::string text;
	for (const auto word: words) {
		if (!text.empty()) text += ',';
		text += fmt::format("{:08x}", word);
	}
	return text;
}

static void Capture(CommandBuffer& buffer, uint64_t submit, uint64_t pipeline,
                    std::span<PreparedBindings* const> stages,
                    std::span<const uint32_t> draw_data) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PROBE_MESH_RESOURCES");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (!enabled || !GpuDiagnostics::Enabled() ||
	    !std::ranges::any_of(stages, [](const auto* stage) {
		    return stage && stage->runtime && stage->runtime->program &&
		           stage->runtime->program->stage == ShaderType::Mesh &&
		           stage->runtime->program->shader_hash == 0x67e610ccc37800edull;
	    })) return;
	static std::atomic<uint32_t> captures {0};
	const auto capture = captures.fetch_add(1, std::memory_order_relaxed);
	if (capture >= 64u) return;
	auto& scheduler = buffer.GetContext().GetCommandScheduler();
	auto& cache = buffer.GetContext().GetTextureCache();
	std::string log = fmt::format(
	    "MeshResources: begin capture={} submit={} tick={} pipeline=0x{:x} "
	    "draw_data={} metadata_only=true\n", capture, submit, scheduler.CurrentTick(),
	    pipeline, Words(draw_data));
	struct ImageAlias {
		uint32_t stage;
		uint32_t resource;
		ImageId id;
		GuestRange range;
	};
	std::vector<ImageAlias> aliases;
	for (uint32_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
		const auto& prepared = *stages[stage_index];
		const auto& program = *prepared.runtime->program;
		const auto& snapshot = *prepared.runtime->resources;
		log += fmt::format(
		    "MeshResources: stage capture={} stage_index={} stage={} hash=0x{:016x} "
		    "push_start={} memory_offset_dword={} shader_data={} user_data_base={} user_data={}\n",
		    capture, stage_index, static_cast<uint32_t>(program.stage), program.shader_hash,
		    program.bindings.push_data_start_dword, program.bindings.memory_offset_dword,
		    Words(prepared.shader_data), program.user_data_base, Words(snapshot.user_data));
		for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
			const auto& resource = program.info.buffers[i];
			const auto& raw = snapshot.buffers.at(i);
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(raw);
			log += fmt::format(
			    "MeshResources: buffer capture={} stage_index={} resource={} source={} pc=0x{:x} "
			    "descriptor={} scalar={} read={} write={} image_alias={} raw={} guest=0x{:x} "
			    "size={} stride={} records={} format={}\n",
			    capture, stage_index, i, resource.source, resource.first_use_pc,
			    resource.descriptor_index, resource.scalar, resource.read, resource.written,
			    resource.image_alias, Words(std::span {raw.dwords}.first(raw.dword_count)),
			    descriptor.Base48(), descriptor.GetSize(), descriptor.Stride(),
			    descriptor.NumRecords(), descriptor.RawFormat());
			const auto index = resource.descriptor_index;
			if (index == UINT32_MAX) continue;
			const auto& view = prepared.buffers.at(index);
			const auto& source = prepared.buffer_sources.at(index);
			const auto packed = prepared.shader_data.at(program.bindings.memory_offset_dword + index / 4u);
			log += fmt::format(
			    "MeshResources: buffer_binding capture={} stage_index={} resource={} binding={} "
			    "element={} handle=0x{:x} offset={} range={} prefix={} guest=0x{:x} size={}\n",
			    capture, stage_index, i, ShaderRecompiler::IR::NativeBinding(
			        program.stage, ShaderRecompiler::IR::DescriptorBindingKind::Buffers), index,
			    GpuDiagnostics::HandleValue(view.buffer), view.offset, view.range,
			    (packed >> ((index % 4u) * 8u)) & 0xffu, source.address, source.size);
		}
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& raw = snapshot.images.at(i);
			const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(raw);
			const auto& binding = prepared.images.at(i);
			const auto* image = cache.m_slot_images.try_get(binding.image_id);
			const auto kind = ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			log += fmt::format(
			    "MeshResources: image capture={} stage_index={} resource={} source={} pc=0x{:x} "
			    "binding={} element={} read={} write={} atomic64={} raw={} guest=0x{:x} "
			    "extent={}x{}x{} format={} base_level={} last_level={} max_mip={} "
			    "base_array={} tile={} requested_view_format={} slot={}:{} live={}\n",
			    capture, stage_index, i, resource.source, resource.first_use_pc,
			    kind ? ShaderRecompiler::IR::NativeBinding(program.stage, *kind) : UINT32_MAX,
			    resource.descriptor_index, resource.read, resource.written, resource.atomic64,
			    Words(std::span {raw.dwords}.first(raw.dword_count)), descriptor.Base40(),
			    uint32_t {descriptor.Width5()} + 1u, uint32_t {descriptor.Height5()} + 1u,
			    uint32_t {descriptor.Depth()} + 1u, static_cast<uint32_t>(descriptor.Format()),
			    descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(),
			    descriptor.BaseArray5(), static_cast<uint32_t>(descriptor.TileMode()),
			    vk::to_string(binding.desc.view_info.format), binding.image_id.index,
			    binding.image_id.generation, image != nullptr);
			if (image == nullptr) continue;
			aliases.push_back({stage_index, i, binding.image_id, binding.desc.info.data});
			log += fmt::format(
			    "MeshResources: image_backing capture={} stage_index={} resource={} handle=0x{:x} "
			    "format={} extent={}x{}x{} levels={} layers={} samples={} guest=0x{:x} size={} "
			    "metadata=0x{:x}:{}:{} layout={} descriptor_layout={} force_general={} target={} "
			    "shader_write={} registered={} usage=0x{:x} flags=0x{:x}\n",
			    capture, stage_index, i, GpuDiagnostics::HandleValue(image->backing.image),
			    vk::to_string(image->backing.format), image->backing.extent.width,
			    image->backing.extent.height, image->backing.extent.depth, image->backing.mip_levels,
			    image->backing.layers, image->backing.samples, image->info.data.address,
			    image->info.data.size, image->info.metadata.range.address,
			    image->info.metadata.range.size, static_cast<uint32_t>(image->info.metadata.kind),
			    vk::to_string(image->backing.state.layout), vk::to_string(binding.layout),
			    image->binding.force_general, image->binding.is_target, image->binding.shader_write,
			    image->registered, static_cast<uint32_t>(image->backing.usage),
			    static_cast<uint32_t>(image->backing.flags));
			const auto count = binding.mip_views.empty() ? 1u : binding.mip_views.size();
			for (uint32_t mip = 0; mip < count; ++mip) {
				const auto view = MakeImageInfo(binding, mip);
				const auto found = std::ranges::find(image->views, view.imageView, &CachedImageView::view);
				log += fmt::format(
				    "MeshResources: image_view capture={} stage_index={} resource={} mip={} "
				    "handle=0x{:x} layout={} cached={}\n", capture, stage_index, i, mip,
				    GpuDiagnostics::HandleValue(view.imageView), vk::to_string(view.imageLayout),
				    found != image->views.end());
				if (found == image->views.end()) continue;
				const auto& info = found->info;
				log += fmt::format(
				    "MeshResources: view_info capture={} stage_index={} resource={} mip={} format={} "
				    "type={} aspect=0x{:x} levels={}:{} layers={}:{} min_lod={} "
				    "swizzle={},{},{},{} usage=0x{:x}\n", capture, stage_index, i, mip,
				    vk::to_string(info.format), vk::to_string(info.type), static_cast<uint32_t>(info.aspect),
				    info.base_level, info.level_count, info.base_layer, info.layer_count, info.min_lod,
				    static_cast<uint32_t>(info.mapping.r), static_cast<uint32_t>(info.mapping.g),
				    static_cast<uint32_t>(info.mapping.b), static_cast<uint32_t>(info.mapping.a),
				    static_cast<uint32_t>(info.usage));
				// backing.state is an aggregate; partial mip transitions retain a
				// separate state per level/layer. Inspect the actual view's range too.
				uint32_t checked = 0, mismatched = 0;
				auto first_layout = image->backing.state.layout;
				const auto& states = image->backing.subresource_states;
				if (states.empty()) {
					checked = 1;
					mismatched = first_layout != view.imageLayout;
				} else {
					const uint32_t base_layer = image->info.IsVolume() ? 0u : info.base_layer;
					const uint32_t layers = image->info.IsVolume() ? 1u : info.layer_count;
					for (uint64_t level = info.base_level;
					     level < uint64_t {info.base_level} + info.level_count &&
					     level < image->info.resources.levels; ++level) {
						for (uint64_t layer = base_layer;
						     layer < uint64_t {base_layer} + layers &&
						     layer < image->info.resources.layers; ++layer) {
							const auto index = level * image->info.resources.layers + layer;
							if (index >= states.size()) continue;
							if (checked == 0) first_layout = states[index].layout;
							++checked;
							mismatched += states[index].layout != view.imageLayout;
						}
					}
				}
				log += fmt::format(
				    "MeshResources: view_layouts capture={} stage_index={} resource={} mip={} "
				    "tracked={} checked={} mismatched={} first={}\n", capture, stage_index, i,
				    mip, !states.empty(), checked, mismatched, vk::to_string(first_layout));
			}
		}
	}
	for (size_t i = 0; i < aliases.size(); ++i) {
		for (size_t j = i + 1; j < aliases.size(); ++j) {
			const auto& a = aliases[i];
			const auto& b = aliases[j];
			const bool overlap = !a.range.Empty() && !b.range.Empty() &&
			    ImageRangeOverlaps(a.range.address, a.range.size, b.range.address, b.range.size);
			if (a.id == b.id || overlap) {
				log += fmt::format(
				    "MeshResources: image_alias capture={} a={}:{} b={}:{} same_image={} "
				    "guest_overlap={}\n", capture, a.stage, a.resource, b.stage, b.resource,
				    a.id == b.id, overlap);
			}
		}
	}
	log += fmt::format("MeshResources: end capture={} limit_reached={}\n", capture, capture == 63u);
	::Log::WriteToConsoleAndLog(log.c_str());
	std::fflush(stdout);
}

} // namespace ZeroCompanyMeshResources
