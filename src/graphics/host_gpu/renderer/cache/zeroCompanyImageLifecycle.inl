// Included inside Libs::Graphics. Observes existing CPU metadata only; never
// reads guest bytes, resolves cache entries, submits commands, or waits.
// "returned" means the CPU recording helper returned, not GPU completion.
// value0/value1: overlap-examine = safe-to-delete/binding type; copy-mip = mip/layer;
// upload/download = buffer handle/offset; invalidate-cpu = previous flags;
// gc-free = GC tick/age. Unavailable ImageIds are explicitly marked invalid.
namespace ZeroCompanyImageLifecycle {

constexpr uint64_t TargetSize = 0x23f0000ull;
constexpr uint64_t EventLimit = 4096;

struct Details {
	ImageId other_id {};
	const Image* other = nullptr;
	const ImageInfo* request = nullptr;
	uint64_t address = 0;
	uint64_t size = 0;
	uint64_t value0 = 0;
	uint64_t value1 = 0;
};

static bool Matches(const ImageInfo& info) {
	// Guest allocation addresses vary between boots. Match the complete observed
	// surface signature from creation onward, then correlate actual addresses with
	// MeshResources. For a 2D surface, kRenderTarget selects RenderTarget64KB.
	return info.data.address != 0 && info.data.size == TargetSize &&
	       info.extent == vk::Extent3D {2848, 1600, 1} &&
	       info.guest_format == Prospero::BufferFormat::k32_32UInt &&
	       info.pixel_format == vk::Format::eR32G32Uint &&
	       info.tile_mode == Prospero::TileMode::kRenderTarget &&
	       info.type == Prospero::ImageType::kColor2D && info.resources.levels == 1 &&
	       info.resources.layers == 1 && info.samples == 1 &&
	       info.metadata.kind == ImageMetadataKind::None && info.metadata.range.Empty() &&
	       info.metadata.compression == VideoOutCompression::Uncompressed;
}

static uint32_t Flags(const Image& image) {
	return uint32_t {image.IsGpuModified()} | (uint32_t {image.IsDefinitelyCpuDirty()} << 1u) |
	       (uint32_t {image.IsMaybeCpuDirty()} << 2u) | (uint32_t {image.IsBufferModified()} << 3u) |
	       (uint32_t {image.registered} << 4u) | (uint32_t {image.binding.is_bound} << 5u) |
	       (uint32_t {image.binding.is_target} << 6u) | (uint32_t {image.binding.needs_rebind} << 7u);
}

static void Record(const CommandScheduler& scheduler, const char* reason, ImageId id,
                   const Image& image, Details details = {}) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PROBE_IMAGE_LIFECYCLE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (!enabled || !GpuDiagnostics::Enabled() ||
	    (!Matches(image.info) && !(details.other && Matches(details.other->info)) &&
	     !(details.request && Matches(*details.request)))) return;
	static std::atomic<uint64_t> events {0};
	const auto sequence = events.fetch_add(1, std::memory_order_relaxed);
	if (sequence >= EventLimit) return;
	std::string line = fmt::format(
	    "ImageLifecycle: seq={} selector=2848x1600x1-RG32Uint-RenderTarget64KB-size0x23f0000-1mip-1layer-1sample-no-metadata "
	    "tick={} reason={} id={}:{} id_valid={} "
	    "handle=0x{:x} guest=0x{:x}:0x{:x} format={} extent={}x{}x{} levels={} layers={} "
	    "tile={} layout={} access=0x{:x} flags=0x{:x} last_use={} "
	    "address=0x{:x} size=0x{:x} value0=0x{:x} value1=0x{:x}",
	    sequence, scheduler.CurrentTick(), reason, id.index, id.generation,
	    static_cast<bool>(id),
	    GpuDiagnostics::HandleValue(image.backing.image), image.info.data.address, image.info.data.size,
	    vk::to_string(image.backing.format), image.info.extent.width, image.info.extent.height,
	    image.info.extent.depth, image.info.resources.levels, image.info.resources.layers,
	    static_cast<uint32_t>(image.info.tile_mode), vk::to_string(image.backing.state.layout),
	    static_cast<uint64_t>(image.backing.state.access_mask), Flags(image), image.tick_accessed_last,
	    details.address, details.size, details.value0, details.value1);
	if (details.other) {
		const auto& other = *details.other;
		line += fmt::format(
		    " other={}:{} other_handle=0x{:x} other_guest=0x{:x}:0x{:x} other_format={} "
		    "other_layout={} other_flags=0x{:x} other_last_use={}",
		    details.other_id.index, details.other_id.generation,
		    GpuDiagnostics::HandleValue(other.backing.image), other.info.data.address,
		    other.info.data.size, vk::to_string(other.backing.format),
		    vk::to_string(other.backing.state.layout), Flags(other), other.tick_accessed_last);
	}
	if (details.request) {
		const auto& request = *details.request;
		line += fmt::format(
		    " request=0x{:x}:0x{:x} request_format={} request_extent={}x{}x{} "
		    "request_levels={} request_layers={} request_tile={}", request.data.address,
		    request.data.size, vk::to_string(request.pixel_format), request.extent.width,
		    request.extent.height, request.extent.depth, request.resources.levels,
		    request.resources.layers, static_cast<uint32_t>(request.tile_mode));
	}
	line += fmt::format(" limit_reached={} flags_bits=gpu,cpu,maybe,buffer,registered,bound,target,rebind\n",
	                    sequence + 1 == EventLimit);
	::Log::WriteToConsoleAndLog(line.c_str());
	std::fflush(stdout);
}

} // namespace ZeroCompanyImageLifecycle
