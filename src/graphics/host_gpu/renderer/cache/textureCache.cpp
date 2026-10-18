#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

vk::Extent2D TextureCache::ViewExtent(const Image& image, uint32_t level, vk::Format view_format) {
	const auto mip_width  = std::max(image.info.extent.width >> level, 1u);
	const auto mip_height = std::max(image.info.extent.height >> level, 1u);
	const auto source     = vk::blockExtent(image.backing.format);
	const auto target     = vk::blockExtent(view_format);
	if (source[0] == target[0] && source[1] == target[1]) {
		return {mip_width, mip_height};
	}
	return {(mip_width + source[0] - 1) / source[0] * target[0],
	        (mip_height + source[1] - 1) / source[1] * target[1]};
}

#include "zeroCompanyImageLifecycle.inl"

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 32;

// Rendering-defect forensics (review 2026-10-13): a request that overlaps a live GPU-owned image but
// matches neither its backing nor an overlap rule creates a second VkImage over the same guest
// memory. Tiled GPU pixels never reach guest memory, so the second image is filled from stale
// bytes (briefing-panel hypothesis: gray box / stripes / noise). The same applies when an overlap
// rule frees a GPU-owned image instead of reusing it. Log both signatures once per
// (owner address, request signature, kind).
std::string DescribeImageInfo(const ImageInfo& info) {
	return fmt::format("0x{:x}+0x{:x} {}x{}x{} mips={} layers={} samples={} {} guest={} type={} "
	                   "tile={} bpb={} pitch={} meta={}:0x{:x}+0x{:x}",
	                   info.data.address, info.data.size, info.extent.width, info.extent.height,
	                   info.extent.depth, info.resources.levels, info.resources.layers, info.samples,
	                   vk::to_string(info.pixel_format), static_cast<uint32_t>(info.guest_format),
	                   static_cast<uint32_t>(info.type), static_cast<uint32_t>(info.tile_mode),
	                   info.bytes_per_block, info.pitch, static_cast<uint32_t>(info.metadata.kind),
	                   info.metadata.range.address, info.metadata.range.size);
}

// KYTY_PROBE_KEEP_GPU_IMAGES=1 (discriminator 2026-10-13): a partial GPU buffer write at an image's
// base, or a CPU write fault inside a GPU-owned image, currently hands authority to guest memory and
// the next bind re-uploads the whole surface (stale outside the written bytes) over the GPU pixels.
// The probe keeps such images GPU-owned instead (the written bytes are lost rather than the GPU
// pixels); a visible difference means these re-uploads are what blackens GPU-rendered content.
// =1: ignore both partial GPU buffer writes and CPU write faults; =2: ignore only the partial GPU
// buffer writes (ng-61: ignoring CPU writes blacked out the creator preview, so CPU writes into
// GPU-owned images carry real content and must keep uploading).
int KeepGpuImagesProbe() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_PROBE_KEEP_GPU_IMAGES");
		return value != nullptr && value[1] == '\0' && (value[0] == '1' || value[0] == '2')
		           ? value[0] - '0'
		           : 0;
	}();
	return level;
}

void ReportGpuOwnedOverlap(const char* kind, TextureCache::BindingType binding,
                           const ImageInfo& request, const ImageInfo& owner, uint64_t owner_age) {
	static std::unordered_set<uint64_t> reported; // texture-cache lock held by the callers
	static uint32_t                     alias_lines = 0;
	// ng-55: transient aliasing (different base or different extent) dominates and is legitimate
	// (UE5 RDG reuses heap memory within a frame); keep it to a small sample and spend the budget
	// on same-base/same-extent requests, which are views of one surface.
	// ng-56: same base + same extent with a different element size is still RDG reuse of a larger
	// allocation; a view of one surface keeps base, extent, element size and backing size.
	const bool same_surface = owner.data.address == request.data.address &&
	                          owner.extent == request.extent &&
	                          owner.bytes_per_block * owner.samples ==
	                              request.bytes_per_block * request.samples &&
	                          owner.data.size == request.data.size;
	if (!same_surface && alias_lines++ >= 48) {
		return;
	}
	const uint64_t key = owner.data.address ^ (uint64_t {request.extent.width} << 32u) ^
	                     (uint64_t {request.extent.height} << 48u) ^
	                     (static_cast<uint64_t>(request.pixel_format) << 8u) ^
	                     (static_cast<uint64_t>(request.tile_mode) << 4u) ^ request.data.size ^
	                     (uint64_t {static_cast<uint8_t>(binding)} << 60u) ^
	                     (kind[0] == 's' ? 0u : 1u);
	if (reported.size() >= 256 || !reported.insert(key).second) {
		return;
	}
	LOGF("Image%s: %s request type=%d %s | gpu-owned age=%" PRIu64 " %s\n", kind,
	     same_surface ? "same-surface" : "alias", static_cast<int>(binding),
	     DescribeImageInfo(request).c_str(), owner_age, DescribeImageInfo(owner).c_str());
}

[[nodiscard]] bool DecodeColorClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.view_info.format;
	const bool  cmask    = metadata.kind == ImageMetadataKind::Cmask;
	if (cmask ? code == 0 : code == 0x20) {
		// Register clears belong to the color buffer; the texture pipe cannot decode them.
		return desc.type == TextureCache::BindingType::RenderTarget &&
		       metadata.clear_register_valid &&
		       DecodePackedColorClear(format, metadata.clear_word, clear);
	}
	if (cmask) {
		return false;
	}
	switch (code) {
		case 0x00:
		case 0x40:
		case 0x80:
		case 0xc0: break;
		default: return false;
	}
	clear = {};
	if (code == 0x00) {
		return true;
	}
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eR5G6B5UnormPack16:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR4G4B4A4UnormPack16:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32: break;
		default: return false;
	}
	const float          rgb   = (code & 0x80u) != 0 ? 1.0f : 0.0f;
	const float          alpha = (code & 0x40u) != 0 ? 1.0f : 0.0f;
	std::array<float, 4> channels {rgb, rgb, rgb, alpha};
	if (!metadata.dcc_alpha_msb) {
		std::swap(channels[0], channels[3]);
	}
	// DCC clear decoding clamps missing lanes before applying the format swizzle.
	const auto components = vk::componentCount(format);
	if (components == 1) {
		channels[0] = channels[3];
	} else if (components == 2) {
		channels[1] = channels[3];
	}
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR5G6B5UnormPack16: std::swap(channels[0], channels[2]); break;
		case vk::Format::eR4G4B4A4UnormPack16:
			std::reverse(channels.begin(), channels.end());
			break;
		default: break;
	}
	clear.float32 = channels;
	return true;
}

[[nodiscard]] std::vector<vk::BufferImageCopy> BuildDepthCopies(const ImageInfo& info,
                                                              uint64_t slice_stride,
                                                              vk::ImageAspectFlags aspect) {
	std::vector<vk::BufferImageCopy> copies(info.resources.layers);
	for (uint32_t layer = 0; layer < info.resources.layers; ++layer) {
		auto& copy             = copies[layer];
		copy.bufferOffset      = slice_stride * layer;
		copy.bufferRowLength   = info.pitch;
		copy.bufferImageHeight = info.extent.height;
		copy.imageSubresource  = {aspect, 0, layer, 1};
		copy.imageExtent       = {info.extent.width, info.extent.height, 1};
	}
	return copies;
}

[[nodiscard]] std::vector<GpuTileInfo> BuildDepthTiles(const ImageInfo& info) {
	TileBlockLayout block {};
	EXIT_NOT_IMPLEMENTED(
	    !TileGetBlockLayout(TileBlockFamily::Depth64KB, info.bytes_per_block, block));
	const auto               full_slice_size = info.data.size / info.resources.layers;
	std::vector<GpuTileInfo> tiles;
	tiles.reserve(info.resources.layers);
	for (uint32_t layer = 0; layer < info.resources.layers; ++layer) {
		const auto offset = full_slice_size * layer;
		tiles.push_back({block.family, block.bytes_per_element, offset, full_slice_size, offset,
		                 full_slice_size, 0, info.extent.width, info.extent.height, 1, info.pitch});
		tiles.back().surface_z = layer;
	}
	return tiles;
}

} // namespace

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)),
      m_buffer_cache(buffer_cache),
      m_readback_linear_images(Config::ReadbackLinearImagesEnabled()) {
	if (m_graphics.CanReportMemoryUsage()) {
		constexpr int64_t GiB = 1024ll * 1024 * 1024;
		const auto        budget =
		    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
		const auto threshold = std::min<int64_t>(budget, 8 * GiB);
		m_pressure_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
		m_critical_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
		m_trigger_gc_memory = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
	}
}

TextureCache::~TextureCache() {
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered) {
			UnregisterImage(id);
		}
	});
}

bool TextureCache::SameBacking(const ImageInfo& cached, const ImageInfo& requested,
                               bool exact_format) {
	if (cached.data.address != requested.data.address) {
		return false;
	}
	if (cached.data.size != requested.data.size) {
		return false;
	}
	if (cached.extent != requested.extent) {
		return false;
	}
	if (cached.resources.levels < requested.resources.levels ||
	    cached.resources.layers < requested.resources.layers) {
		return false;
	}
	if (cached.samples != requested.samples) {
		return false;
	}
	if (cached.bytes_per_block != requested.bytes_per_block) {
		return false;
	}
	if (cached.tile_mode != requested.tile_mode) {
		return false;
	}
	if (cached.GetColorTransform() != requested.GetColorTransform()) {
		return false;
	}
	if (!ImageViewOps::FormatsCompatible(cached.pixel_format, requested.pixel_format) ||
	    cached.type != requested.type) {
		return false;
	}
	if (exact_format && cached.pixel_format != requested.pixel_format) {
		return false;
	}
	return true;
}

TextureCache::BindingType TextureCache::UploadBinding(const Image& image) {
	if (image.info.IsDepth()) {
		if (image.info.tile_mode == Prospero::TileMode::kDepth ||
		    image.info.tile_mode == Prospero::TileMode::kLinear) {
			return BindingType::DepthTarget;
		}
		return BindingType::Texture;
	}
	if (image.usage.render_target) {
		return BindingType::RenderTarget;
	}
	if (image.usage.video_out) {
		return BindingType::VideoOut;
	}
	return image.usage.storage ? BindingType::Storage : BindingType::Texture;
}

ImageId TextureCache::InsertImage(const ImageInfo& info) {
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	if (!info.data.Empty()) {
		RegisterImage(id);
	}
	ZeroCompanyImageLifecycle::Record(m_scheduler, "insert", id, m_slot_images[id]);
	return id;
}

void TextureCache::RegisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.registered || image.info.data.Empty()) {
		EXIT("TextureCache: invalid image registration\n");
	}
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: image registration is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		m_image_page_table[page].push_back(id);
	});
	image.registered = true;
	image.lru_id     = m_lru_cache.Insert(id, m_gc_tick);
	m_total_used_memory += image.AccountedSize();
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	UntrackImage(id);
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: registered image is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr || !owners->Erase(id)) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
	});
	m_lru_cache.Free(image.lru_id);
	const auto accounted = image.AccountedSize();
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	image.registered = false;
}

void TextureCache::DeleteImage(ImageId id) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return;
	}
	ZeroCompanyImageLifecycle::Record(m_scheduler, "delete-begin", id, *image);
	if (image->depth_id) {
		std::erase(m_slot_images[image->depth_id].stencil_ids, id);
	} else {
		// Detach first: deleting each association also unlinks it from its depth owner.
		for (const auto association: std::exchange(image->stencil_ids, {})) {
			FreeImage(association);
		}
	}
	if (image->IsGpuModified()) {
		EXIT("TextureCache: deleting a GPU-modified image without resolving its contents\n");
	}
	m_download_images.erase(id);
	if (image->info.HasMetadata()) {
		const auto metadata = m_surface_metas.find(image->info.metadata.range.address);
		if (metadata != m_surface_metas.end() &&
		    image->info.metadata.kind == ImageMetadataKind::Htile &&
		    metadata->second.type == MetaDataInfo::Type::HTile) {
			// A later binding may have reused this address for another metadata type.
			m_surface_metas.erase(metadata);
		}
	}
	UnregisterImage(id);
	ZeroCompanyImageLifecycle::Record(m_scheduler, "delete-unregistered", id, *image);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] {
			ZeroCompanyImageLifecycle::Record(m_scheduler, "delete-deferred-erase", id, m_slot_images[id]);
			m_slot_images.erase(id);
		});
	} else {
		ZeroCompanyImageLifecycle::Record(m_scheduler, "delete-immediate-erase", id, m_slot_images[id]);
		m_slot_images.erase(id);
	}
}

void TextureCache::FreeImage(ImageId id) {
	auto& image = m_slot_images[id];
	ZeroCompanyImageLifecycle::Record(m_scheduler, "free-before-ownership-clear", id, image);
	if (image.IsGpuModified()) {
		image.ClearGpuModified();
	}
	DeleteImage(id);
}

void TextureCache::TouchImage(Image& image) {
	if (image.registered) {
		m_lru_cache.Touch(image.lru_id, m_gc_tick);
	}
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	image.MarkMaybeCpuDirty();
	if (image.NeedsMaybeCpuHash()) {
		image.SetMaybeCpuHash(image.HashGuestEdges());
	}
	UntrackImage(id);
}

void TextureCache::TrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	const auto image_end   = image.info.data.End();
	if (image_begin == image.track_addr && image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked()) {
		image.track_addr     = image_begin;
		image.track_addr_end = image_end;
		m_page_manager.UpdatePageWatchers<true>(image_begin, image.info.data.size);
		return;
	}
	if (image_begin < image.track_addr) {
		TrackImageHead(id);
	}
	if (image.track_addr_end < image_end) {
		TrackImageTail(id);
	}
}

void TextureCache::TrackImageHead(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	if (image_begin == image.track_addr) {
		return;
	}
	if (!image.IsTracked() || image_begin > image.track_addr) {
		EXIT("TextureCache: invalid image head tracking range\n");
	}
	const auto size  = image.track_addr - image_begin;
	image.track_addr = image_begin;
	m_page_manager.UpdatePageWatchers<true>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_end = image.info.data.End();
	if (image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked() || image.track_addr_end > image_end) {
		EXIT("TextureCache: invalid image tail tracking range\n");
	}
	const auto address   = image.track_addr_end;
	const auto size      = image_end - address;
	image.track_addr_end = image_end;
	m_page_manager.UpdatePageWatchers<true>(address, size);
}

void TextureCache::UntrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.IsTracked()) {
		return;
	}
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	image.track_addr     = 0;
	image.track_addr_end = 0;
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::UntrackImageHead(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto begin = image.info.data.address;
	if (!image.IsTracked() || begin < image.track_addr) {
		return;
	}
	const auto address = Common::AlignDown(begin + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
	const auto size    = address - begin;
	image.track_addr   = address;
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(begin, size);
	}
}

void TextureCache::UntrackImageTail(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto end   = image.info.data.End();
	if (!image.IsTracked() || image.track_addr_end < end) {
		return;
	}
	const auto address   = Common::AlignDown(end, TRACKER_PAGE_SIZE);
	const auto size      = end - address;
	image.track_addr_end = address;
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

// KYTY_IMAGE_READBACK=0 restores unprotected GPU-written linear images (the game's CPU then reads
// whatever guest memory holds; Zero Company's dossier portrait became heap garbage, R28 2026-10-14).
static bool ImageReadbackEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IMAGE_READBACK");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

void TextureCache::TrackImageDownload(ImageId id, Image& image) {
	if (m_readback_linear_images && !image.info.IsTiled() && !image.info.data.Empty()) {
		if (!image.IsGpuModified()) {
			EXIT("TextureCache: cannot enroll a non-GPU-owned image for download\n");
		}
		m_download_images.insert(id);
	}
}

TextureCache::ImageIds TextureCache::FindImagesInRegion(uint64_t address, uint64_t size,
                                                        bool page_overlap) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return {};
	}

	uint32_t query_epoch = ++m_image_query_epoch;
	if (query_epoch == 0) {
		m_slot_images.ForEach([](ImageId, const Image& image) { image.query_epoch = 0; });
		query_epoch = ++m_image_query_epoch;
	}

	ImageIds result;
	ForEachPage(address, size, [&](uint64_t page) {
		const auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr) {
			return;
		}
		owners->ForEach([&](ImageId id) {
			auto* image = m_slot_images.try_get(id);
			if (image == nullptr) {
				return;
			}
			if (image->query_epoch == query_epoch) {
				return;
			}
			image->query_epoch = query_epoch;
			if (image->Overlaps(address, size, page_overlap)) {
				result.push_back(id);
			}
		});
	});
	return result;
}

ImageId TextureCache::GetNullImage(const ImageDesc& desc) {
	const auto format = desc.info.pixel_format;
	if (const auto found = m_null_images.find(format); found != m_null_images.end()) {
		return found->second;
	}
	ImageInfo info {};
	info.pixel_format    = desc.info.pixel_format;
	info.guest_format    = desc.info.guest_format;
	info.type            = Prospero::ImageType::kColor2D;
	info.extent          = {1, 1, 1};
	info.resources       = {1, 1};
	info.pitch           = 1;
	info.bytes_per_block = std::max(desc.info.bytes_per_block, 1u);
	info.samples         = 1;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.mip_layout[0]   = {0, info.bytes_per_block, 1, 1};
	const auto id        = InsertImage(info);
	m_null_images.emplace(format, id);
	return id;
}

void TextureCache::ValidateImageDesc(const ImageDesc& desc) const {
	ImageOps::Validate(desc.info);
	if (desc.view_info.format == vk::Format::eUndefined || desc.view_info.level_count == 0 ||
	    desc.view_info.layer_count == 0 ||
	    desc.view_info.base_level >= desc.info.resources.levels ||
	    desc.view_info.level_count > desc.info.resources.levels - desc.view_info.base_level ||
	    (!desc.info.IsVolume() &&
	     (desc.view_info.base_layer >= desc.info.resources.layers ||
	      desc.view_info.layer_count > desc.info.resources.layers - desc.view_info.base_layer))) {
		EXIT("TextureCache: invalid image view description\n");
	}
	if (desc.type == BindingType::DepthTarget && !IsSupportedDepthTargetFormat(desc.info)) {
		EXIT("TextureCache: unsupported depth image description\n");
	}
	if (desc.type == BindingType::VideoOut && !IsSupportedVideoOutFormat(desc.info)) {
		EXIT("TextureCache: unsupported video-out image description\n");
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression == VideoOutCompression::Unsupported) {
		EXIT("TextureCache: unsupported compressed video-out description\n");
	}
}

void TextureCache::PrepareImageCopy(Image& image) {
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

void TextureCache::RefreshCopySource(ImageId id) {
	auto& image = m_slot_images[id];
	RefreshImage(id);
	if (image.IsDefinitelyCpuDirty()) {
		EXIT("TextureCache: image copy source remained CPU-dirty after refresh\n");
	}
}

bool TextureCache::CopyD16(Image& destination, Image& source) {
	const bool source_depth      = source.info.IsDepth();
	const bool destination_depth = destination.info.IsDepth();
	if (source_depth == destination_depth) {
		return false;
	}
	auto&      depth          = source_depth ? source : destination;
	auto&      color          = source_depth ? destination : source;
	const auto transfer_bytes = DepthAspectTransferBytes(depth.backing.format);
	if (depth.info.bytes_per_block != sizeof(uint16_t) ||
	    color.info.bytes_per_block != sizeof(uint16_t) || transfer_bytes != sizeof(uint32_t)) {
		return false;
	}
	EXIT_IF(source.backing.samples != 1 || destination.backing.samples != 1 ||
	        source.info.resources.levels != 1 || destination.info.resources.levels != 1 ||
	        source.info.extent != destination.info.extent ||
	        source.info.resources.layers != destination.info.resources.layers);

	const auto     layers = depth.info.resources.layers;
	const uint64_t depth_slice =
	    static_cast<uint64_t>(depth.info.pitch) * depth.info.extent.height * transfer_bytes;
	const uint64_t color_slice =
	    static_cast<uint64_t>(color.info.pitch) * color.info.extent.height * sizeof(uint16_t);
	EXIT_IF(layers == 0 || depth_slice > UINT64_MAX / layers || color_slice > UINT64_MAX / layers);
	const auto                       depth_size = depth_slice * layers;
	const auto                       color_size = color_slice * layers;
	std::vector<vk::BufferImageCopy> depth_copies(layers);
	std::vector<vk::BufferImageCopy> color_copies(layers);
	for (uint32_t layer = 0; layer < layers; layer++) {
		depth_copies[layer].bufferOffset      = depth_slice * layer;
		depth_copies[layer].bufferRowLength   = depth.info.pitch;
		depth_copies[layer].bufferImageHeight = depth.info.extent.height;
		depth_copies[layer].imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, layer, 1};
		depth_copies[layer].imageExtent       = depth.info.extent;
		color_copies[layer].bufferOffset      = color_slice * layer;
		color_copies[layer].bufferRowLength   = color.info.pitch;
		color_copies[layer].bufferImageHeight = color.info.extent.height;
		color_copies[layer].imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, layer, 1};
		color_copies[layer].imageExtent       = color.info.extent;
	}

	auto                         depth_buffer = m_tiler.GetScratchBuffer(depth_size);
	auto                         color_buffer = m_tiler.GetScratchBuffer(color_size, depth_buffer.buffer);
	const TileManager::D16Layout promote_layout {
	    .width               = depth.info.extent.width,
	    .height              = depth.info.extent.height,
	    .layers              = layers,
	    .source_row_stride   = static_cast<uint64_t>(color.info.pitch) * sizeof(uint16_t),
	    .target_row_stride   = static_cast<uint64_t>(depth.info.pitch) * transfer_bytes,
	    .source_slice_stride = color_slice,
	    .target_slice_stride = depth_slice,
	};
	const bool d32 = DepthAspectTransferFormat(depth.backing.format) == vk::Format::eD32Sfloat;
	if (source_depth) {
		source.Download(depth_copies, depth_buffer.buffer, depth_buffer.offset, depth_buffer.size);
		m_tiler.ConvertD16(depth_buffer, color_buffer, TileManager::D16Direction::Demote, d32,
		                   {.width               = promote_layout.width,
		                    .height              = promote_layout.height,
		                    .layers              = promote_layout.layers,
		                    .source_row_stride   = promote_layout.target_row_stride,
		                    .target_row_stride   = promote_layout.source_row_stride,
		                    .source_slice_stride = promote_layout.target_slice_stride,
		                    .target_slice_stride = promote_layout.source_slice_stride});
		destination.Upload(color_copies, color_buffer.buffer, color_buffer.offset,
		                   color_buffer.size);
	} else {
		source.Download(color_copies, color_buffer.buffer, color_buffer.offset, color_buffer.size);
		m_tiler.ConvertD16(color_buffer, depth_buffer, TileManager::D16Direction::Promote, d32,
		                   promote_layout);
		destination.Upload(depth_copies, depth_buffer.buffer, depth_buffer.offset,
		                   depth_buffer.size);
	}
	return true;
}

void TextureCache::CopyImage(ImageId destination_id, ImageId source_id) {
	ZeroCompanyImageLifecycle::Record(m_scheduler, "copy-begin", destination_id,
	    m_slot_images[destination_id], {.other_id = source_id, .other = &m_slot_images[source_id]});
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: cannot issue an unequal-sample image copy\n");
	}
	PrepareImageCopy(destination);
	if (source.IsBufferModified()) {
		if (source.info.data == destination.info.data) {
			destination.MarkBufferModified();
		}
		ZeroCompanyImageLifecycle::Record(m_scheduler, "copy-skip-buffer-owned", destination_id,
		    destination, {.other_id = source_id, .other = &source});
		return;
	}
	const bool source_depth = source.info.IsDepth();
	const bool dest_depth   = destination.info.IsDepth();
	const bool direct_copy =
	    source.info.GetColorTransform() == destination.info.GetColorTransform() &&
	    (source.backing.image_type == destination.backing.image_type ||
	     (source.backing.image_type != vk::ImageType::e1D &&
	      destination.backing.image_type != vk::ImageType::e1D)) &&
	    (source.backing.format == destination.backing.format ||
	     (!source_depth && !dest_depth &&
	      vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format)));
	if (direct_copy) {
		ZeroCompanyImageLifecycle::Record(m_scheduler, "copy-direct-record", destination_id,
		    destination, {.other_id = source_id, .other = &source});
		destination.CopyImage(source);
	} else if (!CopyD16(destination, source)) {
		if (source.backing.samples != 1 || destination.backing.samples != 1) {
			EXIT("TextureCache: cross-format multisample image copy is unsupported\n");
		}
		auto& copy_buffer = m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
		ZeroCompanyImageLifecycle::Record(m_scheduler, "copy-via-buffer-record", destination_id,
		    destination, {.other_id = source_id, .other = &source,
		                  .value0 = GpuDiagnostics::HandleValue(copy_buffer.Handle())});
		destination.CopyImageWithBuffer(source, copy_buffer, m_tiler);
	}
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
	destination.ClearBufferModified();
	ZeroCompanyImageLifecycle::Record(m_scheduler, "copy-returned", destination_id,
	    destination, {.other_id = source_id, .other = &source});
}

void TextureCache::CopyImageMip(ImageId destination_id, ImageId source_id, uint32_t mip,
                                uint32_t layer) {
	ZeroCompanyImageLifecycle::Record(m_scheduler, "copy-mip-begin", destination_id,
	    m_slot_images[destination_id], {.other_id = source_id, .other = &m_slot_images[source_id],
	                                    .value0 = mip, .value1 = layer});
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.IsBufferModified() || source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: invalid mip-copy ownership or sample count\n");
	}
	destination.CopyMip(source, mip, layer);
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
	ZeroCompanyImageLifecycle::Record(m_scheduler, "copy-mip-returned", destination_id,
	    destination, {.other_id = source_id, .other = &source, .value0 = mip, .value1 = layer});
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	auto& cached = m_slot_images[cached_id];
	if ((!cached.info.IsDepth() && !requested.IsDepth()) ||
	    cached.info.tile_mode != requested.tile_mode) {
		return {};
	}
	const bool stencil_match = requested.HasStencil() == cached.info.HasStencil();
	const bool bpp_match     = requested.bytes_per_block == cached.info.bytes_per_block;
	// PPSA04264
	const bool raw_d16_texture =
	    binding == BindingType::Texture && cached.info.IsDepth() &&
	    cached.info.guest_format == Prospero::BufferFormat::k16UNorm &&
	    requested.guest_format == Prospero::BufferFormat::k16UInt &&
	    requested.pixel_format == vk::Format::eR16Uint && cached.backing.samples == 1 &&
	    requested.samples == 1 && requested.data == cached.info.data &&
	    requested.extent == cached.info.extent && requested.resources == cached.info.resources &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    !cached.info.HasMetadata();
	// PPSA04264, PPSA04288
	// A partial view retains the entire matching array layout. HTile belongs to
	// the depth source, independently of the data slices copied into color storage.
	const bool retain_cached_layout =
	    requested.samples == 1 && cached.info.samples == 1 && cached.backing.samples == 1 &&
	    requested.bytes_per_block == cached.info.bytes_per_block &&
	    requested.data.address == cached.info.data.address &&
	    requested.data.size < cached.info.data.size && requested.extent == cached.info.extent &&
	    requested.resources.levels == 1 && cached.info.resources.levels == 1 &&
	    requested.resources.layers != 0 && cached.info.resources.layers != 0 &&
	    requested.resources.layers < cached.info.resources.layers &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    requested.mip_layout[0].offset == 0 &&
	    cached.info.mip_layout[0].offset == 0 &&
	    requested.mip_layout[0].size == requested.data.size &&
	    cached.info.mip_layout[0].size == cached.info.data.size &&
	    requested.data.size % requested.resources.layers == 0 &&
	    cached.info.data.size % cached.info.resources.layers == 0 &&
	    requested.data.size / requested.resources.layers ==
	        cached.info.data.size / cached.info.resources.layers &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    (cached.info.metadata.kind == ImageMetadataKind::None ||
	     cached.info.metadata.kind == ImageMetadataKind::Htile);
	bool recreate = cached.info.resources < requested.resources ||
	                requested.IsVolume() != cached.info.IsVolume();
	switch (binding) {
		case BindingType::Texture:
			recreate |= requested.IsDepth() && !cached.info.IsDepth();
			recreate |= raw_d16_texture;
			break;
		case BindingType::Storage: recreate |= cached.info.IsDepth(); break;
		case BindingType::RenderTarget: recreate |= cached.info.IsDepth(); break;
		case BindingType::DepthTarget:
			recreate |= !cached.info.IsDepth();
			recreate |= cached.info.IsDepth() && !(stencil_match && bpp_match);
			break;
		case BindingType::VideoOut: recreate |= cached.info.IsDepth(); break;
	}
	if (!recreate) {
		return cached_id;
	}
	ZeroCompanyImageLifecycle::Record(m_scheduler, "depth-overlap-recreate", cached_id, cached,
	    {.request = &requested, .value0 = static_cast<uint32_t>(binding)});
	RefreshImage(cached_id);
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout = cached.info.mip_layout;
	}
	info.htile_clear_mask     = 0;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
	}
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported) {
			CopyImage(replacement_id, cached_id);
		} else {
			LOGF_COLOR(Log::Color::BrightYellow,
			           "TextureCache: unsupported cross-format multisample depth copy\n");
		}
	} else if (cached.backing.samples == 1 && replacement.backing.samples > 1 &&
	           replacement.info.IsDepth()) {
		RefreshCopySource(cached_id);
		if (cached.IsBufferModified() || cached.IsDefinitelyCpuDirty()) {
			EXIT("TextureCache: multisample depth conversion source is not native-current\n");
		}
		PrepareImageCopy(replacement);
		m_blit_helper.ReinterpretColorAsMsDepth(cached, replacement);
		CommitGpuWrite(replacement);
	} else {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "TextureCache: unsupported unequal-sample depth overlap copy (%u -> %u)\n",
		           cached.backing.samples, replacement.backing.samples);
	}
	FreeImage(cached_id);
	return replacement_id;
}

TextureCache::OverlapResult TextureCache::ResolveOverlap(const ImageInfo& requested,
                                                         BindingType binding, ImageId cached_id,
                                                         ImageId merged_id) {
	auto owner = m_slot_images.try_get(cached_id);
	if (owner == nullptr) {
		return {merged_id};
	}
	auto&      cached       = *owner;
	const auto current_tick = m_scheduler.CurrentTick();
	const bool safe_to_delete =
	    current_tick - std::min(current_tick, cached.tick_accessed_last) > NumFramesBeforeRemoval;
	ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-examine", cached_id, cached,
	    {.request = &requested, .value0 = safe_to_delete, .value1 = static_cast<uint32_t>(binding)});

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			return {depth_id};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     requested.mip_layout != cached.info.mip_layout)) {
			ZeroCompanyImageLifecycle::Record(m_scheduler,
			    safe_to_delete ? "overlap-layout-free" : "overlap-layout-retain-young",
			    cached_id, cached, {.request = &requested});
			if (safe_to_delete) {
				if (cached.IsGpuModified()) {
					ReportGpuOwnedOverlap("FreeGpuOwned", binding, requested, cached.info,
					                      current_tick - std::min(current_tick, cached.tick_accessed_last));
				}
				FreeImage(cached_id);
			}
			return {merged_id};
		}
		if (requested.GetColorTransform() != cached.info.GetColorTransform()) {
			ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-expand-transform", cached_id,
			    cached, {.request = &requested});
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-expand-block", cached_id,
			    cached, {.request = &requested});
			return {ExpandImage(requested, cached_id)};
		}
		// Volume depth is not an array-layer count. A larger depth can retain the
		// same block-slice layout while requiring a larger native image.
		if ((requested.IsVolume() || cached.info.IsVolume()) &&
		    (requested.data.size == cached.info.data.size ||
		     (requested.type == cached.info.type && requested.resources == cached.info.resources &&
		      requested.extent.width == cached.info.extent.width &&
		      requested.extent.height == cached.info.extent.height &&
		      requested.extent.depth > cached.info.extent.depth))) {
			ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-expand-volume", cached_id,
			    cached, {.request = &requested});
			return {ExpandImage(requested, cached_id)};
		}
		// PPSA08394
		// A view cannot change the native image type or grow its extent.
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
		    (requested.type != cached.info.type
		         ? requested.extent == cached.info.extent
		         : requested.extent.width > cached.info.extent.width &&
		               requested.extent.height >= cached.info.extent.height &&
		               requested.extent.depth >= cached.info.extent.depth)) {
			ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-expand-type-extent", cached_id,
			    cached, {.request = &requested});
			return {ExpandImage(requested, cached_id)};
		}
		// PS5 mip tails can expose more levels without increasing the guest allocation.
		if (requested.pixel_format == cached.info.pixel_format &&
		    requested.type == cached.info.type && requested.resources > cached.info.resources &&
		    (requested.data.size > cached.info.data.size ||
		     (requested.data.size == cached.info.data.size &&
		      requested.extent == cached.info.extent &&
		      cached.info.resources.levels > 1 &&
		      requested.resources.layers == cached.info.resources.layers))) {
			ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-expand-mips", cached_id,
			    cached, {.request = &requested});
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.type != cached.info.type) {
			ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-reject-type", cached_id,
			    cached, {.request = &requested});
			return {merged_id};
		}
		if (requested.pixel_format != cached.info.pixel_format ||
		    requested.data.size <= cached.info.data.size) {
			const auto result_id = merged_id ? merged_id : cached_id;
			const auto result    = m_slot_images.try_get(result_id);
			ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-compatible-candidate", cached_id,
			    cached, {.other_id = result_id, .other = result, .request = &requested});
			return {result != nullptr && ImageViewOps::FormatsCompatible(result->info.pixel_format,
			                                                             requested.pixel_format)
			            ? result_id
			            : ImageId {}};
		}
		EXIT("TextureCache: unresolvable equal-address image overlap, address=0x%016" PRIx64
		     " requested=%ux%u "
		     "cached=%ux%u requested_size=0x%016" PRIx64 " cached_size=0x%016" PRIx64
		     " type=%u/%u tile=%u/%u\n",
		     requested.data.address, requested.resources.levels, requested.resources.layers,
		     cached.info.resources.levels, cached.info.resources.layers, requested.data.size,
		     cached.info.data.size, static_cast<uint32_t>(requested.type),
		     static_cast<uint32_t>(cached.info.type), static_cast<uint32_t>(requested.tile_mode),
		     static_cast<uint32_t>(cached.info.tile_mode));
	}

	const int32_t requested_mip = requested.MipOf(cached.info);
	if (requested_mip >= 0) {
		const int32_t layer = requested.SliceOf(cached.info, requested_mip);
		ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-use-mip", cached_id, cached,
		    {.request = &requested, .value0 = static_cast<uint32_t>(requested_mip),
		     .value1 = static_cast<uint32_t>(layer)});
		return {cached_id, requested_mip, layer};
	}

	const int32_t mip = cached.info.MipOf(requested);
	if (mip >= 0) {
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (!merged_id) {
			ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-expand-contained", cached_id,
			    cached, {.request = &requested});
			return {ExpandImage(requested, cached_id)};
		}
		cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
		ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-merge-mip", cached_id, cached,
		    {.other_id = merged_id, .other = &m_slot_images[merged_id], .request = &requested,
		     .value0 = static_cast<uint32_t>(mip), .value1 = static_cast<uint32_t>(layer)});
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		FreeImage(cached_id);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-unmatched-aged-free", cached_id,
		    cached, {.request = &requested});
		FreeImage(cached_id);
	} else {
		ZeroCompanyImageLifecycle::Record(m_scheduler, "overlap-unmatched-retain", cached_id,
		    cached, {.request = &requested, .value0 = safe_to_delete});
	}
	return {merged_id};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	ZeroCompanyImageLifecycle::Record(m_scheduler, "expand-begin", source_id,
	    m_slot_images[source_id], {.request = &info});
	RefreshCopySource(source_id);
	const auto expanded_id = InsertImage(info);
	auto&      source      = m_slot_images[source_id];
	if (source.binding.is_bound || source.binding.is_target) {
		source.binding.needs_rebind = true;
	}
	InitializeImage(expanded_id);
	const int32_t mip = source.info.MipOf(info);
	const int32_t layer = source.info.SliceOf(info, mip);
	if (layer >= 0) {
		CopyImageMip(expanded_id, source_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
	} else {
		CopyImage(expanded_id, source_id);
	}
	FreeImage(source_id);
	ZeroCompanyImageLifecycle::Record(m_scheduler, "expand-returned", expanded_id,
	    m_slot_images[expanded_id], {.other_id = source_id});
	return expanded_id;
}

struct TextureCache::TextureTransfer {
	TextureUploadLayout              layout;
	std::vector<vk::BufferImageCopy> regions;
	std::vector<GpuTileInfo>         tiles;
	ColorTransform                   color_transform = ColorTransform::None;
	bool                             valid           = false;

	[[nodiscard]] uint64_t LinearSize() const {
		uint64_t size = 0;
		for (const auto& tile: tiles) {
			size = std::max(size, tile.linear_offset + tile.linear_size);
		}
		return size;
	}
};

struct TextureCache::ImageDownload {
	TextureTransfer texture;
	bool                depth_target = false;
	bool                valid        = false;
};

TextureCache::TextureTransfer
TextureCache::BuildTextureTransfer(const Image& image, BindingType binding,
                                    TransferDirection direction) const {
	const auto& info             = image.info;
	const bool  upload           = direction == TransferDirection::Upload;
	const bool  render_target    = binding == BindingType::RenderTarget;
	const bool  video_out        = binding == BindingType::VideoOut;
	uint32_t    layers           = info.TransferLayers();
	bool        volume           = info.IsVolume();
	const char* owner            = "TextureCache readback";

	TextureTransfer transfer;
	transfer.color_transform = info.GetColorTransform();
	if (upload) {
		if ((render_target || video_out) &&
		    (info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
		     info.samples != 1 || image.backing.samples != 1)) {
			EXIT("TextureCache: invalid color-attachment upload\n");
		}
		owner = "TextureCache";
		if (render_target) {
			owner = "RenderTarget";
		} else if (binding == BindingType::Storage) {
			owner = "StorageTextureCache";
		} else if (video_out) {
			if (info.metadata.compression != VideoOutCompression::Uncompressed) {
				EXIT("TextureCache: invalid color-attachment upload\n");
			}
			layers = info.resources.layers;
			volume = false;
			owner  = "VideoOut";
		}
	}

	transfer.layout = TextureCalcUploadLayout(info.guest_format, info.extent.width,
	                                         info.extent.height, info.resources.levels, layers,
	                                         info.tile_mode, info.data.size, volume, owner);
	transfer.regions = TextureBuildImageCopies(transfer.layout);
	if (info.IsDepth()) {
		for (auto& region: transfer.regions) {
			region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
		}
	}
	if (transfer.layout.surface.description.tile_mode != Prospero::TileMode::kLinear) {
		if (!TextureBuildGpuTileInfos(info.data.size, transfer.regions, transfer.layout,
		                              info.resources.levels, transfer.tiles)) {
			return transfer;
		}
	}
	transfer.valid = true;
	return transfer;
}

TextureCache::ImageDownload TextureCache::BuildDownload(const Image& image) const {
	const auto&  info    = image.info;
	const auto   binding = UploadBinding(image);
	ImageDownload transfer {.depth_target = binding == BindingType::DepthTarget};
	if (info.samples != 1 || image.backing.samples != 1) {
		return transfer;
	}
	if (transfer.depth_target) {
		transfer.valid = IsSupportedDepthPlaneReadback(info) && info.resources.layers != 0 &&
		             info.data.size % info.resources.layers == 0 &&
		             Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
		return transfer;
	}
	if (info.metadata.compression != VideoOutCompression::Uncompressed) {
		return transfer;
	}
	transfer.texture = BuildTextureTransfer(image, binding, TransferDirection::Download);
	transfer.valid   = transfer.texture.valid;
	return transfer;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	GpuTimestampOrigin ts_origin(GpuTimestamps::Origin::Refresh);
	ZeroCompanyImageLifecycle::Record(m_scheduler, "upload-begin", {}, image,
	    {.address = source.CpuAddress(), .size = source.Size(),
	     .value0 = GpuDiagnostics::HandleValue(source.Handle()), .value1 = source_offset});
	auto& destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto binding = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	const auto  upload  = [&](std::vector<vk::BufferImageCopy>& copies, TileManager::Result linear) {
		for (auto& copy: copies) {
			copy.bufferOffset += linear.offset;
		}
		destination.Upload(copies, linear.buffer, linear.offset, linear.size);
		ZeroCompanyImageLifecycle::Record(m_scheduler, "upload-copy-returned", {}, image,
		    {.other = &destination, .size = linear.size,
		     .value0 = GpuDiagnostics::HandleValue(linear.buffer), .value1 = linear.offset});
	};

	if (binding != BindingType::DepthTarget) {
		const auto& info = image.info;
		auto transfer = BuildTextureTransfer(image, binding, TransferDirection::Upload);
		if (!transfer.valid) {
			EXIT("TextureCache: invalid texture upload: binding=%u addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 " format=%u tile=%u family=%u extent=%ux%ux%u "
			     "pitch=%u levels=%u layers=%u samples=%u\n",
			     static_cast<uint32_t>(binding), info.data.address, info.data.size,
			     static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode),
			     static_cast<uint32_t>(transfer.layout.surface.texture.block.family), info.extent.width,
			     info.extent.height, info.extent.depth, info.pitch, info.resources.levels,
			     info.resources.layers, info.samples);
		}
		TileManager::Result linear {source.Handle(), source_offset, info.data.size};
		if (!transfer.tiles.empty()) {
			linear = m_tiler.Detile(source.Handle(), source_offset, info.data.size,
			                        transfer.LinearSize(), transfer.tiles, transfer.color_transform);
		} else if (transfer.color_transform != ColorTransform::None) {
			linear = m_tiler.TransformColor(linear, transfer.color_transform, true);
		}
		upload(transfer.regions, linear);
		return;
	}

	// The stencil plane has its own row pitch and shares the native image
	// with the depth plane.
	auto info = destination.info;
	if (image.depth_id) {
		info.data            = image.info.data;
		info.resources       = {image.stencil_subresources.level_count,
		                        image.stencil_subresources.layer_count};
		info.guest_format    = Prospero::BufferFormat::k8UInt;
		info.bytes_per_block = 1;
		if (info.IsTiled()) info.pitch = TileGetDepthPitch(info.extent.width, 1, 0);
	}
	if (info.samples != 1 || destination.backing.samples != 1 ||
	    info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto          layers          = info.resources.layers;
	const auto          full_slice_size = info.data.size / layers;
	auto copies = BuildDepthCopies(info, full_slice_size, image.depth_id
	                                                        ? vk::ImageAspectFlagBits::eStencil
	                                                        : vk::ImageAspectFlagBits::eDepth);
	if (image.depth_id) {
		for (auto& copy: copies) {
			copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
		}
	}
	TileManager::Result linear {source.Handle(), source_offset, source.Size() - source_offset};
	if (info.IsTiled()) {
		const auto tiles = BuildDepthTiles(info);
		linear =
		    m_tiler.Detile(source.Handle(), source_offset, info.data.size, info.data.size, tiles);
	}
	const auto transfer_bytes = image.depth_id ? 1u : DepthAspectTransferBytes(info.pixel_format);
	if (transfer_bytes != info.bytes_per_block) {
		const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
		EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
		                     transfer_bytes != sizeof(uint32_t) || texels_per_slice > UINT32_MAX ||
		                     texels_per_slice > UINT64_MAX / transfer_bytes);
		const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
		EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
		auto promoted = m_tiler.GetScratchBuffer(transfer_slice * layers, linear.buffer);
		m_tiler.ConvertD16(
		    linear, promoted, TileManager::D16Direction::Promote,
		    info.pixel_format == vk::Format::eD32SfloatS8Uint,
		    {.width               = info.extent.width,
		     .height              = info.extent.height,
		     .layers              = layers,
		     .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
		     .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
		     .source_slice_stride = full_slice_size,
		     .target_slice_stride = transfer_slice});
		linear = promoted;
		for (uint32_t layer = 0; layer < layers; layer++) {
			copies[layer].bufferOffset = transfer_slice * layer;
		}
	}
	upload(copies, linear);
}

void TextureCache::InitializeImage(ImageId id) {
	auto& image = m_slot_images[id];
	ZeroCompanyImageLifecycle::Record(m_scheduler, "initialize-begin", id, image);
	if (image.info.data.Empty()) {
		return;
	}
	TrackImage(id);
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (image.IsCpuDirty()) {
			image.RefreshComplete();
		}
		return;
	}
	if (image.info.samples > 1) {
		return;
	}
	const bool upload = (image.IsBufferModified() && !image.IsGpuModified()) || image.IsCpuDirty();
	if (upload) {
		ZeroCompanyImageLifecycle::Record(m_scheduler, "initialize-needs-upload", id, image);
		// Rendering-defect forensics (2026-10-13): the dossier portrait and the briefing panel show
		// noise/stripes that look like a CPU-produced (video-decoded) texture uploaded with the wrong
		// layout. Report images the CPU keeps rewriting, once per signature at 30 and 300 uploads.
		{
			static std::unordered_map<uint64_t, uint32_t> uploads; // GPU thread only
			const auto& info = image.info;
			const uint64_t key = (static_cast<uint64_t>(info.pixel_format) << 48u) ^
			                     (uint64_t {info.extent.width} << 32u) ^
			                     (uint64_t {info.extent.height} << 16u) ^
			                     (uint64_t {static_cast<uint8_t>(info.tile_mode)} << 8u) ^
			                     info.resources.levels ^ (info.data.size << 4u);
			const auto count = ++uploads[key];
			// An upload over GPU-owned pixels replaces GPU results with guest memory that tiled
			// targets never receive (stale or zero): a candidate mechanism for the black meshes.
			static uint64_t over_gpu = 0;
			const bool      gpu      = image.IsGpuModified();
			if (gpu) ++over_gpu;
			if (::Log::TraceEnabled() && (count == 1 || count == 30 || count == 300 || (gpu && count % 100 == 0))) {
				LOGF("ImageCpuRefresh: uploads=%u gpu_owned=%d buffer_modified=%d cpu_dirty=%d "
				     "maybe_dirty=%d over_gpu_total=%" PRIu64 " %s pitch=%u guest_format=%u\n",
				     count, gpu ? 1 : 0, image.IsBufferModified() ? 1 : 0,
				     image.IsDefinitelyCpuDirty() ? 1 : 0, image.IsMaybeCpuDirty() ? 1 : 0, over_gpu,
				     DescribeImageInfo(info).c_str(), info.pitch,
				     static_cast<unsigned>(info.guest_format));
			}
		}
		const auto [source, source_offset] =
		    m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		UploadImage(image, *source, source_offset);
		image.ClearBufferModified();
	} else if (image.IsBufferModified()) {
		image.ClearBufferModified();
	}
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
	ZeroCompanyImageLifecycle::Record(m_scheduler, "initialize-returned", id, image);
}

void TextureCache::MaterializeColorClear(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	    desc.info.metadata.kind != ImageMetadataKind::Cmask) {
		return;
	}
	const auto range = desc.info.metadata.range;
	{
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		ZeroCompanyImageLifecycle::Record(m_scheduler, "materialize-color-clear-check", id,
		    image, {.request = &desc.info, .address = range.address, .size = range.size});
		image.info.metadata = desc.info.metadata;
		// Native color metadata must not retain a reused HTile/CMask/FMask clear flag.
		m_surface_metas.erase(range.address);
		if (range.size == 0 || desc.info.resources.levels != 1) {
			return;
		}
	}
	const auto layers = desc.info.TransferLayers();
	// These one-mip surfaces use complete 4 KiB color metadata blocks.
	constexpr uint64_t MetadataBlockSize = 0x1000;
	if (!range.Valid() || range.address % MetadataBlockSize != 0 || layers == 0 ||
	    range.size % layers != 0 || (range.size / layers) % MetadataBlockSize != 0) {
		EXIT("TextureCache: color metadata slices must contain aligned 4 KiB blocks\n");
	}
	const auto& view           = desc.view_info;
	const bool  volume_texture = desc.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto  first          = volume_texture ? 0u : metadata_base_layer;
	const auto  image_first    = volume_texture ? 0u : view.base_layer;
	const auto  count          = volume_texture ? desc.info.extent.depth : view.layer_count;
	if (first >= layers || count > layers - first) {
		EXIT("TextureCache: color view exceeds its native metadata slices\n");
	}
	// Finish native metadata writes before reading backing bytes. This can submit the scheduler,
	// so discovery runs before final draw uploads and never holds the texture lock across it.
	static const bool clear_log = [] {
		const char* value = std::getenv("KYTY_PROBE_READBACK_LOG");
		return value != nullptr && value[0] == '1';
	}();
	const auto slice_size = range.size / layers;
	const bool gpu_dirty = m_buffer_cache.IsRegionGpuModified(range.address, range.size);
	if (gpu_dirty && ColorClearResolver::Enabled() && (slice_size & 3u) == 0 &&
	    slice_size <= UINT32_MAX && desc.view_info.level_count == 1 && count <= 2048) {
		// Volume textures use a 2D-array view over their depth slices (images are created
		// 2D-array compatible), which also clears only the detected slices.
		// GPU path: the metadata was written by the GPU; check and clear on the GPU timeline
		// instead of draining the pipeline for a readback (see ColorClearResolver).
		ColorClearResolver::Request request {};
		for (uint32_t code_slot = 0; code_slot < ColorClearResolver::CodeSlots; ++code_slot) {
			vk::ClearColorValue color {};
			if (DecodeColorClear(desc, static_cast<uint8_t>(code_slot << 5u), color)) {
				request.valid_mask |= 1u << code_slot;
				request.colors[code_slot] = color;
			}
		}
		if (request.valid_mask == 0) {
			return; // no code could decode to a clear for this view: nothing to detect
		}
		const auto meta_address = range.address + slice_size * first;
		const auto meta_size    = slice_size * count;
		{
			std::scoped_lock lock {m_lock};
			const auto it = m_resolved_dcc_meta.find(meta_address);
			if (it != m_resolved_dcc_meta.end() && it->second.second == m_dcc_meta_epoch) {
				return; // Already resolved and expanded on GPU; no intervening guest write
			}
		}
		auto [meta_buffer, meta_offset] =
		    m_buffer_cache.ObtainBuffer(meta_address, meta_size, /*is_written=*/true, false);
		if (m_color_clear_resolver == nullptr) {
			m_color_clear_resolver = std::make_unique<ColorClearResolver>(m_graphics, m_scheduler);
		}
		std::scoped_lock lock {m_lock};
		auto& image   = m_slot_images[id];
		auto& command = m_scheduler.Current();
		TrackImage(id);
		if (image.IsBufferModified() || image.IsCpuDirty()) {
			InitializeImage(id); // the resolve may clear only some slices: upload guest data first
		}
		command.EndRendering();
		ImageViewInfo attachment_view {};
		attachment_view.format      = view.format;
		attachment_view.type        = vk::ImageViewType::e2DArray;
		attachment_view.base_level  = view.base_level;
		attachment_view.base_layer  = image_first;
		attachment_view.layer_count = count;
		attachment_view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
		              vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
		              {}, command.Handle());
		request.meta_buffer = meta_buffer->Handle();
		request.meta_offset = meta_offset;
		request.slice_bytes = static_cast<uint32_t>(slice_size);
		request.slice_count = count;
		request.attachment  = image.FindView(attachment_view);
		request.extent      = ViewExtent(image, view.base_level, view.format);
		m_color_clear_resolver->Resolve(command, request);
		CommitGpuWrite(image);
		// Protected by m_lock acquired at line 1411 (held across image lookup, resolve, and commit).
		// Do NOT add an inner scoped_lock here: TrackingSpinLock is non-recursive and will abort.
		m_resolved_dcc_meta[meta_address] = {meta_size, m_dcc_meta_epoch};
		ZeroCompanyImageLifecycle::Record(m_scheduler, "materialize-color-clear-gpu", id, image,
		    {.request = &desc.info, .address = meta_address, .size = meta_size});
		if (clear_log) {
			static uint64_t gpu_resolves = 0;
			if (gpu_resolves++ < 200 || gpu_resolves % 1000 == 0) {
				LOGF("ColorClearCheck: gpu-resolve image data=0x%016" PRIx64 " meta=0x%016" PRIx64
				     "+0x%" PRIx64 " slices=%u mask=0x%02x\n",
				     desc.info.data.address, meta_address, meta_size, count, request.valid_mask);
			}
		}
		return;
	}
	if (gpu_dirty) {
		if (clear_log) {
			LOGF("ColorClearCheck: readback image data=0x%016" PRIx64 " meta=0x%016" PRIx64
			     "+0x%" PRIx64 " kind=%d type=%d volume=%d levels=%u slice=0x%" PRIx64 " count=%u\n",
			     desc.info.data.address, range.address, range.size,
			     static_cast<int>(desc.info.metadata.kind), static_cast<int>(desc.type),
			     volume_texture ? 1 : 0, desc.view_info.level_count, slice_size, count);
		}
		m_buffer_cache.ReadMemory(range.address, range.size, false, WaitReasonProbe::TextureReadback);
	}
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = range.address + slice_size * (first + slice);
		uint8_t code = 0;
		if (!LibKernel::Memory::TryReadBacking(address, &code, sizeof(code))) {
			EXIT("TextureCache: failed to read color metadata backing\n");
		}
		vk::ClearValue clear {};
		if (!DecodeColorClear(desc, code, clear.color)) {
			continue;
		}
		std::vector<uint8_t> bytes(slice_size);
		if (!LibKernel::Memory::TryReadBacking(address, bytes.data(), bytes.size())) {
			EXIT("TextureCache: failed to read color metadata slice\n");
		}
		if (!std::all_of(bytes.begin(), bytes.end(), [code](uint8_t byte) { return byte == code; })) {
			continue;
		}
		if (clear_log) {
			LOGF("ColorClearCheck: CLEAR detected image data=0x%016" PRIx64 " slice=%u code=0x%02x\n",
			     desc.info.data.address, slice, code);
		}
		{
			std::scoped_lock lock {m_lock};
			ClearImage(m_scheduler.Current(), id, view.format,
			           {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
			            image_first + slice, 1}, clear);
		}
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data. Invalidate the buffer before updating its backing.
		if (desc.type != BindingType::VideoOut) {
			std::fill(bytes.begin(), bytes.end(), uint8_t {0xff});
			m_buffer_cache.InvalidateMemory(address, slice_size);
			LibKernel::Memory::WriteBacking(address, bytes.data(), bytes.size());
		}
	}
}

void TextureCache::RefreshImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
		return;
	}
	TrackImage(id);
	if (image.IsMaybeCpuDirty()) {
		ZeroCompanyImageLifecycle::Record(m_scheduler, "refresh-maybe-dirty", id, image);
		const auto hash = image.HashGuestEdges();
		if (image.NeedsMaybeCpuHash()) {
			image.SetMaybeCpuHash(hash);
			return;
		}
		(void)image.ResolveMaybeCpuHash(hash);
	}
	bool cpu_dirty = image.IsBufferModified() || image.IsDefinitelyCpuDirty();
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (cpu_dirty) {
			EXIT("TextureCache: compressed guest image refresh is unsupported\n");
		}
		return;
	}
	if (!cpu_dirty) {
		return;
	}
	ZeroCompanyImageLifecycle::Record(m_scheduler, "refresh-needs-upload", id, image);
	InitializeImage(id);
}

ImageId TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	ImageId association {};
	for (const auto id: FindImagesInRegion(stencil.address, stencil.size, false)) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->info.data == stencil &&
		    owner->info.extent == depth.info.extent) {
			association = id;
		}
	}
	if (!association) {
		ImageInfo info {};
		info.data   = stencil;
		info.extent = depth.info.extent;
		association = InsertImage(info);
	}
	auto& record = m_slot_images[association];
	TouchImage(record);
	if (record.depth_id != depth_id) {
		if (record.depth_id) {
			std::erase(m_slot_images[record.depth_id].stencil_ids, association);
		}
		depth.stencil_ids.push_back(association);
	}
	record.depth_id             = depth_id;
	record.stencil_subresources = depth.stencil_subresources;
	return association;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	ValidateImageDesc(desc);
	if (desc.info.data.Empty()) {
		std::scoped_lock lock {m_lock};
		return GetNullImage(desc);
	}
	const auto metadata_base_layer = desc.view_info.base_layer;

	ImageId result {};
	{
		std::scoped_lock lock {m_lock};
		const auto       candidates =
		    FindImagesInRegion(desc.info.data.address, desc.info.data.size, false);

		for (const auto id: candidates) {
			const auto& image = m_slot_images[id];
			if (SameBacking(image.info, desc.info, exact_format)) {
				ZeroCompanyImageLifecycle::Record(m_scheduler, "find-same-backing", id, image,
				    {.request = &desc.info, .value0 = static_cast<uint32_t>(desc.type)});
				result = id;
			}
		}

		int32_t view_mip   = -1;
		int32_t view_layer = -1;
		if (!result) {
			for (const auto candidate: candidates) {
				view_mip                = -1;
				view_layer              = -1;
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const auto  overlap     = ResolveOverlap(merged_info, desc.type, candidate, result);
				if (overlap.image) {
					result     = overlap.image;
					view_mip   = overlap.mip;
					view_layer = overlap.layer;
				}
			}
		}

		if (result) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				ZeroCompanyImageLifecycle::Record(m_scheduler, "find-reject-exact-format", result,
				    resolved, {.request = &desc.info});
				result = {};
			} else if (resolved.info.resources < desc.info.resources) {
				ZeroCompanyImageLifecycle::Record(m_scheduler, "find-free-small-resources", result,
				    resolved, {.request = &desc.info});
				FreeImage(result);
				result = {};
			}
		}
		if (!result) {
			for (const auto candidate: candidates) {
				const auto* owner = m_slot_images.try_get(candidate);
				if (owner != nullptr && owner->registered && owner->IsGpuModified()) {
					const auto now = m_scheduler.CurrentTick();
					ReportGpuOwnedOverlap("Split", desc.type, desc.info, owner->info,
					                      now - std::min(now, owner->tick_accessed_last));
				}
			}
			result         = InsertImage(desc.info);
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
			                                    inserted.info.data.size)) {
				inserted.MarkBufferModified();
			}
			ZeroCompanyImageLifecycle::Record(m_scheduler, "find-created", result, inserted,
			    {.request = &desc.info, .value0 = static_cast<uint32_t>(desc.type)});
		}
		auto& image = m_slot_images[result];
		if (view_mip >= 0) {
			desc.view_info.base_level = static_cast<uint32_t>(view_mip);
		}
		if (view_layer >= 0) {
			desc.view_info.base_layer = static_cast<uint32_t>(view_layer);
		}
		image.tick_accessed_last = m_scheduler.CurrentTick();
		TouchImage(image);
	}
	MaterializeColorClear(result, desc, metadata_base_layer);
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
		std::scoped_lock lock {m_lock};
		const auto& image = m_slot_images[result];
		const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
		const bool native_current =
		    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
		if (!native_current) {
			EXIT("TextureCache: compressed video-out read requires clean native GPU "
			     "contents\n");
		}
	}
	return result;
}

void TextureCache::UpdateImage(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	ZeroCompanyImageLifecycle::Record(m_scheduler, "update-begin", id, image);
	TouchImage(image);
	RefreshImage(id);
	ZeroCompanyImageLifecycle::Record(m_scheduler, "update-returned", id, image);
}

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid) {
	if (!GuestRange {address, size}.Valid()) {
		return {};
	}
	std::scoped_lock lock {m_lock};
	ImageIds         matches;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr || owner->info.data.address != address) {
			continue;
		}
		if (ensure_valid && owner->depth_id) {
			owner = m_slot_images.try_get(owner->depth_id);
		}
		if (owner == nullptr || (ensure_valid && !owner->SafeToDownload())) {
			continue;
		}
		matches.push_back(id);
	}
	ImageId selected {};
	if (matches.size() == 1) {
		selected = matches.front();
	} else {
		for (const auto id: matches) {
			const auto& image = m_slot_images[id];
			if (image.info.data.size == size) {
				selected = id;
				break;
			}
		}
	}
	if (selected && ensure_valid) {
		const auto owner = m_slot_images.try_get(selected);
		if (owner != nullptr && owner->depth_id) {
			selected = owner->depth_id;
		}
	}
	return selected;
}

vk::ImageView TextureCache::FindTexture(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	if (!image.info.data.Empty()) {
		if (!image.registered || image.depth_id || image.binding.needs_rebind) {
			EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
		}
	}
	if (desc.type == BindingType::Storage) {
		image.MarkGpuModified();
	}
	if (!image.info.data.Empty()) {
		RefreshImage(id);
		if (image.info.HasStencil() &&
		    desc.info.data.address >= image.info.stencil.address &&
		    desc.info.data.End() <= image.info.stencil.End()) {
			for (const auto stencil_id:
			     FindImagesInRegion(image.info.stencil.address, image.info.stencil.size, false)) {
				const auto* stencil = m_slot_images.try_get(stencil_id);
				if (stencil != nullptr && stencil->depth_id == id &&
				    stencil->info.data == image.info.stencil) {
					RefreshImage(stencil_id);
					break;
				}
			}
		}
	}
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			if (!image.info.data.Empty()) {
				CommitGpuWrite(image);
			}
			TrackImageDownload(id, image);
			EnrollCpuReadback(id, image, /*require_gpu_modified=*/true);
			break;
		default: EXIT("TextureCache: invalid texture binding\n");
	}
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindRenderTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::RenderTarget) {
		EXIT("TextureCache: invalid color-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: color target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	image.usage.render_target = true;
	RefreshImage(id);
	CommitGpuWrite(image);
	TrackImageDownload(id, image);
	// CPU-readback enrolment of linear colour targets happens in PrepareGraphicsBindings
	// (EnrollCpuReadbackTarget): here the draw has already captured its native buffer handles.
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindDepthTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::DepthTarget) {
		EXIT("TextureCache: invalid depth-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: depth target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	image.usage.depth_target = true;
	if (desc.info.HasStencil()) {
		const auto layers = image.info.resources.layers;
		EXIT_IF(layers == 0 || image.info.data.size % layers != 0 ||
		        desc.info.data.address < image.info.data.address);
		const auto slice_size = image.info.data.size / layers;
		const auto offset     = desc.info.data.address - image.info.data.address;
		EXIT_IF(slice_size == 0 || offset % slice_size != 0 || offset / slice_size >= layers ||
		        desc.info.resources.layers > layers - offset / slice_size);
		image.stencil_subresources = {0, desc.info.resources.levels,
		                             static_cast<uint32_t>(offset / slice_size),
		                             desc.info.resources.layers};
	}
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		m_surface_metas.emplace(desc.info.metadata.range.address,
		                        MetaDataInfo {.type       = MetaDataInfo::Type::HTile,
		                                      .clear_mask = image.info.htile_clear_mask});
	}
	RefreshImage(id);
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		RefreshImage(AssociateStencil(id, desc.info.stencil));
	}
	return image.FindView(desc.view_info);
}

void TextureCache::MarkGpuWritten(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id) {
		EXIT("TextureCache: cannot mark an unavailable image GPU-written\n");
	}
	TrackImage(id);
	CommitGpuWrite(image);
	if (image.info.HasStencil()) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
}

void TextureCache::CommitGpuWrite(Image& image) {
	if (!image.depth_id && image.backing.image == nullptr) {
		EXIT("TextureCache: GPU writes require a native image or stencil association\n");
	}
	image.ClearBufferModified();
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
	image.MarkGpuModified();
	ZeroCompanyImageLifecycle::Record(m_scheduler, "commit-gpu-write", {}, image);
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear, bool decline_if_aliased) {
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock      lock {m_lock};
	ImageId               selected {};
	ImageId               stencil_id {};
	vk::ImageAspectFlags  aspect {};
	ImageSubresourceRange subresources;
	bool                  aliased = false; // another registered image overlaps the range
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		ImageSubresourceRange candidate_subresources {
		    0, owner->info.resources.levels, 0, owner->info.TransferLayers()};
		if (owner->depth_id && owner->info.data.address == address &&
		    owner->info.data.size == size) {
			candidate              = vk::ImageAspectFlagBits::eStencil;
			candidate_id           = owner->depth_id;
			candidate_subresources = owner->stencil_subresources;
			owner        = m_slot_images.try_get(candidate_id);
			if (owner == nullptr || owner->backing.image == nullptr || !owner->info.HasStencil()) {
				aliased = true;
				continue;
			}
		} else if (!owner->depth_id && owner->info.data.address == address &&
		           owner->info.data.size == size) {
			candidate = owner->info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
			                                  : vk::ImageAspectFlagBits::eColor;
		}
		if (!candidate) {
			aliased = true; // a larger, smaller or offset image shares these bytes (review 2026-10-13)
			continue;
		}
		if (selected && (selected != candidate_id || subresources != candidate_subresources)) {
			return false;
		}
		selected     = candidate_id;
		stencil_id   = candidate == vk::ImageAspectFlagBits::eStencil ? id : ImageId {};
		aspect       = candidate;
		subresources = candidate_subresources;
	}
	if (!selected || (decline_if_aliased && aliased)) {
		return false; // declined before any state change; the caller takes its ordinary path
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		if (!DecodeColorDwordFill(image.info.pixel_format, packed_clear, clear.color)) {
			return false;
		}
	} else {
		uint8_t stencil_clear = 0;
		if ((aspect == vk::ImageAspectFlagBits::eDepth &&
		     !DecodePackedDepthClear(image.info.pixel_format, packed_clear, clear.depthStencil.depth)) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil &&
		     !DecodePackedStencilClear(packed_clear, stencil_clear))) {
			return false;
		}
		clear.depthStencil.stencil = stencil_clear;
	}
	ClearImage(command, selected, image.backing.format,
	           {aspect, subresources.base_level, subresources.level_count,
	            subresources.base_layer, subresources.layer_count}, clear);
	if (stencil_id) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
	return true;
}

void TextureCache::ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
                              const vk::ImageSubresourceRange& range, const vk::ClearValue& clear) {
	auto& image = m_slot_images[id];
	ZeroCompanyImageLifecycle::Record(m_scheduler, "clear-begin", id, image,
	    {.value0 = static_cast<uint32_t>(format)});
	const auto aspects = image.info.IsDepth() ? ImageViewOps::DepthAspectMask(image.backing.format)
	                                          : vk::ImageAspectFlagBits::eColor;
	EXIT_IF(range.baseMipLevel >= image.info.resources.levels);
	const auto layers = image.info.IsVolume()
	                        ? std::max(image.info.extent.depth >> range.baseMipLevel, 1u)
	                        : image.backing.layers;
	EXIT_IF(command.IsInvalid() || image.depth_id || !range.aspectMask || range.levelCount == 0 ||
	        range.levelCount > image.info.resources.levels - range.baseMipLevel ||
	        range.layerCount == 0 || range.baseArrayLayer >= layers ||
	        range.layerCount > layers - range.baseArrayLayer ||
	        (range.aspectMask & aspects) != range.aspectMask);
	const bool full_subresources = range.baseMipLevel == 0 &&
	                               range.levelCount == image.info.resources.levels &&
	                               range.baseArrayLayer == 0 && range.layerCount == layers;
	const bool full_image = range.aspectMask == aspects && full_subresources;
	TrackImage(id);
	if (!full_image && (image.IsBufferModified() || image.IsCpuDirty())) {
		InitializeImage(id);
		if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
			EXIT("TextureCache: image clear retained guest ownership\n");
		}
	}
	if (image.info.HasStencil() && (range.aspectMask & vk::ImageAspectFlagBits::eStencil)) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		if (!full_subresources) {
			RefreshImage(stencil_id);
		} else {
			TrackImage(stencil_id);
			CommitGpuWrite(m_slot_images[stencil_id]);
		}
	}
	command.EndRendering();
	// Transfer clears use the backing format; aliased clears must encode through their view.
	if (format != image.backing.format || (image.info.IsVolume() && !full_image)) {
		EXIT_NOT_IMPLEMENTED(range.aspectMask != vk::ImageAspectFlagBits::eColor ||
		                     range.levelCount != 1);
		ImageViewInfo view {};
		view.format = format;
		view.type   = range.layerCount == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
		view.base_level  = range.baseMipLevel;
		view.base_layer  = range.baseArrayLayer;
		view.layer_count = range.layerCount;
		view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
		              vk::AccessFlagBits2::eColorAttachmentWrite, {}, command.Handle());
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = image.FindView(view);
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp      = vk::AttachmentLoadOp::eClear;
		attachment.storeOp     = vk::AttachmentStoreOp::eStore;
		attachment.clearValue  = clear;
		vk::RenderingInfo rendering {};
		// In the view's texel units: an uncompressed view of a block image is blocks wide, and a
		// render area in image texels drew past it (NVIDIA Xid 13 "WIDTH CT Violation").
		rendering.renderArea.extent    = ViewExtent(image, range.baseMipLevel, format);
		rendering.layerCount           = range.layerCount;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &attachment;
		{
			GpuTimestampScope clear_scope(command.Handle(), GpuTimestamps::Kind::Clear);
			command.Handle().beginRendering(&rendering);
			command.Handle().endRendering();
		}
		CommitGpuWrite(image);
		return;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Handle());
	auto native_range = range;
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		{ GpuTimestampScope ts_scope(command.Handle(), GpuTimestamps::Kind::Clear); command.Handle().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                                 &clear.color, 1, &native_range); }
	} else {
		{ GpuTimestampScope ts_scope(command.Handle(), GpuTimestamps::Kind::Clear); command.Handle().clearDepthStencilImage(image.backing.image,
		                                        vk::ImageLayout::eTransferDstOptimal,
		                                        &clear.depthStencil, 1, &native_range); }
	}
	CommitGpuWrite(image);
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	std::scoped_lock lock {m_lock};
	InvalidateCpuAliases(address, size);
}

void TextureCache::InvalidateCpuWrittenPages(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid written-page range\n");
	}
	std::scoped_lock lock {m_lock};
	InvalidateCpuAliases(address, size, /*whole_pages=*/true);
}

void TextureCache::DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset) {
	const auto&    info             = image.info;
	const auto     layers           = info.resources.layers;
	const auto     full_slice_size  = info.data.size / layers;
	const auto     transfer_bytes   = DepthAspectTransferBytes(info.pixel_format);
	const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
	EXIT_NOT_IMPLEMENTED(transfer_bytes == 0 || texels_per_slice > UINT32_MAX ||
	                     texels_per_slice > UINT64_MAX / transfer_bytes ||
	                     texels_per_slice > UINT64_MAX / info.bytes_per_block);
	const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
	const uint64_t guest_slice    = texels_per_slice * info.bytes_per_block;
	EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
	const uint64_t transfer_size = transfer_slice * layers;
	EXIT_NOT_IMPLEMENTED(guest_slice > full_slice_size);
	auto copies = BuildDepthCopies(info, full_slice_size, vk::ImageAspectFlagBits::eDepth);
	if (transfer_bytes == info.bytes_per_block) {
		if (!info.IsTiled()) {
			for (auto& copy: copies) {
				copy.bufferOffset += destination_offset;
			}
			image.Download(copies, destination.Handle(), destination_offset, info.data.size);
			return;
		}
		const auto tiles = BuildDepthTiles(info);
		m_tiler.TileImage(image, copies, destination.Handle(), destination_offset, info.data.size,
		                  info.data.size, tiles);
		return;
	}
	EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
	                     transfer_bytes != sizeof(uint32_t));
	for (uint32_t layer = 0; layer < layers; layer++) {
		copies[layer].bufferOffset = transfer_slice * layer;
	}
	auto host_linear = m_tiler.GetScratchBuffer(transfer_size);
	image.Download(copies, host_linear.buffer, 0, host_linear.size);
	const bool tiled        = info.IsTiled();
	auto       guest_linear = tiled ? m_tiler.GetScratchBuffer(info.data.size, host_linear.buffer)
	                                : TileManager::Result {destination.Handle(), destination_offset,
	                                                       destination.Size() - destination_offset};
	m_tiler.ConvertD16(host_linear, guest_linear, TileManager::D16Direction::Demote,
	                   DepthAspectTransferFormat(info.pixel_format) == vk::Format::eD32Sfloat,
	                   {.width               = info.extent.width,
	                    .height              = info.extent.height,
	                    .layers              = layers,
	                    .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
	                    .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
	                    .source_slice_stride = transfer_slice,
	                    .target_slice_stride = full_slice_size});
	if (!tiled) {
		return;
	}
	const auto tiles = BuildDepthTiles(info);
	m_tiler.Tile(guest_linear.buffer, guest_linear.offset, info.data.size, destination.Handle(),
	             destination_offset, info.data.size, tiles);
}

// 2026-10-13: a GPU buffer write that covers only part of a GPU-owned image hands the image's memory
// back to the buffer cache (ClearGpuModified + MarkBufferModified) and the next bind re-uploads the
// WHOLE image from that memory. Tiled targets never publish their pixels there, so every byte outside
// the written range came back stale (pinned by the regression "exact-base buffer write refreshes only
// the written texel"). Seed the host buffer that backs the image with the image's current pixels so
// the later refresh merges GPU pixels with the new bytes. `excluded` lists ranges whose bytes already
// hold newer data (writes recorded before this call); an empty list seeds the whole image, which is
// exact even when the caller's range only bounds the real write (storage descriptors). Everything is
// recorded on the GPU timeline ahead of a pending write; no CPU wait.
bool TextureCache::PreserveGpuPixels(ImageId id, Image& image,
                                     std::span<const std::pair<uint64_t, uint64_t>> excluded) {
	GpuTimestampOrigin ts_origin(GpuTimestamps::Origin::Preserve);
	const auto& info = image.info;
	if (image.depth_id || info.samples != 1 || info.data.Empty() || !image.IsGpuModified() ||
	    image.IsBufferModified() || image.IsDefinitelyCpuDirty()) {
		return false;
	}
	if (image.IsMaybeCpuDirty()) {
		// Same resolution as RefreshImage: a changed guest edge means the CPU wrote newer bytes, and
		// those must win; an unchanged edge clears the suspicion.
		TrackImage(id);
		const auto hash = image.HashGuestEdges();
		if (image.NeedsMaybeCpuHash()) {
			image.SetMaybeCpuHash(hash);
			return false;
		}
		(void)image.ResolveMaybeCpuHash(hash);
		if (image.IsDefinitelyCpuDirty()) {
			return false;
		}
	}
	auto transfer = BuildDownload(image);
	if (!transfer.valid) {
		return false;
	}
	const uint64_t base = info.data.address;
	const uint64_t size = info.data.size;
	// Complement of the excluded ranges inside the image, as (offset, length) runs.
	std::vector<std::pair<uint64_t, uint64_t>> holes;
	for (const auto& [address, length]: excluded) {
		const uint64_t lo = std::clamp(address, base, base + size) - base;
		const uint64_t hi = std::clamp(address + length, base, base + size) - base;
		if (hi > lo) {
			holes.emplace_back(lo, hi);
		}
	}
	std::sort(holes.begin(), holes.end());
	std::vector<vk::BufferCopy> runs;
	uint64_t                    cursor = 0;
	for (const auto& [lo, hi]: holes) {
		if (lo > cursor) {
			runs.emplace_back(cursor, cursor, lo - cursor);
		}
		cursor = std::max(cursor, hi);
	}
	if (cursor < size) {
		runs.emplace_back(cursor, cursor, size - cursor);
	}
	if (runs.empty()) {
		return true; // every byte is being rewritten: nothing to preserve
	}
	const bool whole = runs.size() == 1 && runs.front().size == size;
	if (!whole && transfer.depth_target) {
		return false; // depth downloads are whole-image only
	}
	auto [buffer, offset] = m_buffer_cache.ObtainBuffer(base, size, /*is_written=*/true,
	                                                    /*is_texel_buffer=*/false);
	if (buffer == nullptr) {
		return false;
	}
	auto& command = m_scheduler.Current();
	command.EndRendering();
	if (whole) {
		DownloadImage(image, *buffer, offset, size, std::move(transfer));
	} else {
		auto& texture = transfer.texture;
		auto  scratch = m_tiler.GetScratchBuffer(size);
		if (texture.tiles.empty()) {
			if (texture.color_transform != ColorTransform::None) {
				auto linear = m_tiler.GetScratchBuffer(size, scratch.buffer);
				image.Download(texture.regions, linear.buffer, linear.offset, linear.size);
				m_tiler.TransformColor(linear, scratch, texture.color_transform, false);
			} else {
				auto regions = texture.regions;
				for (auto& copy: regions) {
					copy.bufferOffset += scratch.offset;
				}
				image.Download(regions, scratch.buffer, scratch.offset, size);
			}
		} else {
			m_tiler.TileImage(image, texture.regions, scratch.buffer, scratch.offset, size,
			                  texture.LinearSize(), texture.tiles, texture.color_transform);
		}
		// Tile preparation can wrap its parameter stream and replace the native command buffer.
		const auto native = m_scheduler.Current().Handle();
		std::array<vk::BufferMemoryBarrier, 2> before {};
		before[0].srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eShaderWrite |
		                          vk::AccessFlagBits::eTransferWrite;
		before[0].dstAccessMask       = vk::AccessFlagBits::eTransferRead;
		before[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before[0].buffer              = scratch.buffer;
		before[0].offset              = scratch.offset;
		before[0].size                = size;
		before[1]                     = before[0];
		before[1].srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                          vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		before[1].dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		before[1].buffer        = buffer->Handle();
		before[1].offset        = offset;
		{ GpuTimestampScope ts_scope(native, GpuTimestamps::Kind::BarrierBuffer); native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
		                       0, nullptr, 2, before.data(), 0, nullptr); }
		for (auto& run: runs) {
			run.srcOffset += scratch.offset;
			run.dstOffset += offset;
		}
		{ GpuTimestampScope ts_scope(native, GpuTimestamps::Kind::CopyUploadTexture); if (ts_scope.Active()) ts_scope.SetBytes(SumCopySizes(runs)); native.copyBuffer(scratch.buffer, buffer->Handle(), static_cast<uint32_t>(runs.size()),
		                  runs.data()); }
	}
	vk::BufferMemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eShaderWrite |
	                      vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer              = buffer->Handle();
	after.offset              = offset;
	after.size                = size;
	{ GpuTimestampScope ts_scope(m_scheduler.Current().Handle(), GpuTimestamps::Kind::BarrierBuffer); m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &after, 0, nullptr); }
	const auto tick             = m_scheduler.CurrentTick();
	buffer->last_gpu_write_tick = tick;
	m_buffer_cache.NoteGpuWrite(base, size, tick);
	ZeroCompanyImageLifecycle::Record(m_scheduler, "preserve-gpu-pixels", id, image,
	    {.address = base, .size = size, .value0 = whole ? 1u : 0u, .value1 = runs.size()});
	return true;
}

void TextureCache::DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
                                     uint64_t destination_size, ImageDownload transfer) {
	GpuTimestampOrigin ts_origin(GpuTimestamps::Origin::Download);
	ZeroCompanyImageLifecycle::Record(m_scheduler, "download-begin", {}, image,
	    {.address = destination.CpuAddress(), .size = destination_size,
	     .value0 = GpuDiagnostics::HandleValue(destination.Handle()), .value1 = destination_offset});
	if (!transfer.valid) {
		EXIT("TextureCache: invalid image download transfer\n");
	}
	if (transfer.depth_target) {
		if (destination_size != image.info.data.size) {
			EXIT("TextureCache: partial depth image download is unsupported\n");
		}
		DownloadDepth(image, destination, destination_offset);
		return;
	}

	auto&      texture   = transfer.texture;
	const auto transform = texture.color_transform;
	if (texture.tiles.empty()) {
		if (transform != ColorTransform::None) {
			auto linear = m_tiler.GetScratchBuffer(destination_size);
			image.Download(texture.regions, linear.buffer, 0, linear.size);
			m_tiler.TransformColor(linear,
			                       {destination.Handle(), destination_offset, destination_size},
			                       transform, false);
			return;
		}
		for (auto& copy: texture.regions) {
			copy.bufferOffset += destination_offset;
		}
		image.Download(texture.regions, destination.Handle(), destination_offset, destination_size);
		return;
	}

	m_tiler.TileImage(image, texture.regions, destination.Handle(), destination_offset,
	                  destination_size, texture.LinearSize(), texture.tiles, transform);
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size);
	if (!selected) {
		return false;
	}

	std::scoped_lock lock {m_texture_cache.m_lock};
	auto& image = m_texture_cache.m_slot_images[selected];
	// The GPU thread owns image retirement; CPU invalidation can dirty this image after lookup.
	if (!image.SafeToDownload()) {
		return false;
	}
	if (!buffer.IsInBounds(image.info.data.address, 1)) {
		return false;
	}
	const auto buf_offset = buffer.Offset(image.info.data.address);
	const auto available  = buffer.Size() - buf_offset;
	uint32_t   levels     = 0;
	uint64_t   copy_size  = 0;
	if (buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
	} else if (image.info.IsVolume() || image.info.resources.layers > 1) {
		// Array and volume slices are strided across the full mip chain.
		return false;
	} else {
		for (; levels < image.info.resources.levels; ++levels) {
			const auto& mip = image.info.mip_layout[levels];
			if (mip.size == 0 || mip.offset > available || mip.size > available - mip.offset) {
				break;
			}
			copy_size = std::max(copy_size, mip.offset + mip.size);
		}
	}
	if (copy_size == 0) {
		return false;
	}
	auto transfer = m_texture_cache.BuildDownload(image);
	if (!transfer.valid) {
		return false;
	}
	if (transfer.depth_target && copy_size != image.info.data.size) {
		return false;
	}
	if (!transfer.depth_target && levels < image.info.resources.levels) {
		auto& texture = transfer.texture;
		std::erase_if(texture.regions, [levels](const vk::BufferImageCopy& region) {
			return region.imageSubresource.mipLevel >= levels;
		});
		if (texture.regions.empty()) {
			return false;
		}
		if (!texture.tiles.empty()) {
			texture.tiles.clear();
			if (!TextureBuildGpuTileInfos(copy_size, texture.regions, texture.layout, levels,
			                              texture.tiles)) {
				return false;
			}
		}
	}
	buffer.last_gpu_write_tick = m_scheduler.CurrentTick(); // the image->buffer copy recorded below
	m_texture_cache.DownloadImage(image, buffer, buf_offset, copy_size, std::move(transfer));
	return true;
}

bool TextureCache::DownloadImageMemory(ImageId id) {
	auto& image = m_slot_images[id];
	ZeroCompanyImageLifecycle::Record(m_scheduler, "download-memory-begin", id, image);
	if (image.depth_id) {
		return false;
	}
	auto transfer = BuildDownload(image);
	if (!transfer.valid || !image.SafeToDownload()) {
		ZeroCompanyImageLifecycle::Record(m_scheduler, "download-memory-rejected", id, image,
		    {.value0 = transfer.valid});
		return false;
	}
	const auto range    = image.info.data;
	auto&      download = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	auto [mapped, offset] =
	    download.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
	if (mapped == nullptr) {
		EXIT("TextureCache: failed to map reusable download buffer\n");
	}
	download.Commit();
	if (!LibKernel::Memory::TryReadBacking(range.address, mapped, range.size)) {
		return false;
	}
	download.Flush(offset, range.size);

	DownloadImage(image, download, offset, range.size, std::move(transfer));
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download.Handle();
	barrier.offset              = offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	{ GpuTimestampScope ts_scope(m_scheduler.Current().Handle(), GpuTimestamps::Kind::BarrierBuffer); m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr); }
	m_scheduler.DeferPriorityOperation([&download, range, mapped, offset] {
		download.Invalidate(offset, range.size);
		LibKernel::Memory::WriteBacking(range.address, mapped, range.size);
	});
	ZeroCompanyImageLifecycle::Record(m_scheduler, "download-cpu-publish-queued", id, image,
	    {.address = range.address, .size = range.size, .value0 = offset});
	return true;
}

void TextureCache::EnrollCpuReadbackTarget(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr) {
		return;
	}
	// The draw's colour write makes the image GPU-owned later, in FindRenderTarget; the enrolment's
	// buffer join must happen now, before the draw captures native buffer handles.
	EnrollCpuReadback(id, *image, /*require_gpu_modified=*/false);
}

void TextureCache::EnrollCpuReadback(ImageId id, Image& image, bool require_gpu_modified) {
	// The host fault that publishes the image needs the GPU thread (BufferCache::ReadMemory uses
	// SendCommandSync); contexts without one (unit tests) must not read-protect image pages.
	if (!ImageReadbackEnabled() || !m_scheduler.Context().HasGpu() || !image.registered ||
	    image.depth_id || image.info.IsDepth() || image.info.IsTiled() || image.info.data.Empty() ||
	    (require_gpu_modified && !image.IsGpuModified())) {
		return;
	}
	const auto range = image.info.data;
	if (!GuestRange {range.address, range.size}.Valid()) {
		return;
	}
	// Once per command stream, unless a host access already published (and unprotected) the range.
	// Tracker-only query: BufferCache::IsRegionGpuModified flushes write faults into this cache's lock.
	const auto tick = m_scheduler.CurrentTick();
	if (image.cpu_readback_pending && image.cpu_readback_tick == tick &&
	    m_buffer_cache.IsLabelRegionGpuModified(range.address, range.size)) {
		return;
	}
	image.cpu_readback_tick    = tick;
	image.cpu_readback_pending = true;
	// A written binding marks the range GPU-modified in the memory tracker, which read-protects its
	// pages; the buffer itself is only seeded when a host read actually faults.
	(void)m_buffer_cache.ObtainBuffer(range.address, range.size, /*is_written=*/true,
	                                  /*is_texel_buffer=*/false);
	static std::atomic<uint64_t> enrolled {0};
	const auto count = enrolled.fetch_add(1, std::memory_order_relaxed) + 1;
	if (count <= 8 || count % 1000 == 0) {
		LOGF("ImageReadback: enrolled #%" PRIu64 " %s\n", count, DescribeImageInfo(image.info).c_str());
	}
	ZeroCompanyImageLifecycle::Record(m_scheduler, "cpu-readback-enroll", id, image);
}

bool TextureCache::SyncGpuImagesForHostRead(Buffer& buffer, uint64_t& address, uint64_t& size,
                                            bool record) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	std::scoped_lock    lock {m_lock};
	uint64_t             begin = address;
	uint64_t             end   = address + size;
	std::vector<ImageId> selected;
	// Fixpoint over the page-rounded hull: the publish unprotects whole pages, and a page at the
	// hull's edge may hold the head or tail of another enrolled image, which must be synced too
	// (review 2026-10-14, C2). Only images EnrollCpuReadback enrolled: elsewhere a GPU-modified image
	// over a GPU-dirty buffer range is not proof that the image is newer (depth/stencil planes).
	for (;;) {
		const auto page_begin = Common::AlignDown(begin, TRACKER_PAGE_SIZE);
		const auto page_end   = Common::AlignUp(end, TRACKER_PAGE_SIZE);
		bool       grew       = false;
		for (const auto id: FindImagesInRegion(page_begin, page_end - page_begin, true)) {
			if (std::find(selected.begin(), selected.end(), id) != selected.end()) {
				continue;
			}
			auto* image = m_slot_images.try_get(id);
			if (image == nullptr || !image->registered || !image->cpu_readback_pending ||
			    image->depth_id || !image->SafeToDownload() ||
			    !image->Overlaps(page_begin, page_end - page_begin)) {
				continue;
			}
			const auto range = image->info.data;
			if (range.Empty()) {
				continue;
			}
			if (!buffer.IsInBounds(range.address, range.size)) {
				static std::atomic<uint64_t> outside {0};
				if (outside.fetch_add(1, std::memory_order_relaxed) < 8) {
					LOGF("ImageReadback: host read at 0x%" PRIx64 " hits %s outside its buffer; stale bytes\n",
					     address, DescribeImageInfo(image->info).c_str());
				}
				continue;
			}
			selected.push_back(id);
			begin = std::min(begin, range.address);
			end   = std::max(end, range.address + range.size);
			grew  = true;
		}
		if (!grew) {
			break;
		}
	}
	if (selected.empty()) {
		return false;
	}
	if (record) {
		for (const auto id: selected) {
			auto&      image = m_slot_images[id];
			const auto range = image.info.data;
			auto       transfer = BuildDownload(image);
			// The publish below unprotects the pages either way; an image that cannot be downloaded
			// keeps the old (stale) behaviour instead of faulting forever.
			image.cpu_readback_pending = false; // the next GPU write enrols again
			if (!transfer.valid) {
				LOGF("ImageReadback: cannot download %s for a host read; stale bytes\n",
				     DescribeImageInfo(image.info).c_str());
				continue;
			}
			m_scheduler.EndRendering();
			buffer.last_gpu_write_tick = m_scheduler.CurrentTick(); // the image->buffer copy below
			DownloadImage(image, buffer, buffer.Offset(range.address), range.size, std::move(transfer));
			static std::atomic<uint64_t> count {0};
			const auto n = count.fetch_add(1, std::memory_order_relaxed) + 1;
			if (n <= 8 || n % 100 == 0) {
				LOGF("ImageReadback: synced #%" PRIu64 " %s for a host read at 0x%" PRIx64 "\n", n,
				     DescribeImageInfo(image.info).c_str(), address);
			}
			ZeroCompanyImageLifecycle::Record(m_scheduler, "cpu-readback-sync", id, image,
			    {.address = address, .size = size});
		}
	}
	address = begin;
	size    = end - begin;
	return true;
}

bool TextureCache::HasGpuOwnedImageIn(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto* owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->IsGpuModified() && owner->Overlaps(address, size)) {
			return true;
		}
	}
	return false;
}

// Registered images whose bytes overlap [address, address + size), as (tick_accessed_last, id).
// The scan visits every image slot; the page-table query visits only the 1 MiB pages of the range.
// Both return the same set: RegisterImage adds an image to every page of info.data (and
// UnregisterImage removes it, EXITing on a missing membership), the page query filters by the same
// exact-byte Image::Overlaps, and only registered images are ever in the page table.
// Texture-cache lock held.
std::vector<std::pair<uint64_t, ImageId>> TextureCache::CollectWriteOverlaps(
    uint64_t address, uint64_t size, bool use_page_table) const {
	std::vector<std::pair<uint64_t, ImageId>> result;
	if (use_page_table) {
		for (const auto id: FindImagesInRegion(address, size, false)) {
			result.emplace_back(m_slot_images[id].tick_accessed_last, id);
		}
	} else {
		m_slot_images.ForEach([&](ImageId id, const Image& image) {
			if (image.registered && image.Overlaps(address, size)) {
				result.emplace_back(image.tick_accessed_last, id);
			}
		});
	}
	return result;
}

bool TextureCache::WriteOverlapPageQueryFits(uint64_t address, uint64_t size) {
	// A page walk costs about one table lookup per 1 MiB; keep huge unbounded writes on the scan.
	constexpr uint64_t MaxPages = 256;
	ImagePageTable::PageRange pages {};
	return ImagePageTable::TryGetPageRange(address, size, pages) &&
	       pages.last_exclusive - pages.first <= MaxPages;
}

// Caller holds m_lock (called from InvalidateMemoryFromGPU and InvalidateCpuAliases).
void TextureCache::InvalidateDccMetadata(uint64_t address, uint64_t size) {
	if (m_resolved_dcc_meta.empty()) {
		return;
	}
	const uint64_t end = address + size;
	for (auto it = m_resolved_dcc_meta.begin(); it != m_resolved_dcc_meta.end(); ) {
		const uint64_t meta_start = it->first;
		const uint64_t meta_end   = meta_start + it->second.first;
		if (address < meta_end && end > meta_start) {
			it = m_resolved_dcc_meta.erase(it);
		} else {
			++it;
		}
	}
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size, bool write_pending,
                                           std::span<const std::pair<uint64_t, uint64_t>> recorded,
                                           GpuWriteCoverage coverage) {
	if (!GuestRange {address, size}.Valid()) {
		return;
	}
	std::scoped_lock lock {m_lock};
	InvalidateDccMetadata(address, size);
	// Both exact DMA and conservative shader writes can start inside an image at any size.
	// The threshold chooses the lookup strategy, not whether an overlap is invalidated.
	constexpr uint64_t SmallWrite = 64 * 1024;
	static const bool  merge      = [] {
		const char* value = std::getenv("KYTY_MERGE_PARTIAL_WRITES");
		return value == nullptr || value[0] != '0'; // default on; =0 restores the lossy refresh
	}();
	const bool overlap = merge;
	// Oldest image first, so when several aliases share the bytes the most recently used one seeds
	// the buffer last and wins. This is not a per-byte authority proof among image aliases.
	std::vector<std::pair<uint64_t, ImageId>> ordered;
	if (overlap && size > SmallWrite) {
		// Bounded by cached image slots (scan) or by the 1 MiB pages of the range (page table).
		ordered = CollectWriteOverlaps(address, size, WriteOverlapPageQueryFits(address, size));
	} else {
		for (const auto id: FindImagesInRegion(address, overlap ? size : 1, true)) {
			ordered.emplace_back(m_slot_images[id].tick_accessed_last, id);
		}
	}
	std::sort(ordered.begin(), ordered.end());
	// Bytes that already hold the new data when the write was recorded before this call.
	std::vector<std::pair<uint64_t, uint64_t>> excluded;
	if (!write_pending) {
		excluded.emplace_back(address, size);
		excluded.insert(excluded.end(), recorded.begin(), recorded.end());
	}
	for (const auto& [age, id]: ordered) {
		auto& image = m_slot_images[id];
		if (overlap ? !image.Overlaps(address, size) : image.info.data.address != address) {
			continue;
		}
		const bool partial = address > image.info.data.address ||
		                     address + size < image.info.data.address + image.info.data.size;
		ZeroCompanyImageLifecycle::Record(m_scheduler, "invalidate-gpu-before", id, image,
		    {.address = address, .size = size});
		if (image.IsGpuModified() && KeepGpuImagesProbe() != 0 && partial) {
			static uint64_t kept = 0;
			if (++kept == 1 || kept % 1000 == 0) {
				LOGF("KeepGpuImages: ignored partial GPU write 0x%" PRIx64 " bytes at the base of %s "
				     "(kept=%" PRIu64 ")\n",
				     size, DescribeImageInfo(image.info).c_str(), kept);
			}
			continue;
		}
		if (image.IsGpuModified()) {
			// Rendering-defect forensics (2026-10-13): a GPU buffer write starting at this image's
			// base hands authority back to guest memory; the image's own GPU pixels are then
			// replaced by the buffer contents at the next bind (ng-58: scene targets re-uploaded
			// hundreds of times). Report once per signature with the writer's size.
			static std::unordered_map<uint64_t, uint32_t> hits; // texture-cache lock held
			const auto&    info = image.info;
			const uint64_t key  = (static_cast<uint64_t>(info.pixel_format) << 48u) ^
			                     (uint64_t {info.extent.width} << 32u) ^
			                     (uint64_t {info.extent.height} << 16u) ^
			                     (uint64_t {static_cast<uint8_t>(info.tile_mode)} << 8u) ^
			                     (info.data.size << 4u);
			const auto count = ++hits[key];
			if (::Log::TraceEnabled() && (count == 1 || count == 30 || count == 300)) {
				LOGF("ImageGpuInvalidatedByBuffer: hits=%u write_size=0x%" PRIx64 " %s\n", count,
				     size, DescribeImageInfo(info).c_str());
			}
			// A descriptor bounds possible writes; even a full-image binding may write only
			// one DWORD. Seed its untouched pixels before the shader executes. Only exact
			// full overwrites can discard the old image without preservation.
			if (merge && (partial || coverage == GpuWriteCoverage::Conservative)) {
				static uint64_t merged = 0;
				static uint64_t failed = 0;
				if (PreserveGpuPixels(id, image, excluded)) {
					if (::Log::TraceEnabled() && (++merged == 1 || merged % 1000 == 0)) {
						LOGF("PartialWriteMerge: preserved GPU pixels around 0x%" PRIx64 " bytes in %s "
						     "(merged=%" PRIu64 " unmergeable=%" PRIu64 ")\n",
						     size, DescribeImageInfo(info).c_str(), merged, failed);
					}
				} else if (::Log::TraceEnabled() && (++failed == 1 || failed % 1000 == 0)) {
					LOGF("PartialWriteMerge: could not preserve %s (unmergeable=%" PRIu64 ")\n",
					     DescribeImageInfo(info).c_str(), failed);
				}
			}
			image.ClearGpuModified();
		}
		image.MarkBufferModified();
		ZeroCompanyImageLifecycle::Record(m_scheduler, "invalidate-gpu-after", id, image,
		    {.address = address, .size = size});
	}
}

void TextureCache::InvalidateCpuAliases(uint64_t address, uint64_t size, bool whole_pages) {
	InvalidateDccMetadata(address, size);
	const auto page_begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		// The bytes this write certainly changed in the image: the faulting byte, or with
		// whole_pages the written pages the image covers completely.
		uint64_t write_address = address;
		uint64_t write_size    = size;
		if (whole_pages) {
			const auto first = std::max(address, Common::AlignUp(owner->info.data.address, TRACKER_PAGE_SIZE));
			const auto last  = std::min(address + size, Common::AlignDown(owner->info.data.End(), TRACKER_PAGE_SIZE));
			write_address = first;
			write_size    = first < last ? last - first : 0;
		}
		if (write_size != 0 && owner->Overlaps(write_address, write_size)) {
			if (owner->IsGpuModified() && KeepGpuImagesProbe() == 1) {
				static uint64_t kept = 0;
				if (++kept == 1 || kept % 1000 == 0) {
					LOGF("KeepGpuImages: ignored CPU write at 0x%" PRIx64 " inside %s (kept=%" PRIu64 ")\n",
					     write_address, DescribeImageInfo(owner->info).c_str(), kept);
				}
				// The pages must still be unprotected or the guest re-faults on the same store forever
				// (ng-60 froze at t=14 s with 16 M repeated faults).
				UntrackImage(id);
				continue;
			}
			const auto flags_before = ZeroCompanyImageLifecycle::Flags(*owner);
			owner->InvalidateCpuWrite(write_address, write_size);
			UntrackImage(id);
			if (flags_before != ZeroCompanyImageLifecycle::Flags(*owner)) {
				ZeroCompanyImageLifecycle::Record(m_scheduler, "invalidate-cpu-changed", id, *owner,
				    {.address = write_address, .size = write_size, .value0 = flags_before});
			}
			continue;
		}
		const auto image_begin = owner->info.data.address;
		const auto image_end   = owner->info.data.End();
		if (page_end < image_end) {
			UntrackImageHead(id);
		} else if (image_begin < page_begin) {
			UntrackImageTail(id);
		} else {
			MarkAsMaybeDirty(id, *owner);
		}
	}
}

bool TextureCache::IsMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end();
}

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	return (found->second.clear_mask & (1u << slice)) != 0;
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask = UINT32_MAX;
	return true;
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	if (is_clear) {
		found->second.clear_mask |= 1u << slice;
	} else {
		found->second.clear_mask &= ~(1u << slice);
	}
	return true;
}

bool TextureCache::IsRegionRegistered(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return !FindImagesInRegion(address, size, false).empty();
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid unmap range\n");
	}
	std::scoped_lock lock {m_lock};
	for (auto metadata = m_surface_metas.begin(); metadata != m_surface_metas.end();) {
		const auto base = metadata->first;
		if (base >= address && base < address + size) {
			metadata = m_surface_metas.erase(metadata);
		} else {
			++metadata;
		}
	}
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		ZeroCompanyImageLifecycle::Record(m_scheduler, "unmap-free", id, *owner,
		    {.address = address, .size = size});
		FreeImage(id);
	}
}

std::string TextureCache::DescribeImagesIn(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	const auto ids = FindImagesInRegion(address, size, true);
	if (ids.empty()) {
		return "none";
	}
	const auto  tick = m_scheduler.CurrentTick();
	std::string text;
	size_t      count = 0;
	for (const auto id: ids) {
		if (count == 2) {
			text += fmt::format(" +{}img", ids.size() - count);
			break;
		}
		const auto& image = m_slot_images[id];
		text += fmt::format("{}img[0x{:x}+0x{:x} {}x{}x{} {} age={}]", count == 0 ? "" : " ",
		                    image.track_addr, image.track_addr_end - image.track_addr,
		                    image.info.extent.width, image.info.extent.height,
		                    image.info.extent.depth, vk::to_string(image.info.pixel_format),
		                    tick - image.tick_accessed_last);
		++count;
	}
	return text;
}

void TextureCache::RunGarbageCollector() {
	std::scoped_lock lock {m_lock};
	const uint64_t   tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	const auto collect = [&](bool allow_aggressive) {
		bool           pressured  = m_total_used_memory >= m_pressure_gc_memory;
		bool           aggressive = allow_aggressive && m_total_used_memory >= m_critical_gc_memory;
		const uint64_t age       = std::min<uint64_t>(aggressive ? 160 : pressured ? 80 : 16, tick);
		size_t         deletions = aggressive ? 40 : pressured ? 20 : 10;
		std::vector<ImageId> candidates;
		candidates.reserve(deletions);
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(tick - age, [&](ImageId id) {
			// Pending images must survive until publication, but must not exhaust the batch
			// before later obsolete images can be considered for retirement.
			const auto* owner = m_slot_images.try_get(id);
			if (owner != nullptr && owner->cpu_readback_pending) {
				return false;
			}
			candidates.push_back(id);
			return candidates.size() == deletions;
		});
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			--deletions;
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || owner->depth_id) {
				continue;
			}
			// Enrolled for a CPU read: its pages are read-protected on the buffer side and the buffer
			// holds no pixels yet. Freeing it here would make the next host fault publish stale bytes
			// over the pixels DownloadImageMemory wrote (review 2026-10-14, C1).
			if (owner->cpu_readback_pending) {
				continue;
			}
			if (owner->IsGpuModified()) {
				const bool safe = owner->SafeToDownload();
				if (safe && owner->info.IsTiled()) {
					continue;
				}
				if (safe && !pressured) {
					continue;
				}
				if (safe && !DownloadImageMemory(id)) {
					continue;
				}
			}
			ZeroCompanyImageLifecycle::Record(m_scheduler, "gc-free", id, *owner,
			    {.value0 = tick, .value1 = age});
			FreeImage(id);
			if (m_total_used_memory < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (m_total_used_memory < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false);
	if (m_total_used_memory >= m_critical_gc_memory) {
		collect(true);
	}
}

void TextureCache::ProcessDownloadImages() {
	std::scoped_lock lock {m_lock};
	for (const auto id: m_download_images) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->registered && owner->IsGpuModified()) {
			(void)DownloadImageMemory(id);
		}
	}
	m_download_images.clear();
}

} // namespace Libs::Graphics
