#include "graphics/host_gpu/vulkanCommon.h"

#include "common/assert.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/gpuDiagnostics.h"

#include <array>
#include <chrono>
#include <thread>
#include <mutex>
#include <cstring>
#include <atomic>
#include <vector>
#include <cstdio>
#include <cinttypes>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace Libs::Graphics {
namespace {

struct FormatMapping {
	Prospero::BufferFormat guest;
	vk::Format             host;
};

constexpr FormatMapping kFormatMappings[] = {
    {Prospero::BufferFormat::k8UNorm, vk::Format::eR8Unorm},
    {Prospero::BufferFormat::k8SNorm, vk::Format::eR8Snorm},
    {Prospero::BufferFormat::k8UScaled, vk::Format::eR8Uscaled},
    {Prospero::BufferFormat::k8SScaled, vk::Format::eR8Sscaled},
    {Prospero::BufferFormat::k8UInt, vk::Format::eR8Uint},
    {Prospero::BufferFormat::k8SInt, vk::Format::eR8Sint},
    {Prospero::BufferFormat::k16UNorm, vk::Format::eR16Unorm},
    {Prospero::BufferFormat::k16SNorm, vk::Format::eR16Snorm},
    {Prospero::BufferFormat::k16UScaled, vk::Format::eR16Uscaled},
    {Prospero::BufferFormat::k16SScaled, vk::Format::eR16Sscaled},
    {Prospero::BufferFormat::k16UInt, vk::Format::eR16Uint},
    {Prospero::BufferFormat::k16SInt, vk::Format::eR16Sint},
    {Prospero::BufferFormat::k16Float, vk::Format::eR16Sfloat},
    {Prospero::BufferFormat::k8_8UNorm, vk::Format::eR8G8Unorm},
    {Prospero::BufferFormat::k8_8SNorm, vk::Format::eR8G8Snorm},
    {Prospero::BufferFormat::k8_8UScaled, vk::Format::eR8G8Uscaled},
    {Prospero::BufferFormat::k8_8SScaled, vk::Format::eR8G8Sscaled},
    {Prospero::BufferFormat::k8_8UInt, vk::Format::eR8G8Uint},
    {Prospero::BufferFormat::k8_8SInt, vk::Format::eR8G8Sint},
    {Prospero::BufferFormat::k32UInt, vk::Format::eR32Uint},
    {Prospero::BufferFormat::k32SInt, vk::Format::eR32Sint},
    {Prospero::BufferFormat::k32Float, vk::Format::eR32Sfloat},
    {Prospero::BufferFormat::k16_16UNorm, vk::Format::eR16G16Unorm},
    {Prospero::BufferFormat::k16_16SNorm, vk::Format::eR16G16Snorm},
    {Prospero::BufferFormat::k16_16UScaled, vk::Format::eR16G16Uscaled},
    {Prospero::BufferFormat::k16_16SScaled, vk::Format::eR16G16Sscaled},
    {Prospero::BufferFormat::k16_16UInt, vk::Format::eR16G16Uint},
    {Prospero::BufferFormat::k16_16SInt, vk::Format::eR16G16Sint},
    {Prospero::BufferFormat::k16_16Float, vk::Format::eR16G16Sfloat},
    {Prospero::BufferFormat::k11_11_10Float, vk::Format::eB10G11R11UfloatPack32},
    {Prospero::BufferFormat::k10_10_10_2UNorm, vk::Format::eA2B10G10R10UnormPack32},
    {Prospero::BufferFormat::k10_10_10_2SNorm, vk::Format::eA2B10G10R10SnormPack32},
    {Prospero::BufferFormat::k10_10_10_2UScaled, vk::Format::eA2B10G10R10UscaledPack32},
    {Prospero::BufferFormat::k10_10_10_2UInt, vk::Format::eA2B10G10R10UintPack32},
    {Prospero::BufferFormat::k8_8_8_8UNorm, vk::Format::eR8G8B8A8Unorm},
    {Prospero::BufferFormat::k8_8_8_8SNorm, vk::Format::eR8G8B8A8Snorm},
    {Prospero::BufferFormat::k8_8_8_8UScaled, vk::Format::eR8G8B8A8Uscaled},
    {Prospero::BufferFormat::k8_8_8_8SScaled, vk::Format::eR8G8B8A8Sscaled},
    {Prospero::BufferFormat::k8_8_8_8UInt, vk::Format::eR8G8B8A8Uint},
    {Prospero::BufferFormat::k8_8_8_8SInt, vk::Format::eR8G8B8A8Sint},
    {Prospero::BufferFormat::k32_32UInt, vk::Format::eR32G32Uint},
    {Prospero::BufferFormat::k32_32SInt, vk::Format::eR32G32Sint},
    {Prospero::BufferFormat::k32_32Float, vk::Format::eR32G32Sfloat},
    {Prospero::BufferFormat::k16_16_16_16UNorm, vk::Format::eR16G16B16A16Unorm},
    {Prospero::BufferFormat::k16_16_16_16SNorm, vk::Format::eR16G16B16A16Snorm},
    {Prospero::BufferFormat::k16_16_16_16UScaled, vk::Format::eR16G16B16A16Uscaled},
    {Prospero::BufferFormat::k16_16_16_16SScaled, vk::Format::eR16G16B16A16Sscaled},
    {Prospero::BufferFormat::k16_16_16_16UInt, vk::Format::eR16G16B16A16Uint},
    {Prospero::BufferFormat::k16_16_16_16SInt, vk::Format::eR16G16B16A16Sint},
    {Prospero::BufferFormat::k16_16_16_16Float, vk::Format::eR16G16B16A16Sfloat},
    {Prospero::BufferFormat::k32_32_32UInt, vk::Format::eR32G32B32Uint},
    {Prospero::BufferFormat::k32_32_32SInt, vk::Format::eR32G32B32Sint},
    {Prospero::BufferFormat::k32_32_32Float, vk::Format::eR32G32B32Sfloat},
    {Prospero::BufferFormat::k32_32_32_32UInt, vk::Format::eR32G32B32A32Uint},
    {Prospero::BufferFormat::k32_32_32_32SInt, vk::Format::eR32G32B32A32Sint},
    {Prospero::BufferFormat::k32_32_32_32Float, vk::Format::eR32G32B32A32Sfloat},
    // Narrow-channel sRGB formats are optional in Vulkan. Keep a same-width fallback until
    // sampler-aware sRGB emulation is available.
    {Prospero::BufferFormat::k8Srgb, vk::Format::eR8Unorm},
    {Prospero::BufferFormat::k8_8Srgb, vk::Format::eR8G8Unorm},
    {Prospero::BufferFormat::k8_8_8_8Srgb, vk::Format::eR8G8B8A8Srgb},
    {Prospero::BufferFormat::k9_9_9_5Float, vk::Format::eE5B9G9R9UfloatPack32},
    {Prospero::BufferFormat::k5_6_5UNorm, vk::Format::eB5G6R5UnormPack16},
    {Prospero::BufferFormat::k5_5_5_1UNorm, vk::Format::eA1R5G5B5UnormPack16},
    {Prospero::BufferFormat::k1_5_5_5UNorm, vk::Format::eR5G5B5A1UnormPack16},
    {Prospero::BufferFormat::k4_4_4_4UNorm, vk::Format::eR4G4B4A4UnormPack16},
    {Prospero::BufferFormat::kBc1UNorm, vk::Format::eBc1RgbaUnormBlock},
    {Prospero::BufferFormat::kBc1Srgb, vk::Format::eBc1RgbaSrgbBlock},
    {Prospero::BufferFormat::kBc2UNorm, vk::Format::eBc2UnormBlock},
    {Prospero::BufferFormat::kBc2Srgb, vk::Format::eBc2SrgbBlock},
    {Prospero::BufferFormat::kBc3UNorm, vk::Format::eBc3UnormBlock},
    {Prospero::BufferFormat::kBc3Srgb, vk::Format::eBc3SrgbBlock},
    {Prospero::BufferFormat::kBc4UNorm, vk::Format::eBc4UnormBlock},
    {Prospero::BufferFormat::kBc4SNorm, vk::Format::eBc4SnormBlock},
    {Prospero::BufferFormat::kBc5UNorm, vk::Format::eBc5UnormBlock},
    {Prospero::BufferFormat::kBc5SNorm, vk::Format::eBc5SnormBlock},
    {Prospero::BufferFormat::kBc6UFloat, vk::Format::eBc6HUfloatBlock},
    {Prospero::BufferFormat::kBc6SFloat, vk::Format::eBc6HSfloatBlock},
    {Prospero::BufferFormat::kBc7UNorm, vk::Format::eBc7UnormBlock},
    {Prospero::BufferFormat::kBc7Srgb, vk::Format::eBc7SrgbBlock},
};

constexpr auto MakeFormatLookup() {
	constexpr auto kMaxFormat = static_cast<size_t>(Prospero::BufferFormat::kBc7Srgb);
	std::array<vk::Format, kMaxFormat + 1> lookup {};
	lookup.fill(vk::Format::eUndefined);
	for (const auto& mapping: kFormatMappings) {
		lookup[static_cast<size_t>(mapping.guest)] = mapping.host;
	}
	return lookup;
}

constexpr auto kFormatLookup = MakeFormatLookup();

} // namespace

vk::Format VulkanFormat(Prospero::BufferFormat guest_format) {
	const auto index = static_cast<size_t>(guest_format);
	return index < kFormatLookup.size() ? kFormatLookup[index] : vk::Format::eUndefined;
}

void RequireVulkanSuccess(vk::Result result, const char* operation) {
	if (result != vk::Result::eSuccess) {
		EXIT("%s failed: %s (%d)\n", operation, vk::to_string(result).c_str(),
		     static_cast<int>(result));
	}
}

vk::ShaderModule CompileSPV(std::span<const uint32_t> code, vk::Device device) {
	vk::ShaderModuleCreateInfo create_info {};
	create_info.codeSize    = code.size_bytes();
	create_info.pCode       = code.data();
	vk::ShaderModule module = nullptr;
	RequireVulkanSuccess(device.createShaderModule(&create_info, nullptr, &module),
	                     "create SPIR-V shader module");
	return module;
}

namespace {
struct AddressBindingRecord {
	uint64_t seq         = 0;
	uint64_t base        = 0;
	uint64_t size        = 0;
	uint64_t object      = 0;
	uint32_t object_type = 0;
	bool     bind        = false;
	bool     internal    = false;
	char     name[56]    = {};
};
constexpr size_t                           kAddressRingSize = 1u << 16u;
std::array<AddressBindingRecord, kAddressRingSize> g_address_ring {};
std::atomic<uint64_t>                      g_address_seq {0};
std::mutex                                 g_address_mutex;
} // namespace

void RecordAddressBinding(uint64_t base, uint64_t size, bool bind, bool internal,
                          uint32_t object_type, uint64_t object, const char* name) {
	std::lock_guard lock(g_address_mutex);
	const auto seq  = g_address_seq.fetch_add(1, std::memory_order_relaxed) + 1;
	auto&      slot = g_address_ring[seq % kAddressRingSize];
	slot            = {};
	slot.seq        = seq;
	slot.base       = base;
	slot.size       = size;
	slot.object     = object;
	slot.object_type = object_type;
	slot.bind       = bind;
	slot.internal   = internal;
	if (name != nullptr) {
		std::strncpy(slot.name, name, sizeof(slot.name) - 1);
	}
}

void DumpAddressBindingsNear(uint64_t address) {
	std::lock_guard lock(g_address_mutex);
	const auto total = g_address_seq.load(std::memory_order_relaxed);
	std::printf("  address bindings recorded: %" PRIu64 " (ring %zu); records touching 0x%016" PRIx64
	            " and neighbours within 64 MiB, newest first:\n", total, kAddressRingSize, address);
	uint32_t printed = 0;
	const uint64_t window = 64ull << 20u;
	for (uint64_t seq = total; seq > 0 && total - seq < kAddressRingSize && printed < 40; --seq) {
		const auto& r = g_address_ring[seq % kAddressRingSize];
		if (r.seq != seq) continue;
		const bool contains = address >= r.base && address - r.base < r.size;
		const bool near     = (r.base > address ? r.base - address : address - (r.base + r.size)) < window;
		if (!contains && !near) continue;
		std::printf("    %s seq=%" PRIu64 " %s [0x%016" PRIx64 ", +0x%" PRIx64 ") type=%s%s handle=0x%" PRIx64
		            " %s\n", contains ? "HIT " : "near", r.seq, r.bind ? "bind  " : "unbind",
		            r.base, r.size, vk::to_string(static_cast<vk::ObjectType>(r.object_type)).c_str(),
		            r.internal ? "(internal)" : "", r.object, r.name);
		printed++;
	}
	std::fflush(stdout);
}

void ReportQueueCheckpoints(vk::Queue queue) {
	if (queue == nullptr || VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV == nullptr) {
		return;
	}
	uint32_t count = 0;
	queue.getCheckpointDataNV(&count, nullptr);
	std::vector<vk::CheckpointDataNV> data(count);
	queue.getCheckpointDataNV(&count, data.data());
	std::printf("Vulkan NV checkpoints: %u%s\n", count,
	            GpuDiagnostics::Enabled() ? "" : " (markers are only set with KYTY_GPU_DIAGNOSTICS=1)");
	for (const auto& entry: data) {
		const auto marker = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(entry.pCheckpointMarker));
		std::printf("  checkpoint stage=%s marker=%" PRIu64 ": %s\n", vk::to_string(entry.stage).c_str(),
		            marker, GpuDiagnostics::Describe(marker).c_str());
	}
	std::fflush(stdout);
}

bool ReportDeviceFault(vk::Device device, vk::Queue queue) {
	if (device == nullptr || VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT == nullptr) {
		std::printf("Vulkan device fault: VK_EXT_device_fault not available\n");
		std::fflush(stdout);
		return false;
	}
	vk::DeviceFaultCountsEXT counts {};
	vk::Result               count_result = vk::Result::eErrorUnknown;
	// The NVIDIA vendor binary (crash dump) can appear a little after the loss is first visible.
	for (uint32_t attempt = 0; attempt < 6; ++attempt) {
		count_result = device.getFaultInfoEXT(&counts, nullptr);
		if (count_result == vk::Result::eSuccess && counts.vendorBinarySize != 0) break;
		if (count_result != vk::Result::eSuccess) break;
		std::printf("Vulkan device fault: attempt %u: addresses=%u vendor=%u binary=%" PRIu64 "; waiting 1 s\n",
		            attempt, counts.addressInfoCount, counts.vendorInfoCount,
		            static_cast<uint64_t>(counts.vendorBinarySize));
		std::fflush(stdout);
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	if (count_result != vk::Result::eSuccess) {
		std::printf("Vulkan device fault: vkGetDeviceFaultInfoEXT(counts) = %s\n",
		            vk::to_string(count_result).c_str());
		std::fflush(stdout);
		return false;
	}
	if (queue != nullptr) {
		ReportQueueCheckpoints(queue);
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	std::vector<uint8_t>                       binary(static_cast<size_t>(counts.vendorBinarySize));
	vk::DeviceFaultInfoEXT info {};
	info.pAddressInfos     = addresses.data();
	info.pVendorInfos      = vendors.data();
	info.pVendorBinaryData = binary.empty() ? nullptr : binary.data();
	const auto result  = device.getFaultInfoEXT(&counts, &info);
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		std::printf("Vulkan device fault: vkGetDeviceFaultInfoEXT = %s\n", vk::to_string(result).c_str());
		std::fflush(stdout);
		return false;
	}
	std::printf("Vulkan device fault: \"%s\" addresses=%u vendor=%u binary=%" PRIu64 "\n",
	            info.description.data(), counts.addressInfoCount, counts.vendorInfoCount,
	            static_cast<uint64_t>(counts.vendorBinarySize));
	if (!binary.empty()) {
		if (auto* file = std::fopen("_device_fault.nv-gpudmp", "wb"); file != nullptr) {
			std::fwrite(binary.data(), 1, binary.size(), file);
			std::fclose(file);
			std::printf("  vendor binary written to _device_fault.nv-gpudmp (cwd)\n");
		}
	}
	for (uint32_t i = 0; i < counts.addressInfoCount; ++i) {
		const auto& address = addresses[i];
		std::printf("  address fault: %s at 0x%016" PRIx64 " (precision 0x%" PRIx64 ")\n",
		            vk::to_string(address.addressType).c_str(),
		            static_cast<uint64_t>(address.reportedAddress),
		            static_cast<uint64_t>(address.addressPrecision));
		DumpAddressBindingsNear(static_cast<uint64_t>(address.reportedAddress));
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; ++i) {
		const auto& vendor = vendors[i];
		std::printf("  vendor fault: %s code=0x%" PRIx64 " data=0x%" PRIx64 "\n",
		            vendor.description.data(), static_cast<uint64_t>(vendor.vendorFaultCode),
		            static_cast<uint64_t>(vendor.vendorFaultData));
	}
	std::fflush(stdout);
	return counts.addressInfoCount + counts.vendorInfoCount > 0;
}

} // namespace Libs::Graphics
