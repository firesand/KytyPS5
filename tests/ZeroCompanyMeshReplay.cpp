// Standalone, deliberately narrow replay of one unchanged captured mesh module.
// No emulator, guest package, CPU guest memory, or original pixel images are used.
#include <vulkan/vulkan.h>
#include <spirv-tools/libspirv.hpp>
#include <nlohmann/json.hpp>
#include "zero_company_replay_frag.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
constexpr uint64_t MaxBytes = 768ull * 1024 * 1024;
constexpr uint64_t FnvBasis = 14695981039346656037ull;
constexpr uint64_t MeshFnv = 0xca310497219a1d7aull;
constexpr size_t MeshBytes = 158472;
constexpr const char* MeshSha = "0f0e5672764bbf39aa0ac59fca99eae3a042dbad7d86a958bb2aaf707b33cc79";

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void Vk(VkResult result, const char* operation) {
    Require(result == VK_SUCCESS, std::string(operation) + ": VkResult=" + std::to_string(result));
}
uint64_t Hash(const void* data, size_t size, uint64_t hash = FnvBasis) {
    for (size_t i = 0; i < size; ++i) hash = (hash ^ static_cast<const uint8_t*>(data)[i]) * 1099511628211ull;
    return hash;
}
std::string Hex(uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}
uint64_t Number(const Json& value, uint64_t limit, const char* label) {
    Require(value.is_number_unsigned(), std::string(label) + " must be an unsigned integer");
    auto n = value.get<uint64_t>();
    Require(n <= limit, std::string(label) + " exceeds bound");
    return n;
}
uint64_t StreamFile(const fs::path& file, uint64_t size, void* destination = nullptr) {
    Require(fs::is_regular_file(file) && fs::file_size(file) == size, "wrong file size: " + file.string());
    std::ifstream in(file, std::ios::binary);
    Require(bool(in), "cannot open " + file.string());
    std::array<char, 1024 * 1024> scratch {};
    uint64_t hash = FnvBasis;
    for (uint64_t offset = 0; offset < size;) {
        auto count = static_cast<size_t>(std::min<uint64_t>(scratch.size(), size - offset));
        auto* bytes = destination ? static_cast<char*>(destination) + offset : scratch.data();
        in.read(bytes, static_cast<std::streamsize>(count));
        Require(in.gcount() == static_cast<std::streamsize>(count), "short read: " + file.string());
        hash = Hash(bytes, count, hash);
        offset += count;
    }
    Require(in.peek() == std::char_traits<char>::eof(), "file grew during read: " + file.string());
    return hash;
}
void Write(const fs::path& file, const void* bytes, size_t size) {
    // A new output directory is mandatory. Failed writes retain only .partial.
    auto partial = file.string() + ".partial";
    auto* out = std::fopen(partial.c_str(), "wbx");
    Require(out != nullptr, "cannot exclusively create " + partial);
    bool ok = std::fwrite(bytes, 1, size, out) == size;
    ok = std::fflush(out) == 0 && ok;
    ok = std::fclose(out) == 0 && ok;
    Require(ok, "failed write: " + partial);
    Require(!fs::exists(file), "refusing existing output " + file.string());
    fs::rename(partial, file);
}
void WriteJson(const fs::path& file, const Json& value) {
    auto text = value.dump(2) + '\n';
    Write(file, text.data(), text.size());
}

struct InputChunk {
    fs::path path;
    uint64_t size = 0, hash = 0;
};
struct Capture {
    Json manifest;
    std::map<uint32_t, InputChunk> chunks;
    std::array<uint32_t, 9> descriptor_chunks {};
    std::array<uint32_t, 32> push {};
    std::array<uint32_t, 3> groups {};
    std::vector<uint32_t> mesh;
    uint64_t bytes = 0;
    uint32_t width = 0, height = 0;

    Capture(const fs::path& manifest_path, const fs::path& spv) {
        Require(std::endian::native == std::endian::little, "little endian host required");
        Require(fs::file_size(manifest_path) <= 1024 * 1024, "oversize manifest");
        std::ifstream in(manifest_path);
        Require(bool(in), "cannot open manifest");
        in >> manifest;
        Require(manifest.at("schema") == 1 && manifest.at("status") == "complete", "complete schema1 manifest required");
        Require(manifest.at("indexed") == false && manifest.at("sync") == true && manifest.at("sync_prep") == true,
                "only completed serialized nonindexed captures supported");
        Require(manifest.at("preparation_tick") == manifest.at("preparation_completed") &&
                manifest.at("copy_tick") == manifest.at("copy_completed"), "incomplete capture fences");
        auto draw = manifest.at("draw_data");
        Require(draw.size() == 6, "draw_data must have six words");
        for (size_t i = 0; i < 6; ++i) push[i] = Number(draw.at(i), UINT32_MAX, "draw_data");
        Require(push[0] >= 1 && push[0] <= 2 && push[1] == 0 && push[2] == 0 &&
                push[3] == 0 && push[4] == 0 && push[5] == 0, "unsupported draw or BDA path");
        auto g = manifest.at("mesh_groups");
        Require(g.size() == 3, "mesh_groups must have three words");
        for (size_t i = 0; i < 3; ++i) groups[i] = Number(g.at(i), 2, "group count");
        Require(groups[0] == push[0] && groups[1] == 1 && groups[2] == 1, "unsupported group shape");
        const auto& meta = manifest.at("mesh");
        Require(meta.at("host_subgroup_size") == 32 && meta.at("wave_size") == 64 &&
                meta.at("threads") == Json::array({256, 1, 1}) && meta.at("lds_dwords") == 1024 &&
                meta.at("max_vertices") == 256 && meta.at("max_primitives") == 254 &&
                meta.at("fast_launch") == false, "unexpected mesh configuration");
        Require(manifest.at("stages").size() == 2, "expected exactly two captured stages");
        const auto& ms = manifest.at("stages").at(0);
        const auto& ps = manifest.at("stages").at(1);
        Require(ms.at("stage") == 5 && ms.at("hash") == "0x67e610ccc37800ed" &&
                ms.at("push_start") == 8 && ms.at("memory_offset_dword") == 1 &&
                ms.at("shader_data").size() == 4 && ms.at("images_uncaptured").empty(), "unexpected mesh binding layout");
        Require(ps.at("stage") == 2 && ps.at("hash") == "0x548ab529f34a1307" &&
                ps.at("push_start") == 6 && ps.at("shader_data").size() == 2, "unexpected pixel push layout");
        for (const auto& stage : manifest.at("stages")) {
            auto start = Number(stage.at("push_start"), 31, "push start");
            const auto& words = stage.at("shader_data");
            Require(words.is_array() && words.size() <= 32 - start, "push words out of bounds");
            for (size_t i = 0; i < words.size(); ++i) push[start + i] = Number(words.at(i), UINT32_MAX, "push word");
        }
        Require(ms.at("user_data").size() > 8 && ms.at("user_data").at(8) == push[8], "mesh user data mismatch");
        const auto& buffers = ms.at("buffers");
        Require(buffers.is_array() && buffers.size() == 9, "expected nine mesh views");
        std::set<uint32_t> seen;
        for (const auto& resource : buffers) {
            auto index = uint32_t(Number(resource.at("descriptor"), 8, "descriptor index"));
            Require(seen.insert(index).second && resource.at("resource") == index && resource.at("binding") == 0 &&
                    resource.at("read") == true && resource.at("write") == false, "unexpected resource binding/access");
            auto chunk = uint32_t(Number(resource.at("chunk"), 63, "chunk index"));
            descriptor_chunks[index] = chunk;
            const auto& all_chunks = manifest.at("chunks");
            Require(all_chunks.is_array() && all_chunks.size() <= 64 && chunk < all_chunks.size(), "missing chunk");
            const auto& c = all_chunks.at(chunk);
            Require(c.at("chunk") == chunk, "chunk index mismatch");
            auto size = Number(c.at("range"), MaxBytes, "view range");
            auto offset = Number(c.at("offset"), UINT64_MAX, "original view offset");
            Require(size > 0 && size % 4 == 0 && offset % 4 == 0 && offset <= UINT64_MAX - size, "invalid view bounds");
            auto prefix = Number(resource.at("prefix"), 255, "prefix");
            auto packed = push[9 + index / 4];
            Require(prefix == ((packed >> ((index % 4) * 8)) & 255) && prefix % 4 == 0 && prefix <= size &&
                    Number(resource.at("source_size"), MaxBytes, "source size") <= size - prefix, "prefix/range mismatch");
            auto name = c.at("file").get<std::string>();
            Require(fs::path(name).filename() == name && !name.empty() && name != "." && name != "..", "unsafe chunk filename");
            if (!chunks.contains(chunk)) {
                Require(size <= MaxBytes - bytes, "mesh inputs exceed memory cap");
                bytes += size;
                InputChunk input {manifest_path.parent_path() / name, size, 0};
                input.hash = StreamFile(input.path, size);
                chunks.emplace(chunk, std::move(input));
            }
        }
        // Match the captured image extent, but use an independent color attachment.
        auto extent = ps.at("images_uncaptured").at(0).at("extent");
        width = Number(extent.at(0), 4096, "width");
        height = Number(extent.at(1), 4096, "height");
        Require(width && height && uint64_t(width) * height * 4 <= 32 * 1024 * 1024, "oversize offscreen target");
        Require(fs::file_size(spv) == MeshBytes, "unrecognized mesh module size");
        mesh.resize(MeshBytes / 4);
        Require(StreamFile(spv, MeshBytes, mesh.data()) == MeshFnv, "unrecognized mesh module; refusing altered SPIR-V");
        spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_3);
        validator.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& pos, const char* text) {
            std::cerr << "SPIR-V word " << pos.index << ": " << text << '\n';
        });
        Require(validator.Validate(mesh), "captured mesh failed Vulkan1.3 SPIR-V validation");
    }

    Json Description() const {
        Json j { {"mesh_spirv_unchanged", true}, {"mesh_spirv_sha256_reference", MeshSha},
            {"mesh_spirv_fnv1a64", Hex(MeshFnv)}, {"mesh_spirv_bytes", MeshBytes}, {"mesh_bytes", bytes},
            {"groups", groups}, {"push_constants", push}, {"extent", {width, height}},
            {"original_pixel_shader", false}, {"original_pixel_images", false},
            {"original_raster_state", false}, {"captured_view_offsets_rebased_to_zero", true},
            {"ordering_changed", true}, {"buffer_prefixes_preserved", true},
            {"push_constant_range", {{"offset", 0}, {"size", 128}, {"stages", "MeshEXT"}}},
            {"fragment_fixture", "flat hash of locations0..3; observes provoking-vertex attributes only; alpha1 coverage"},
            {"raster_fixture", "full captured-image extent viewport; fill; no cull; no depth; no blend; one sample"},
            {"bda_descriptors", "zero dummy; exact module indexed branch disabled by push[3]=0"} };
        for (const auto& [index, c] : chunks) j["inputs"].push_back({{"chunk", index}, {"file", c.path.string()},
            {"range", c.size}, {"fnv1a64", Hex(c.hash)}});
        return j;
    }
};

struct Buffer { VkBuffer buffer {}; VkDeviceMemory memory {}; VkDeviceSize size {}; void* mapped {}; };
class Replay {
public:
    // Timeout/device-loss uses _Exit: never destroy possibly in-flight objects or
    // invoke vkDeviceWaitIdle. This executable is one experiment per process.
    VkInstance instance {};
    VkPhysicalDevice physical {};
    VkDevice device {};
    VkQueue queue {};
    uint32_t queue_family = 0;
    VkPhysicalDeviceMemoryProperties memory_properties {};
    VkPhysicalDeviceProperties properties {};
    std::vector<Buffer> buffers;
    VkImage image {};
    VkDeviceMemory image_memory {};
    VkImageView image_view {};
    VkDescriptorSetLayout descriptor_layout {};
    VkDescriptorPool descriptor_pool {};
    VkPipelineLayout pipeline_layout {};
    VkPipeline pipeline {};
    VkShaderModule ms_module {}, ps_module {};
    VkCommandPool command_pool {};
    VkFence fence {};
    VkQueryPool primitives {}, statistics {};
    PFN_vkCmdDrawMeshTasksEXT draw_mesh = nullptr;
    bool use_required_subgroup = false;
    bool mesh_queries = false, pipeline_queries = false;
    uint64_t allocated = 0;
    Json host;

    ~Replay() {
        // Every successful queue submission has been fenced before destruction.
        if (device) {
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (primitives) vkDestroyQueryPool(device, primitives, nullptr);
            if (statistics) vkDestroyQueryPool(device, statistics, nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (ms_module) vkDestroyShaderModule(device, ms_module, nullptr);
            if (ps_module) vkDestroyShaderModule(device, ps_module, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            if (descriptor_layout) vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
            if (image_view) vkDestroyImageView(device, image_view, nullptr);
            if (image) vkDestroyImage(device, image, nullptr);
            if (image_memory) vkFreeMemory(device, image_memory, nullptr);
            for (auto& b : buffers) {
                if (b.mapped) vkUnmapMemory(device, b.memory);
                if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
                if (b.memory) vkFreeMemory(device, b.memory, nullptr);
            }
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    void Init(uint32_t device_index, const Capture& capture) {
        VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "ZeroCompanyMeshReplay";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &app;
        Vk(vkCreateInstance(&info, nullptr, &instance), "vkCreateInstance");
        uint32_t count = 0;
        Vk(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate devices");
        Require(device_index < count && count <= 64, "device index unavailable");
        std::vector<VkPhysicalDevice> devices(count);
        Vk(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "enumerate devices");
        physical = devices[device_index];
        vkGetPhysicalDeviceProperties(physical, &properties);
        Require(properties.apiVersion >= VK_API_VERSION_1_3, "Vulkan1.3 required");
        Vk(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr), "enumerate extensions");
        std::vector<VkExtensionProperties> extensions(count);
        Vk(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data()), "enumerate extensions");
        for (auto name : {VK_EXT_MESH_SHADER_EXTENSION_NAME, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME}) {
            Require(std::any_of(extensions.begin(), extensions.end(), [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; }),
                    std::string("missing extension ") + name);
        }
        VkPhysicalDeviceRobustness2FeaturesEXT robust {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
        VkPhysicalDeviceMeshShaderFeaturesEXT mesh {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceVulkan13Features f13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features f12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 f {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f.pNext = &f12; f12.pNext = &f13; f13.pNext = &mesh; mesh.pNext = &robust;
        vkGetPhysicalDeviceFeatures2(physical, &f);
        Require(mesh.meshShader && f.features.shaderInt64 && f.features.vertexPipelineStoresAndAtomics &&
                f.features.robustBufferAccess && robust.robustBufferAccess2 && f12.bufferDeviceAddress &&
                f13.dynamicRendering && f13.synchronization2, "required mesh/int64/BDA/robustness/dynamic-rendering features unavailable");
        mesh_queries = mesh.meshShaderQueries;
        pipeline_queries = mesh_queries && f.features.pipelineStatisticsQuery;
        VkPhysicalDeviceMeshShaderPropertiesEXT mp {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
        VkPhysicalDeviceSubgroupProperties sg {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceVulkan12Properties p12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
        VkPhysicalDeviceVulkan13Properties p13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
        VkPhysicalDeviceProperties2 p {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p.pNext = &mp; mp.pNext = &sg; sg.pNext = &p12; p12.pNext = &p13;
        vkGetPhysicalDeviceProperties2(physical, &p);
        Require(p12.shaderSignedZeroInfNanPreserveFloat32, "captured float controls unsupported");
        constexpr auto ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        Require((sg.supportedStages & VK_SHADER_STAGE_MESH_BIT_EXT) && (sg.supportedOperations & ops) == ops,
                "mesh subgroup ballot/shuffle unsupported");
        use_required_subgroup = f13.subgroupSizeControl && (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_MESH_BIT_EXT) &&
                                p13.minSubgroupSize <= 32 && p13.maxSubgroupSize >= 32;
        Require(use_required_subgroup || sg.subgroupSize == 32, "exact module requires subgroup32");
        auto round = [](uint64_t n, uint64_t g) { return ((n + g - 1) / g) * g; };
        Require(mp.meshOutputPerVertexGranularity && mp.meshOutputPerPrimitiveGranularity, "invalid mesh granularity");
        // Conservative output storage includes position + four vec4 parameters,
        // three triangle indices and one cull word; shared arrays total 4104B.
        auto output_bytes = round(256, mp.meshOutputPerVertexGranularity) * 80 +
                            round(254, mp.meshOutputPerPrimitiveGranularity) * 16;
        Require(mp.maxMeshWorkGroupInvocations >= 128 && mp.maxMeshWorkGroupSize[0] >= 128 &&
                mp.maxMeshOutputVertices >= 256 && mp.maxMeshOutputPrimitives >= 254 &&
                mp.maxMeshOutputComponents >= 24 && mp.maxMeshSharedMemorySize >= 4104 &&
                mp.maxMeshOutputMemorySize >= output_bytes && mp.maxMeshPayloadAndOutputMemorySize >= output_bytes &&
                mp.maxMeshPayloadAndSharedMemorySize >= 4104 && mp.maxMeshWorkGroupTotalCount >= capture.groups[0] &&
                mp.maxMeshWorkGroupCount[0] >= capture.groups[0], "captured mesh exceeds device limits");
        Require(properties.limits.maxPushConstantsSize >= 128 && properties.limits.maxPerStageDescriptorStorageBuffers >= 11 &&
                properties.limits.maxDescriptorSetStorageBuffers >= 11 && properties.limits.maxFramebufferWidth >= capture.width &&
                properties.limits.maxFramebufferHeight >= capture.height && properties.limits.maxImageDimension2D >= std::max(capture.width, capture.height) &&
                properties.limits.maxViewportDimensions[0] >= capture.width && properties.limits.maxViewportDimensions[1] >= capture.height,
                "replay exceeds graphics limits");
        for (const auto& [index, c] : capture.chunks) Require(c.size <= properties.limits.maxStorageBufferRange, "captured storage view exceeds host range");
        vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
        auto q = std::find_if(queues.begin(), queues.end(), [](const auto& x) { return x.queueFlags & VK_QUEUE_GRAPHICS_BIT; });
        Require(q != queues.end(), "graphics queue unavailable");
        queue_family = uint32_t(q - queues.begin());
        float priority = 1;
        VkDeviceQueueCreateInfo qi {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = queue_family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
        // Enable only used capabilities. No image atomics or 8/16-bit storage
        // capability occurs in this pinned module or the replacement fragment shader.
        robust = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT}; robust.robustBufferAccess2 = VK_TRUE;
        mesh = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT}; mesh.meshShader = VK_TRUE; mesh.meshShaderQueries = mesh_queries; mesh.pNext = &robust;
        f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES}; f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE; f13.subgroupSizeControl = use_required_subgroup; f13.pNext = &mesh;
        f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.bufferDeviceAddress = VK_TRUE; f12.pNext = &f13;
        VkPhysicalDeviceFeatures enabled {};
        enabled.shaderInt64 = VK_TRUE; enabled.vertexPipelineStoresAndAtomics = VK_TRUE; enabled.robustBufferAccess = VK_TRUE; enabled.pipelineStatisticsQuery = pipeline_queries;
        const char* names[] {VK_EXT_MESH_SHADER_EXTENSION_NAME, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME};
        VkDeviceCreateInfo di {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.pNext = &f12; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi; di.pEnabledFeatures = &enabled;
        di.enabledExtensionCount = 2; di.ppEnabledExtensionNames = names;
        Vk(vkCreateDevice(physical, &di, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, queue_family, 0, &queue);
        draw_mesh = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksEXT"));
        Require(draw_mesh != nullptr, "vkCmdDrawMeshTasksEXT unavailable");
        host = {{"device", properties.deviceName}, {"vendor_id", properties.vendorID}, {"device_id", properties.deviceID},
            {"driver_version", properties.driverVersion}, {"api_version", properties.apiVersion}, {"queue_family", queue_family},
            {"default_subgroup_size", sg.subgroupSize}, {"required_subgroup32", use_required_subgroup}, {"robust_buffer_access2", true},
            {"mesh_queries", mesh_queries}, {"pipeline_queries", pipeline_queries}, {"conservative_output_bytes", output_bytes},
            {"enabled_extensions", names},
            {"enabled_features", {"meshShader", "shaderInt64", "vertexPipelineStoresAndAtomics", "robustBufferAccess",
                "robustBufferAccess2", "bufferDeviceAddress", "dynamicRendering", "synchronization2"}},
            {"local_size", {128, 1, 1}}, {"guest_wave_size", 64}, {"virtual_halves", 2},
            {"max_storage_buffer_range", properties.limits.maxStorageBufferRange}};
        std::cout << "device: " << properties.deviceName << "; subgroup32=" << (use_required_subgroup ? "required" : "default") << std::endl;
    }

    VkDeviceMemory Allocate(const VkMemoryRequirements& r, VkMemoryPropertyFlags flags) {
        Require(r.size <= MaxBytes - allocated, "total Vulkan allocations exceed 768MiB");
        uint32_t type = 0;
        for (; type < memory_properties.memoryTypeCount; ++type) if ((r.memoryTypeBits & (1u << type)) &&
            (memory_properties.memoryTypes[type].propertyFlags & flags) == flags) break;
        Require(type < memory_properties.memoryTypeCount, "required memory type unavailable");
        VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = r.size; ai.memoryTypeIndex = type;
        VkDeviceMemory mem {};
        Vk(vkAllocateMemory(device, &ai, nullptr, &mem), "allocate memory");
        allocated += r.size;
        return mem;
    }
    size_t NewBuffer(uint64_t size, VkBufferUsageFlags usage) {
        buffers.emplace_back();
        auto& b = buffers.back(); b.size = size;
        VkBufferCreateInfo bi {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = usage;
        Vk(vkCreateBuffer(device, &bi, nullptr, &b.buffer), "create buffer");
        VkMemoryRequirements r {}; vkGetBufferMemoryRequirements(device, b.buffer, &r);
        b.memory = Allocate(r, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        Vk(vkBindBufferMemory(device, b.buffer, b.memory, 0), "bind buffer");
        Vk(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "map buffer");
        return buffers.size() - 1;
    }
    void Flush(const Buffer& b, bool invalidate = false) {
        VkMappedMemoryRange range {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; range.memory = b.memory; range.size = VK_WHOLE_SIZE;
        Vk(invalidate ? vkInvalidateMappedMemoryRanges(device, 1, &range) : vkFlushMappedMemoryRanges(device, 1, &range), "mapped memory visibility");
    }

    Json Run(const Capture& capture, const fs::path& out, uint32_t timeout_ms) {
        std::map<uint32_t, size_t> uploaded;
        for (const auto& [index, c] : capture.chunks) {
            auto b = NewBuffer(c.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            Require(StreamFile(c.path, c.size, buffers[b].mapped) == c.hash, "input changed since validation");
            Flush(buffers[b]); uploaded[index] = b;
        }
        auto dummy = NewBuffer(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memset(buffers[dummy].mapped, 0, 16); Flush(buffers[dummy]);
        auto readback = NewBuffer(uint64_t(capture.width) * capture.height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        VkDescriptorSetLayoutBinding binding[3] {};
        binding[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr};
        binding[1] = {51, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr};
        binding[2] = {52, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr};
        VkDescriptorSetLayoutCreateInfo li {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; li.bindingCount = 3; li.pBindings = binding;
        Vk(vkCreateDescriptorSetLayout(device, &li, nullptr, &descriptor_layout), "descriptor layout");
        VkDescriptorPoolSize ps {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 11};
        VkDescriptorPoolCreateInfo pi {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pi.maxSets = 1; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
        Vk(vkCreateDescriptorPool(device, &pi, nullptr, &descriptor_pool), "descriptor pool");
        VkDescriptorSetAllocateInfo da {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; da.descriptorPool = descriptor_pool; da.descriptorSetCount = 1; da.pSetLayouts = &descriptor_layout;
        VkDescriptorSet set {}; Vk(vkAllocateDescriptorSets(device, &da, &set), "descriptor set");
        std::array<VkDescriptorBufferInfo, 11> views {};
        for (size_t i = 0; i < 9; ++i) { const auto& b = buffers[uploaded.at(capture.descriptor_chunks[i])]; views[i] = {b.buffer, 0, b.size}; }
        views[9] = views[10] = {buffers[dummy].buffer, 0, 16};
        VkWriteDescriptorSet writes[3] {};
        for (size_t i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = set; writes[i].dstBinding = binding[i].binding;
            writes[i].descriptorCount = binding[i].descriptorCount; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = views.data() + (i == 0 ? 0 : 8 + i);
        }
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
        VkPushConstantRange push_range {VK_SHADER_STAGE_MESH_BIT_EXT, 0, 128};
        VkPipelineLayoutCreateInfo pl {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl.setLayoutCount = 1; pl.pSetLayouts = &descriptor_layout; pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &push_range;
        Vk(vkCreatePipelineLayout(device, &pl, nullptr, &pipeline_layout), "pipeline layout");
        VkFormatProperties format_properties {}; vkGetPhysicalDeviceFormatProperties(physical, VK_FORMAT_R8G8B8A8_UNORM, &format_properties);
        Require((format_properties.optimalTilingFeatures & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT)) ==
                (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT), "offscreen format unsupported");
        VkImageCreateInfo ii {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R8G8B8A8_UNORM;
        ii.extent = {capture.width, capture.height, 1}; ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        Vk(vkCreateImage(device, &ii, nullptr, &image), "create color image");
        VkMemoryRequirements requirements {}; vkGetImageMemoryRequirements(device, image, &requirements);
        image_memory = Allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Vk(vkBindImageMemory(device, image, image_memory, 0), "bind image");
        VkImageViewCreateInfo vi {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = ii.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        Vk(vkCreateImageView(device, &vi, nullptr, &image_view), "create color view");
        VkShaderModuleCreateInfo sm {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize = capture.mesh.size() * 4; sm.pCode = capture.mesh.data();
        Vk(vkCreateShaderModule(device, &sm, nullptr, &ms_module), "create unchanged mesh module");
        sm.codeSize = sizeof(ZERO_COMPANY_REPLAY_FRAG); sm.pCode = ZERO_COMPANY_REPLAY_FRAG;
        Vk(vkCreateShaderModule(device, &sm, nullptr, &ps_module), "create attribute-hash fragment module");
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO}; subgroup.requiredSubgroupSize = 32;
        VkPipelineShaderStageCreateInfo stages[2] {};
        for (auto& s : stages) { s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; s.pName = "main"; }
        stages[0].stage = VK_SHADER_STAGE_MESH_BIT_EXT; stages[0].module = ms_module; stages[0].pNext = use_required_subgroup ? &subgroup : nullptr;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = ps_module;
        VkViewport viewport {0, 0, float(capture.width), float(capture.height), 0, 1};
        VkRect2D scissor {{0, 0}, {capture.width, capture.height}};
        VkPipelineViewportStateCreateInfo vp {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; vp.viewportCount = 1; vp.pViewports = &viewport; vp.scissorCount = 1; vp.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo rs {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo samples {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState attachment {}; attachment.colorWriteMask = 15;
        VkPipelineColorBlendStateCreateInfo blend {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; blend.attachmentCount = 1; blend.pAttachments = &attachment;
        VkPipelineRenderingCreateInfo render {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO}; render.colorAttachmentCount = 1; render.pColorAttachmentFormats = &ii.format;
        VkGraphicsPipelineCreateInfo gp {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; gp.pNext = &render; gp.stageCount = 2; gp.pStages = stages;
        gp.pViewportState = &vp; gp.pRasterizationState = &rs; gp.pMultisampleState = &samples; gp.pColorBlendState = &blend; gp.layout = pipeline_layout;
        std::cout << "creating true MeshEXT graphics pipeline; unchanged mesh, flat attribute-hash PS" << std::endl;
        Vk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeline), "create mesh graphics pipeline");
        VkCommandPoolCreateInfo cp {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cp.queueFamilyIndex = queue_family;
        Vk(vkCreateCommandPool(device, &cp, nullptr, &command_pool), "command pool");
        VkCommandBufferAllocateInfo ca {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ca.commandPool = command_pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
        VkCommandBuffer cmd {}; Vk(vkAllocateCommandBuffers(device, &ca, &cmd), "command buffer");
        VkQueryPoolCreateInfo qc {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qc.queryCount = 1;
        if (mesh_queries) { qc.queryType = VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT; Vk(vkCreateQueryPool(device, &qc, nullptr, &primitives), "primitive query"); }
        if (pipeline_queries) { qc.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS; qc.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT | VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT; Vk(vkCreateQueryPool(device, &qc, nullptr, &statistics), "pipeline query"); }
        VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Vk(vkBeginCommandBuffer(cmd, &begin), "begin commands");
        VkMemoryBarrier2 host_barrier {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2}; host_barrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT; host_barrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        host_barrier.dstStageMask = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT; host_barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkImageMemoryBarrier2 transition {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2}; transition.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT; transition.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        transition.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; transition.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; transition.image = image;
        transition.srcQueueFamilyIndex = transition.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; transition.subresourceRange = vi.subresourceRange;
        VkDependencyInfo dep {VK_STRUCTURE_TYPE_DEPENDENCY_INFO}; dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &host_barrier; dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &transition;
        vkCmdPipelineBarrier2(cmd, &dep);
        if (primitives) vkCmdResetQueryPool(cmd, primitives, 0, 1);
        if (statistics) vkCmdResetQueryPool(cmd, statistics, 0, 1);
        VkRenderingAttachmentInfo color {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}; color.imageView = image_view; color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; color.storeOp = VK_ATTACHMENT_STORE_OP_STORE; color.clearValue.color = {{0, 0, 0, 0}};
        VkRenderingInfo ri {VK_STRUCTURE_TYPE_RENDERING_INFO}; ri.renderArea = scissor; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &color;
        vkCmdBeginRendering(cmd, &ri);
        if (primitives) vkCmdBeginQuery(cmd, primitives, 0, 0);
        if (statistics) vkCmdBeginQuery(cmd, statistics, 0, 0);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_MESH_BIT_EXT, 0, 128, capture.push.data());
        draw_mesh(cmd, capture.groups[0], capture.groups[1], capture.groups[2]);
        if (statistics) vkCmdEndQuery(cmd, statistics, 0);
        if (primitives) vkCmdEndQuery(cmd, primitives, 0);
        vkCmdEndRendering(cmd);
        transition.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT; transition.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        transition.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT; transition.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        transition.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; transition.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        dep.memoryBarrierCount = 0; dep.pMemoryBarriers = nullptr;
        vkCmdPipelineBarrier2(cmd, &dep);
        VkBufferImageCopy copy {}; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageExtent = ii.extent;
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffers[readback].buffer, 1, &copy);
        host_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
        host_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        host_barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT; host_barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        dep.imageMemoryBarrierCount = 0; dep.pImageMemoryBarriers = nullptr; dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &host_barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
        Vk(vkEndCommandBuffer(cmd), "end commands");
        VkFenceCreateInfo fc {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; Vk(vkCreateFence(device, &fc, nullptr, &fence), "fence");
        Json result = capture.Description(); result["host"] = host; result["allocated_bytes"] = allocated; result["fence_timeout_ms"] = timeout_ms;
        result["status"] = "prepared"; WriteJson(out / "prepared.json", result);
        VkSubmitInfo submit {VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd;
        std::cout << "submitting " << capture.groups[0] << " mesh groups; finite fence " << timeout_ms << "ms" << std::endl;
        const auto submitted = vkQueueSubmit(queue, 1, &submit, fence);
        const auto waited = submitted == VK_SUCCESS ? vkWaitForFences(device, 1, &fence, VK_TRUE, uint64_t(timeout_ms) * 1000000) : submitted;
        if (waited != VK_SUCCESS) {
            result["status"] = waited == VK_TIMEOUT ? "gpu_timeout" : "gpu_failure"; result["vk_result"] = waited;
            try { WriteJson(out / "result.json", result); } catch (const std::exception& e) { std::cerr << e.what() << '\n'; }
            std::cerr << "GPU did not complete: VkResult=" << waited << "; exiting without Vulkan teardown" << std::endl;
            std::_Exit(waited == VK_TIMEOUT ? 124 : 3);
        }
        Flush(buffers[readback], true);
        Flush(buffers[dummy], true);
        std::array<uint32_t, 4> faults {};
        std::memcpy(faults.data(), buffers[dummy].mapped, sizeof(faults));
        uint64_t covered = 0;
        auto* pixels = static_cast<const uint8_t*>(buffers[readback].mapped);
        for (uint64_t i = 0; i < buffers[readback].size; i += 4) if (pixels[i + 3]) ++covered;
        result["status"] = "completed"; result["covered_pixels"] = covered;
        result["dummy_bda_words"] = faults;
        const bool fault = std::any_of(faults.begin(), faults.end(), [](uint32_t word) { return word != 0; });
        if (fault) result["status"] = "unexpected_bda_fault";
        result["readback_fnv1a64"] = Hex(Hash(pixels, buffers[readback].size));
        result["raster_observed"] = covered != 0;
        if (primitives) { uint64_t value = 0; Vk(vkGetQueryPoolResults(device, primitives, 0, 1, sizeof(value), &value, sizeof(value), VK_QUERY_RESULT_64_BIT), "primitive query result"); result["mesh_primitives_generated"] = value; }
        if (statistics) {
            std::array<uint64_t, 2> values {};
            Vk(vkGetQueryPoolResults(device, statistics, 0, 1, sizeof(values), values.data(), sizeof(values), VK_QUERY_RESULT_64_BIT), "pipeline query result");
            // Values are ordered by flag bit: fragment (7), then mesh (12).
            result["fragment_shader_invocations"] = values[0]; result["mesh_shader_invocations"] = values[1];
        }
        Write(out / "color.rgba8", pixels, buffers[readback].size);
        WriteJson(out / "result.json", result);
        Require(!fault, "unexpected write to dummy BDA/fault storage; replay assumptions failed");
        std::cout << "completed: covered_pixels=" << covered << "; this is mesh+fixture completion, not original PS/image validation" << std::endl;
        return result;
    }
};
}

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::cerr << "Usage: zero_company_mesh_replay MANIFEST MS_SPV NEW_OUTPUT_DIR [--validate-only] [--device N] [--timeout-ms N]\n";
            return 2;
        }
        bool validate_only = false;
        uint32_t device = 0, timeout = 10000;
        for (int i = 4; i < argc; ++i) {
            std::string option = argv[i];
            if (option == "--validate-only") validate_only = true;
            else if ((option == "--device" || option == "--timeout-ms") && i + 1 < argc) {
                std::string value = argv[++i];
                Require(!value.empty() && value.size() <= 6 && value.find_first_not_of("0123456789") == std::string::npos, "invalid numeric option");
                auto n = std::stoul(value);
                if (option == "--device") { Require(n < 64, "device index out of range"); device = n; }
                else { Require(n >= 100 && n <= 60000, "timeout must be 100..60000ms"); timeout = n; }
            } else throw std::runtime_error("unknown or incomplete option: " + option);
        }
        Capture capture(fs::absolute(argv[1]), fs::absolute(argv[2]));
        fs::path out = fs::absolute(argv[3]);
        Require(!fs::exists(out) && fs::create_directory(out), "output must be a new directory under an existing parent");
        auto validated = capture.Description(); validated["status"] = "cpu_validated";
        validated["manifest"] = fs::absolute(argv[1]).string(); validated["mesh_spirv"] = fs::absolute(argv[2]).string();
        WriteJson(out / "validation.json", validated);
        std::cout << "CPU validation complete: " << capture.bytes << " bytes, groups=" << capture.groups[0] << "; SPIR-V unchanged" << std::endl;
        if (validate_only) return 0; // Does not create a Vulkan instance or device.
        Replay replay;
        replay.Init(device, capture);
        replay.Run(capture, out, timeout);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "replay failed: " << error.what() << std::endl;
        return 1;
    }
}
