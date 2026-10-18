#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUINDIRECTFILTER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUINDIRECTFILTER_H_

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace Libs::Graphics {

// Shader-hash list for the KYTY_CP_GPU_INDIRECT bisection variables:
// - KYTY_CP_GPU_INDIRECT_MESH_HASH: mesh programs admitted to the task-shader path. Unset or empty
//   admits none (default: the mesh path stays off), `*` admits all.
// - KYTY_CP_GPU_INDIRECT_DENY_HASH: vertex or mesh programs always kept on the host path.
// Entries are hex hashes (optional 0x), separated by commas, spaces or semicolons; malformed
// entries are ignored.
struct GpuIndirectHashFilter {
	bool                  all = false;
	std::vector<uint64_t> hashes;

	[[nodiscard]] static GpuIndirectHashFilter Parse(const char* text) {
		GpuIndirectHashFilter filter;
		if (text == nullptr) {
			return filter;
		}
		const std::string_view list(text);
		size_t                 pos = 0;
		while (pos < list.size()) {
			const auto end   = list.find_first_of(", ;", pos);
			auto       entry = list.substr(pos, end == std::string_view::npos ? list.size() - pos : end - pos);
			pos              = end == std::string_view::npos ? list.size() : end + 1;
			if (entry == "*") {
				filter.all = true;
				continue;
			}
			if (entry.size() > 2 && entry[0] == '0' && (entry[1] == 'x' || entry[1] == 'X')) {
				entry.remove_prefix(2);
			}
			if (entry.empty() || entry.size() > 16) {
				continue;
			}
			uint64_t value = 0;
			bool     valid = true;
			for (const char c: entry) {
				uint32_t digit = 0;
				if (c >= '0' && c <= '9') {
					digit = static_cast<uint32_t>(c - '0');
				} else if (c >= 'a' && c <= 'f') {
					digit = static_cast<uint32_t>(c - 'a' + 10);
				} else if (c >= 'A' && c <= 'F') {
					digit = static_cast<uint32_t>(c - 'A' + 10);
				} else {
					valid = false;
					break;
				}
				value = (value << 4u) | digit;
			}
			if (valid) {
				filter.hashes.push_back(value);
			}
		}
		return filter;
	}

	[[nodiscard]] bool Contains(uint64_t hash) const {
		return all || std::find(hashes.begin(), hashes.end(), hash) != hashes.end();
	}

	// The process-wide lists, read once.
	[[nodiscard]] static const GpuIndirectHashFilter& MeshAllow() {
		static const auto filter = Parse(std::getenv("KYTY_CP_GPU_INDIRECT_MESH_HASH"));
		return filter;
	}
	[[nodiscard]] static const GpuIndirectHashFilter& Deny() {
		static const auto filter = Parse(std::getenv("KYTY_CP_GPU_INDIRECT_DENY_HASH"));
		return filter;
	}
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUINDIRECTFILTER_H_
