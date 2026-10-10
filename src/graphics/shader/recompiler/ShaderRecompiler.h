#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shader.h"

#include <array>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions {
	ShaderType                  stage           = ShaderType::Compute;
	uint32_t                    wave_size       = 64;
	uint32_t                    user_data_base  = 0;
	uint64_t                    shader_hash     = 0;
	bool                        dump_ir                    = true;
	bool                        early_dump                 = false;
	const char*                 dump_label                 = nullptr;
	std::span<const uint32_t>   user_data;
	std::span<const uint32_t>   back_code;
	ShaderStageInputInfo        input_info;
};

struct TranslateResult {
	IR::Program program;
	std::string decoded_dump;
	std::string cfg_dump;
};

struct CompileResult {
	std::vector<uint32_t>  spirv;
	std::string            decoded_dump;
	std::string            ir_dump;
	IR::Program            program;
};

// Decoded source and typed scalar call queries are retained once per cache source.
struct ShaderSource {
	// A call whose target is read from a function table with a per-wave (non-uniform) index:
	// the target pair is S_BUFFER_LOADed from `descriptor` at a dynamic offset. Every distinct
	// table entry is inlined behind a dispatch on `compare_reg` (the loaded low target dword).
	struct Table {
		std::array<IR::Value, 4> descriptor;
		uint32_t                 entry_offset = 0; // byte offset of the target within an entry
		uint32_t                 entry_stride = 0; // bytes per entry (the load width)
		uint32_t                 compare_reg  = 0;
		uint32_t                 wave_size    = 64;
	};
	struct Call {
		uint32_t instruction;
		std::array<IR::Value, 2> target;
		std::optional<Table> table;
	};
	struct Linked {
		std::vector<uint32_t> code;
		Decoder::Program decoded;
		std::vector<uint32_t> observed_function;
		// Table calls: guest memory the link depends on (the table and every inlined function).
		std::vector<std::pair<uint64_t, std::vector<uint32_t>>> observed;
	};
	std::vector<uint32_t> code;
	Decoder::Program decoded;
	IR::ResourcePlan call_targets;
	std::optional<Call> call;
	std::optional<Linked> linked;
	uint64_t revision = 0;
	std::vector<std::pair<uint64_t, uint64_t>> reads;
	// Set when the source cannot be translated yet; the caller should skip the work.
	std::string_view unsupported;
};

[[nodiscard]] ShaderSource PrepareShaderSource(std::span<const uint32_t> code,
	                                           const CompileOptions& options);
const Decoder::Program& RefreshShaderSource(ShaderSource& source, const IR::SrtRuntime& runtime);
[[nodiscard]] TranslateResult TranslateProgram(const Decoder::Program& decoded,
	                                           const CompileOptions& options);

[[nodiscard]] TranslateResult TranslateProgram(std::span<const uint32_t> code,
                                               const CompileOptions& options);
[[nodiscard]] CompileResult CompileProgram(TranslateResult translated,
                                           const CompileOptions& options,
                                           const IR::ResourceSpecialization& specialization,
	                                       uint32_t push_data_start_dword = 0);

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_ */
