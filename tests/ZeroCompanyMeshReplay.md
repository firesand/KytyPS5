# Bounded Zero Company mesh replay

This manual diagnostic replays captured MS `67e610ccc37800ed` unchanged through
a real `VK_EXT_mesh_shader` graphics pipeline. It accepts only the known module
(158472 bytes, FNV-1a64 `ca310497219a1d7a`, reference SHA256
`0f0e5672764bbf39aa0ac59fca99eae3a042dbad7d86a958bb2aaf707b33cc79`).
The original module is neither patched nor recompiled. SPIRV-Tools validates it
against Vulkan1.3 before any Vulkan instance is created.

Build explicitly:

```sh
cmake --build _Build/zero-debug --target zero_company_mesh_replay -j 4
```

The target is Linux-only, requires the Vulkan loader development library, and
is excluded from the normal build and CTest. No emulator source is linked.

Run CPU validation first (does not create a Vulkan instance/device):

```sh
_Build/zero-debug/zero_company_mesh_replay MANIFEST MS_SPV NEW_OUTPUT_DIR --validate-only
```

After all other GPU experiments have stopped, run one capture per process:

```sh
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation timeout --kill-after=5s 75s \
  _Build/zero-debug/zero_company_mesh_replay MANIFEST MS_SPV NEW_OUTPUT_DIR \
  --device 0 --timeout-ms 10000
```

Use a new directory under an existing parent for each run. Redirect stdout and
stderr to a separate new log. The external deadline is required because driver
pipeline creation and queue submission are outside the fence timeout. A GPU
timeout exits124 without destroying potentially in-flight Vulkan objects or
calling `vkDeviceWaitIdle`; it is never reported as success.

The complete schema1 manifest and all nine MS view files are required. Captured
view contents include descriptor prefixes; each full view is rebound at offset0
with its original range and packed prefix. The 128-byte push bank is restored:
draw words0..5, original PS words6..7, MS words8..11. Only nonindexed one/two-group
draws starting at zero are admitted. Bindings51/52 receive zero dummy storage:
the pinned shader's BDA calls require draw word3 !=0, which is rejected. The dummy
storage is checked after a mesh-write/host-read barrier and completed fence.

Inputs are streamed directly into mapped Vulkan buffers, flushed, and checked
against the CPU-validation fingerprints. Total Vulkan allocations are capped
at768MiB; no guest CPU memory or package files are accessed. Descriptor view
contents are immutable. Subgroup32 is required (explicit size control where
supported, otherwise verified fixed default32). Mesh/storage/output limits and
every enabled feature are checked before dispatch. This module does not declare
image-atomic64 or 8/16-bit storage capabilities, so these unused features are not
enabled. Production `robustBufferAccess2` behavior is retained.

The replacement fragment shader hashes all four flat vec4 mesh parameters into
RGB with alpha1. It observes only the provoking vertex's attributes. Rendering
uses a new RGBA8 image, captured image extent for its viewport, no culling/depth/
blending, and one sample. It does not restore uncaptured original raster state
or original pixel images. Counts/indices/positions affect rasterization; offscreen,
culled, degenerate, or unreferenced outputs are not exhaustively checked. The
unchanged SPIR-V may still be optimized by the driver for this fixture.

Evidence consists of `validation.json`, `prepared.json`, then `result.json` and
`color.rgba8` only after completion. File writes first use exclusive `.partial`
files. Result metadata records input FNV-1a64 fingerprints (noncryptographic
integrity identifiers), device/features, subgroup policy, push words, range
rebasing, primitive and invocation queries if supported, coverage and readback
fingerprint. The original evidence already carries separate SHA256 inventories.
Zero coverage is reported explicitly and does not prove correct raster output.
`status=completed` means this mesh-plus-fixture submission completed; it does
not establish game playability or validate the original PS/image path.

Specification references:
- [Mesh features and query gating](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMeshShaderFeaturesEXT.html)
- [Query semantics](https://docs.vulkan.org/spec/latest/chapters/queries.html)

Suggested comparison order: earlier completed one-group capture as control,
then the captured two-group stalled draw. Compare query counts, coverage, pixels,
and validation messages; a failure before submission is a harness/preflight
failure and must not be called a reproduced GPU stall.

## Optional unchanged original PS with synthetic images

Add `--original-ps-synthetic PS_SPV` to either CPU validation or the manual GPU
command. This pins the unchanged 36044-byte PS548a module, FNV-1a64
`7928d16868b75057`, reference SHA256
`eb5cf448e698c6deb4cab510e27e5dfd63763ad87b74e36cb267d39d316528a3`.
All seven captured PS buffer views are restored at binding56; shared MS/PS
chunks remain deduplicated. The original PS push words6..7 are made visible to
the fragment stage. Barycentrics, image64 atomics, fragment storage writes,
fragment subgroup32, robust image access2, image formats and extents must be
supported. Unavailable features fail preflight instead of changing the shader.

**Image inputs in this mode are SYNTHETIC, all bits zero.** Visibility uses one
new2848x1600 image with sampled R32G32Uint view(binding66) and atomic R64Uint
view(binding102). Mutable-format/extended-usage flags permit the alias; each
view restricts its usage to sampled or storage. Noise is a new128x8192 R8Unorm
image(binding59). Transfer clears complete before fragment sampled/storage
accesses; the visibility image is copied only after a fragment-access/transfer
barrier and read on the host only after the completed fence and invalidation.

The original PS has no color output. Its color write mask is zero; its result
is `visibility.rg32uint` (tight rows, two little-endian uint32 words per pixel,
also interpretable as one little-endian uint64), modified-from-zero pixel count,
readback fingerprint and optional invocation/primitive counters. A zero write
count is reported as zero; it is not labeled successful rendering. JSON and
console output prominently label synthetic input state.

This mode currently accepts only schema1 manifests. It does not read future
schema2 image payloads, restore captured image contents, or reproduce original
raster state. Passing with synthetic zeros narrows the investigation but cannot
exclude a content-dependent image/PS failure.
