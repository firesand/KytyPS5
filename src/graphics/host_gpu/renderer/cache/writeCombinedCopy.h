#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_WRITECOMBINEDCOPY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_WRITECOMBINEDCOPY_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

// Copies from write-combined / uncached memory to cached memory. The CPU mapping of host-visible
// device-local memory (ReBAR; Vulkan memory type 4 here = DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT,
// no HOST_CACHED) is uncached for reads: ordinary loads are one PCIe round trip each (ng-150: ~50 us
// per GPU-thread BAR readback of a page; ng-173: libc memmove under TryBarReadback ~12 % of the GPU
// thread). SSE4.1 streaming loads (MOVNTDQA) fetch whole 64-byte lines into the streaming-load
// buffers instead. The leading MFENCE invalidates stale streaming-load buffer lines (Intel's SSE4
// streaming-load guidance: required before reading a location that may have changed). Unaligned
// head and tail bytes use ordinary loads.
#if defined(__x86_64__)
__attribute__((target("sse4.1"))) inline void CopyFromWriteCombinedSse41(uint8_t* dst, const uint8_t* src,
                                                                        size_t size) {
	_mm_mfence();
	const auto head = static_cast<size_t>((16u - (reinterpret_cast<uintptr_t>(src) & 15u)) & 15u);
	if (head >= size) {
		std::memcpy(dst, src, size);
		return;
	}
	std::memcpy(dst, src, head);
	dst += head;
	src += head;
	size -= head;
	const auto load = [](const uint8_t* p) {
		return _mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<uint8_t*>(p)));
	};
	// 16-byte blocks up to the next 64-byte line, then whole lines (four loads in flight per line).
	while (size >= 16 && (reinterpret_cast<uintptr_t>(src) & 63u) != 0) {
		_mm_storeu_si128(reinterpret_cast<__m128i*>(dst), load(src));
		dst += 16;
		src += 16;
		size -= 16;
	}
	while (size >= 64) {
		const __m128i a = load(src);
		const __m128i b = load(src + 16);
		const __m128i c = load(src + 32);
		const __m128i d = load(src + 48);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(dst), a);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), b);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 32), c);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 48), d);
		dst += 64;
		src += 64;
		size -= 64;
	}
	while (size >= 16) {
		_mm_storeu_si128(reinterpret_cast<__m128i*>(dst), load(src));
		dst += 16;
		src += 16;
		size -= 16;
	}
	std::memcpy(dst, src, size);
}
#endif

// KYTY_BAR_STREAMING_COPY=0 falls back to memcpy (A/B within one binary).
[[nodiscard]] inline bool WriteCombinedCopyEnabled() {
#if defined(__x86_64__)
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BAR_STREAMING_COPY");
		return !(value != nullptr && value[0] == '0') && __builtin_cpu_supports("sse4.1");
	}();
	return enabled;
#else
	return false;
#endif
}

inline void CopyFromWriteCombined(void* dst, const void* src, size_t size, bool streaming) {
#if defined(__x86_64__)
	if (streaming) {
		CopyFromWriteCombinedSse41(static_cast<uint8_t*>(dst), static_cast<const uint8_t*>(src), size);
		return;
	}
#endif
	(void)streaming;
	std::memcpy(dst, src, size);
}

inline void CopyFromWriteCombined(void* dst, const void* src, size_t size) {
	CopyFromWriteCombined(dst, src, size, WriteCombinedCopyEnabled());
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_WRITECOMBINEDCOPY_H_
