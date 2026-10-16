#pragma once

// Diagnostic shader bisection, all default-off and env controlled (one static bool per draw/dispatch):
//   KYTY_SKIP_PS=<hex,hex,...>   skip draws whose pixel shader hash is listed (up to 64 hashes)
//   KYTY_SKIP_CS=<hex,hex,...>   skip dispatches whose compute shader hash is listed
//   KYTY_SKIP_PS_BITS=<mask hex> KYTY_SKIP_PS_VALUE=<value hex>
//                                skip every pixel shader with (hash & mask) == value (mask 0 = off)
//   KYTY_DRAW_RT_LOG=1           once per second list distinct (ps hash, render-target-0 size/format)
// Skipping only omits the vkCmdDraw*/vkCmdDispatch* call; binds and bookkeeping still run.

#include "common/logging/log.h"

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::Graphics::SkipShader {

inline constexpr size_t MAX_HASHES = 64;

// Parses "0x1234,abcd, 0XFF": comma separated 64-bit hex, optional 0x, surrounding spaces ignored.
// Invalid or empty tokens are dropped; at most MAX_HASHES are kept.
inline std::vector<uint64_t> ParseHashList(std::string_view text) {
	std::vector<uint64_t> out;
	size_t                pos = 0;
	while (pos <= text.size() && out.size() < MAX_HASHES) {
		auto end = text.find(',', pos);
		if (end == std::string_view::npos) end = text.size();
		auto token = text.substr(pos, end - pos);
		while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) token.remove_prefix(1);
		while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) token.remove_suffix(1);
		if (token.size() > 2 && token[0] == '0' && (token[1] == 'x' || token[1] == 'X')) token.remove_prefix(2);
		if (!token.empty() && token.size() <= 16) {
			uint64_t value = 0;
			bool     ok    = true;
			for (const char c: token) {
				uint32_t d = 0;
				if (c >= '0' && c <= '9') d = static_cast<uint32_t>(c - '0');
				else if (c >= 'a' && c <= 'f') d = static_cast<uint32_t>(c - 'a' + 10);
				else if (c >= 'A' && c <= 'F') d = static_cast<uint32_t>(c - 'A' + 10);
				else { ok = false; break; }
				value = (value << 4u) | d;
			}
			if (ok) out.push_back(value);
		}
		pos = end + 1;
	}
	return out;
}

// Parses one hex value (optional 0x); returns false when absent or invalid.
inline bool ParseHex(const char* text, uint64_t& value) {
	if (text == nullptr) return false;
	const auto list = ParseHashList(text);
	if (list.size() != 1) return false;
	value = list[0];
	return true;
}

inline bool BitsMatch(uint64_t hash, uint64_t mask, uint64_t value) {
	return mask != 0 && (hash & mask) == value;
}

inline bool Contains(const std::vector<uint64_t>& set, uint64_t hash) {
	for (const auto h: set) {
		if (h == hash) return true;
	}
	return false;
}

struct Config {
	std::vector<uint64_t> ps;
	std::vector<uint64_t> cs;
	uint64_t              bits_mask  = 0;
	uint64_t              bits_value = 0;
	bool                  draw_rt    = false;

	[[nodiscard]] bool SkipPsEnabled() const { return !ps.empty() || bits_mask != 0; }
	[[nodiscard]] bool SkipCsEnabled() const { return !cs.empty(); }
	[[nodiscard]] bool MatchPs(uint64_t hash) const { return Contains(ps, hash) || BitsMatch(hash, bits_mask, bits_value); }
	[[nodiscard]] bool MatchCs(uint64_t hash) const { return Contains(cs, hash); }
};

inline Config LoadConfig() {
	Config c;
	if (const char* s = std::getenv("KYTY_SKIP_PS")) c.ps = ParseHashList(s);
	if (const char* s = std::getenv("KYTY_SKIP_CS")) c.cs = ParseHashList(s);
	uint64_t mask = 0, value = 0;
	if (ParseHex(std::getenv("KYTY_SKIP_PS_BITS"), mask)) {
		c.bits_mask = mask;
		if (ParseHex(std::getenv("KYTY_SKIP_PS_VALUE"), value)) c.bits_value = value;
	}
	const char* rt = std::getenv("KYTY_DRAW_RT_LOG");
	c.draw_rt      = rt != nullptr && rt[0] != '\0' && rt[0] != '0';
	return c;
}

inline const Config& GetConfig() {
	static const Config config = LoadConfig();
	return config;
}

// One static bool read per draw/dispatch when unset.
inline bool PsSkipEnabled() { static const bool v = GetConfig().SkipPsEnabled(); return v; }
inline bool CsSkipEnabled() { static const bool v = GetConfig().SkipCsEnabled(); return v; }
inline bool DrawRtEnabled() { static const bool v = GetConfig().draw_rt; return v; }

struct SkipCounters {
	std::mutex                      mutex;
	std::map<uint64_t, uint64_t>    ps;
	std::map<uint64_t, uint64_t>    cs;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
	bool                            exit_hook = false;

	void Print() {
		for (const auto& [h, n]: ps) LOGF("SkipShader: ps=0x%016" PRIx64 " skipped %" PRIu64 "\n", h, n);
		for (const auto& [h, n]: cs) LOGF("SkipShader: cs=0x%016" PRIx64 " skipped %" PRIu64 "\n", h, n);
	}
};

inline SkipCounters& Counters() {
	static SkipCounters counters;
	return counters;
}

inline void PrintCountersAtExit() { auto& c = Counters(); std::lock_guard lock(c.mutex); c.Print(); }

inline void NoteSkip(bool pixel, uint64_t hash) {
	auto&                       c = Counters();
	std::lock_guard             lock(c.mutex);
	if (!c.exit_hook) { c.exit_hook = true; std::atexit(PrintCountersAtExit); }
	(pixel ? c.ps : c.cs)[hash]++;
	const auto now = std::chrono::steady_clock::now();
	if (now - c.last >= std::chrono::seconds(5)) { c.last = now; c.Print(); }
}

// Returns true when this draw (pixel=true) or dispatch must not be recorded.
inline bool ShouldSkipPs(uint64_t hash) {
	if (!PsSkipEnabled() || !GetConfig().MatchPs(hash)) return false;
	NoteSkip(true, hash);
	return true;
}
inline bool ShouldSkipCs(uint64_t hash) {
	if (!CsSkipEnabled() || !GetConfig().MatchCs(hash)) return false;
	NoteSkip(false, hash);
	return true;
}

struct DrawRtLog {
	std::mutex                                                       mutex;
	std::map<std::pair<uint64_t, std::string>, uint32_t>             seen; // (ps, "WxH fmt") -> draws
	std::chrono::steady_clock::time_point                            last = std::chrono::steady_clock::now();
};

inline DrawRtLog& DrawRtState() {
	static DrawRtLog log;
	return log;
}

// Call only when DrawRtEnabled(); rt0 is e.g. "2180x1228 R16G16B16A16Sfloat" or "none".
inline void NoteDrawRt(uint64_t ps_hash, const std::string& rt0) {
	auto&           s = DrawRtState();
	std::lock_guard lock(s.mutex);
	const auto      key = std::make_pair(ps_hash, rt0);
	if (auto it = s.seen.find(key); it != s.seen.end()) it->second++;
	else if (s.seen.size() < 64) s.seen.emplace(key, 1u);
	const auto now = std::chrono::steady_clock::now();
	if (now - s.last < std::chrono::seconds(1)) return;
	s.last = now;
	for (const auto& [k, n]: s.seen) {
		LOGF("DrawRT: ps=0x%016" PRIx64 " rt0=%s draws=%u\n", k.first, k.second.c_str(), n);
	}
	s.seen.clear();
}

} // namespace Libs::Graphics::SkipShader
