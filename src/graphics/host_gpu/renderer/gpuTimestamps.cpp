#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace Libs::Graphics {

namespace {
constexpr size_t MaxInstances = 4;
std::atomic<GpuTimestamps*> g_instances[MaxInstances] {};
const char* const KindNames[] = {"draw",       "dispatch",   "dispatch-indirect", "barrier-global",
                                 "barrier-buffer", "barrier-image", "copy-upload", "copy-image",
                                 "blit",       "fill",       "clear",            "cache-dispatch",
                                 "renderpass"};
static_assert(sizeof(KindNames) / sizeof(KindNames[0]) ==
              static_cast<size_t>(GpuTimestamps::Kind::Count));
const char* KindNameOf(uint8_t kind) {
	return kind < static_cast<uint8_t>(GpuTimestamps::Kind::Count) ? KindNames[kind] : "?";
}
} // namespace

const char* GpuTimestamps::KindName(Kind kind) {
	return KindNameOf(static_cast<uint8_t>(kind));
}

GpuTimestamps* GpuTimestamps::ForCommand(vk::CommandBuffer command) noexcept {
	for (auto& slot: g_instances) {
		auto* instance = slot.load(std::memory_order_acquire);
		if (instance != nullptr &&
		    instance->m_recording.load(std::memory_order_acquire) ==
		        static_cast<VkCommandBuffer>(command)) {
			return instance;
		}
	}
	return nullptr;
}

bool GpuTimestamps::Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_TIMESTAMPS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

GpuTimestamps::GpuTimestamps(GraphicContext& graphics): m_graphics(graphics) {
	const auto families = graphics.physical_device.getQueueFamilyProperties();
	uint32_t   bits     = 0;
	if (graphics.queue_family < families.size()) {
		bits = families[graphics.queue_family].timestampValidBits;
	}
	const auto period = graphics.physical_device_properties.limits.timestampPeriod;
	if (bits == 0 || period <= 0.0f) {
		LOGF("GpuTime: timestamps unsupported (validBits=%u period=%f); profiling disabled\n", bits,
		     static_cast<double>(period));
		return;
	}
	m_mask      = bits >= 64 ? ~uint64_t {0} : ((uint64_t {1} << bits) - 1);
	m_period_ns = static_cast<double>(period);

	vk::QueryPoolCreateInfo info {};
	info.queryType  = vk::QueryType::eTimestamp;
	info.queryCount = SlotCount * QueriesPerSlot;
	const auto result = graphics.device.createQueryPool(&info, nullptr, &m_pool);
	if (result != vk::Result::eSuccess) {
		m_pool = nullptr;
		LOGF("GpuTime: vkCreateQueryPool failed (%d); profiling disabled\n",
		     static_cast<int>(result));
		return;
	}
	m_results.resize(QueriesPerSlot);
	m_last_dump = std::chrono::steady_clock::now();
	for (auto& slot: g_instances) {
		GpuTimestamps* expected = nullptr;
		if (slot.compare_exchange_strong(expected, this)) break;
	}
	LOGF("GpuTime: enabled, %u slots x %u queries, period %.3f ns, validBits %u\n", SlotCount,
	     QueriesPerSlot, m_period_ns, bits);
}

GpuTimestamps::~GpuTimestamps() {
	for (auto& slot: g_instances) {
		GpuTimestamps* expected = this;
		slot.compare_exchange_strong(expected, nullptr);
	}
	if (m_pool != nullptr) {
		m_graphics.device.destroyQueryPool(m_pool, nullptr);
	}
}

void GpuTimestamps::BeginCommand(vk::CommandBuffer command, uint64_t tick,
                                 uint64_t completed_tick) {
	if (!Valid()) return;
	std::lock_guard lock(m_mutex);
	m_current = -1;
	m_op_open = false;
	m_depth   = 0;
	m_recording.store(static_cast<VkCommandBuffer>(command), std::memory_order_release);
	auto find = [this] {
		for (uint32_t i = 0; i < SlotCount; ++i) {
			if (m_slots[i].state == SlotState::Free) return static_cast<int>(i);
		}
		return -1;
	};
	int index = find();
	if (index < 0) {
		HarvestLocked(completed_tick);
		index = find();
	}
	if (index < 0) {
		++m_acc.no_slot;
		return;
	}
	auto& slot    = m_slots[static_cast<size_t>(index)];
	slot.state    = SlotState::Recording;
	slot.tick     = tick;
	slot.used     = 2;
	slot.overflow = 0;
	slot.ops.clear();
	m_current = index;
	// Reset must happen outside a render pass; the scheduler calls this right after vkBeginCommandBuffer.
	const uint32_t base = static_cast<uint32_t>(index) * QueriesPerSlot;
	command.resetQueryPool(m_pool, base, QueriesPerSlot);
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_pool, base);
}

void GpuTimestamps::EndCommand(vk::CommandBuffer command) {
	if (!Valid()) return;
	std::lock_guard lock(m_mutex);
	if (m_current < 0) {
		m_recording.store(nullptr, std::memory_order_release);
		return;
	}
	auto& slot = m_slots[static_cast<size_t>(m_current)];
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_pool,
	                       static_cast<uint32_t>(m_current) * QueriesPerSlot + 1);
	slot.state = SlotState::Pending;
	m_current  = -1;
	m_op_open  = false;
	m_depth    = 0;
	m_recording.store(nullptr, std::memory_order_release);
}

void GpuTimestamps::BeginOp(vk::CommandBuffer command, Kind kind, uint64_t hash, uint64_t hash2) {
	if (!Valid()) return;
	std::lock_guard lock(m_mutex);
	// Nested scopes (a copy helper that issues its own barrier) fold into the outermost one.
	if (m_depth++ != 0) return;
	m_op_open = false;
	if (m_current < 0) return;
	auto& slot = m_slots[static_cast<size_t>(m_current)];
	if (slot.used + 2 > QueriesPerSlot) {
		++slot.overflow;
		return;
	}
	const uint32_t query = static_cast<uint32_t>(m_current) * QueriesPerSlot + slot.used;
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_pool, query);
	slot.ops.push_back({hash, hash2, slot.used, kind});
	m_op_open = true;
}

void GpuTimestamps::EndOp(vk::CommandBuffer command) {
	if (!Valid()) return;
	std::lock_guard lock(m_mutex);
	if (m_depth == 0 || --m_depth != 0) return;
	if (!m_op_open || m_current < 0) return;
	auto&          slot  = m_slots[static_cast<size_t>(m_current)];
	const uint32_t query = static_cast<uint32_t>(m_current) * QueriesPerSlot + slot.used + 1;
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_pool, query);
	slot.used += 2;
	m_op_open = false;
}

void GpuTimestamps::Harvest(uint64_t completed_tick) {
	if (!Valid()) return;
	std::lock_guard lock(m_mutex);
	HarvestLocked(completed_tick);
}

void GpuTimestamps::HarvestLocked(uint64_t completed_tick) {
	for (uint32_t i = 0; i < SlotCount; ++i) {
		auto& slot = m_slots[i];
		if (slot.state != SlotState::Pending || slot.tick > completed_tick) continue;
		const auto result = m_graphics.device.getQueryPoolResults(
		    m_pool, i * QueriesPerSlot, slot.used, slot.used * sizeof(uint64_t), m_results.data(),
		    sizeof(uint64_t), vk::QueryResultFlagBits::e64);
		if (result != vk::Result::eSuccess) {
			continue; // not available yet; try again at the next harvest
		}
		const auto delta = [this](uint64_t begin, uint64_t end) {
			if (end < begin) ++m_acc.violations;
			return static_cast<uint64_t>(static_cast<double>((end - begin) & m_mask) * m_period_ns);
		};
		const uint64_t span = delta(m_results[0], m_results[1]);
		uint64_t       sum  = 0;
		for (const auto& op : slot.ops) {
			const uint64_t ns = delta(m_results[op.query], m_results[op.query + 1]);
			sum += ns;
			if (HasShader(op.kind) || op.hash != 0) {
				auto& shader = m_acc.shaders[{static_cast<uint8_t>(op.kind), op.hash, op.hash2}];
				shader.ns += ns;
				++shader.count;
			}
			auto& kind = m_acc.kinds[static_cast<size_t>(op.kind)];
			kind.ns += ns;
			++kind.count;
		}
		++m_acc.ticks;
		m_acc.span_ns += span;
		m_acc.ops += slot.ops.size();
		m_acc.attributed_ns += sum;
		m_acc.unattributed_ns += span > sum ? span - sum : 0;
		m_acc.overflow += slot.overflow;
		slot.state = SlotState::Free;
		slot.ops.clear();
	}
}

GpuTimestamps::Snapshot GpuTimestamps::Take(uint64_t completed_tick) {
	std::lock_guard lock(m_mutex);
	if (Valid()) HarvestLocked(completed_tick);
	Snapshot out = std::move(m_acc);
	m_acc        = Snapshot {};
	return out;
}

std::string GpuTimestamps::Format(const Snapshot& s) {
	char buffer[256];
	std::snprintf(buffer, sizeof(buffer),
	              "GpuTime: ticks %" PRIu64 " span %.2f ms ops %" PRIu64
	              " attributed %.2f ms unattributed %.2f ms overflow %" PRIu64 " noslot %" PRIu64
	              " | top:",
	              s.ticks, static_cast<double>(s.span_ns) / 1e6, s.ops,
	              static_cast<double>(s.attributed_ns) / 1e6,
	              static_cast<double>(s.unattributed_ns) / 1e6, s.overflow, s.no_slot);
	std::string line = buffer;
	using Item       = std::pair<std::tuple<uint8_t, uint64_t, uint64_t>, Entry>;
	std::vector<Item> items(s.shaders.begin(), s.shaders.end());
	std::sort(items.begin(), items.end(),
	          [](const Item& a, const Item& b) { return a.second.ns > b.second.ns; });
	if (items.size() > TopEntries) items.resize(TopEntries);
	for (const auto& [key, entry]: items) {
		const auto kind = std::get<0>(key);
		if (kind == static_cast<uint8_t>(Kind::Draw)) {
			std::snprintf(buffer, sizeof(buffer),
			              " %016" PRIx64 "/%016" PRIx64 " %s %.2f (%" PRIu64 ")",
			              std::get<1>(key), std::get<2>(key), KindNameOf(kind),
			              static_cast<double>(entry.ns) / 1e6, entry.count);
		} else {
			std::snprintf(buffer, sizeof(buffer), " %016" PRIx64 " %s %.2f (%" PRIu64 ")",
			              std::get<1>(key), KindNameOf(kind), static_cast<double>(entry.ns) / 1e6,
			              entry.count);
		}
		line += buffer;
	}
	// Everything that is not a draw/dispatch, sorted by time: cost per command = ms / count.
	std::vector<std::pair<size_t, Entry>> others;
	for (size_t k = 0; k < s.kinds.size(); ++k) {
		if (!HasShader(static_cast<Kind>(k)) && s.kinds[k].count != 0) {
			others.push_back({k, s.kinds[k]});
		}
	}
	std::sort(others.begin(), others.end(),
	          [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
	line += " | other:";
	for (const auto& [k, entry]: others) {
		std::snprintf(buffer, sizeof(buffer), " %s %.2f (%" PRIu64 ")",
		              KindNameOf(static_cast<uint8_t>(k)), static_cast<double>(entry.ns) / 1e6,
		              entry.count);
		line += buffer;
	}
	for (size_t k = 0; k < s.kinds.size(); ++k) {
		if (HasShader(static_cast<Kind>(k))) {
			std::snprintf(buffer, sizeof(buffer), " | %s %.2f (%" PRIu64 ")",
			              KindNameOf(static_cast<uint8_t>(k)),
			              static_cast<double>(s.kinds[k].ns) / 1e6, s.kinds[k].count);
			line += buffer;
		}
	}
	line += "\n";
	return line;
}

void GpuTimestamps::Dump(uint64_t completed_tick, double interval_seconds) {
	if (!Valid()) return;
	const auto now = std::chrono::steady_clock::now();
	{
		std::lock_guard lock(m_mutex);
		if (std::chrono::duration<double>(now - m_last_dump).count() < interval_seconds) return;
		m_last_dump = now;
	}
	const auto snapshot = Take(completed_tick);
	if (snapshot.ticks == 0 && snapshot.no_slot == 0) return;
	const auto line = Format(snapshot);
	LOGF("%s", line.c_str());
}

GpuTimestampScope::GpuTimestampScope(CommandScheduler& scheduler, vk::CommandBuffer command,
                                     GpuTimestamps::Kind kind, uint64_t hash,
                                     uint64_t hash2) noexcept
    : m_command(command) {
	if (!GpuTimestamps::Enabled()) return;
	auto* timestamps = scheduler.Timestamps();
	if (timestamps == nullptr || !timestamps->Valid()) return;
	m_timestamps = timestamps;
	m_timestamps->BeginOp(command, kind, hash, hash2);
}

GpuTimestampScope::GpuTimestampScope(vk::CommandBuffer command, GpuTimestamps::Kind kind,
                                     uint64_t hash, uint64_t hash2) noexcept
    : m_command(command) {
	if (!GpuTimestamps::Enabled()) return;
	auto* timestamps = GpuTimestamps::ForCommand(command);
	if (timestamps == nullptr) return;
	m_timestamps = timestamps;
	m_timestamps->BeginOp(command, kind, hash, hash2);
}

GpuTimestampScope::~GpuTimestampScope() {
	if (m_timestamps != nullptr) m_timestamps->EndOp(m_command);
}

} // namespace Libs::Graphics
