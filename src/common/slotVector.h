#ifndef EMULATOR_SRC_COMMON_SLOTVECTOR_H_
#define EMULATOR_SRC_COMMON_SLOTVECTOR_H_

#include "common/assert.h"
#include "common/abi.h"
#include "common/common.h"

#include <compare>
#include <cstdint>
#include <memory>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace Common {

struct SlotId {
	static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();

	constexpr SlotId() noexcept = default;
	constexpr SlotId(uint32_t value) noexcept: index(value), generation(1) {}
	constexpr SlotId(uint32_t value, uint32_t slot_generation) noexcept
	    : index(value), generation(slot_generation) {}

	[[nodiscard]] constexpr explicit operator bool() const noexcept { return index != INVALID_INDEX; }
	constexpr auto operator<=>(const SlotId&) const noexcept = default;

	uint32_t index = INVALID_INDEX;
	uint32_t generation = 0;
};

// Stable-address slot storage for cache resources that are intentionally non-movable.
template <typename T>
class SlotVector {
public:
	SlotVector() = default;
	KYTY_CLASS_NO_COPY(SlotVector);

	[[nodiscard]] T& operator[](SlotId id) noexcept {
		EXIT_IF(!is_allocated(id));
		return *SlotAt(id.index).value;
	}

	[[nodiscard]] const T& operator[](SlotId id) const noexcept {
		EXIT_IF(!is_allocated(id));
		return *SlotAt(id.index).value;
	}

	[[nodiscard]] T* try_get(SlotId id) noexcept {
		return is_allocated(id) ? &*SlotAt(id.index).value : nullptr;
	}

	[[nodiscard]] const T* try_get(SlotId id) const noexcept {
		return is_allocated(id) ? &*SlotAt(id.index).value : nullptr;
	}

	[[nodiscard]] bool is_allocated(SlotId id) const noexcept {
		if (!id || id.index >= m_slot_count) return false;
		const Slot& slot = SlotAt(id.index);
		return slot.generation == id.generation && slot.value.has_value();
	}

	template <typename... Args>
	[[nodiscard]] SlotId insert(Args&&... args) {
		uint32_t index = 0;
		if (m_free_list.empty()) {
			index = m_slot_count;
			if ((index & PageMask) == 0) {
				m_pages.push_back(std::make_unique<Slot[]>(PageSize));
			}
			++m_slot_count;
			SlotAt(index).value.emplace(std::forward<Args>(args)...);
		} else {
			index = m_free_list.back();
			m_free_list.pop_back();
			EXIT_IF(SlotAt(index).value.has_value());
			SlotAt(index).value.emplace(std::forward<Args>(args)...);
		}
		++m_size;
		return SlotId {index, SlotAt(index).generation};
	}

	void erase(SlotId id) noexcept {
		EXIT_IF(!is_allocated(id));
		auto& slot = SlotAt(id.index);
		slot.value.reset();
		if (++slot.generation == 0) {
			slot.generation = 1;
		}
		m_free_list.push_back(id.index);
		--m_size;
	}

	[[nodiscard]] size_t size() const noexcept { return m_size; }
	[[nodiscard]] size_t capacity() const noexcept { return m_slot_count; }

	template <typename F>
	void ForEach(F&& fn) {
		for (uint32_t index = 0; index < m_slot_count; ++index) {
			auto& slot = SlotAt(index);
			if (slot.value) {
				fn(SlotId {index, slot.generation}, *slot.value);
			}
		}
	}

	template <typename F>
	void ForEach(F&& fn) const {
		for (uint32_t index = 0; index < m_slot_count; ++index) {
			auto& slot = SlotAt(index);
			if (slot.value) {
				fn(SlotId {index, slot.generation}, *slot.value);
			}
		}
	}

private:
	struct Slot {
		std::optional<T> value;
		uint32_t         generation = 1;
	};

	// Fixed 4096-slot pages: stable addresses, two plain loads per lookup (no deque iterator math).
	static constexpr uint32_t PageBits = 12;
	static constexpr uint32_t PageSize = uint32_t {1} << PageBits;
	static constexpr uint32_t PageMask = PageSize - 1;

	[[nodiscard]] Slot& SlotAt(uint32_t index) noexcept {
		return m_pages[index >> PageBits][index & PageMask];
	}
	[[nodiscard]] const Slot& SlotAt(uint32_t index) const noexcept {
		return m_pages[index >> PageBits][index & PageMask];
	}

	std::vector<std::unique_ptr<Slot[]>> m_pages;
	uint32_t                     m_slot_count = 0;
	std::vector<uint32_t>        m_free_list;
	size_t                       m_size = 0;
};

} // namespace Common

template <>
struct std::hash<Common::SlotId> {
	[[nodiscard]] size_t operator()(Common::SlotId id) const noexcept {
		return std::hash<uint64_t> {}((static_cast<uint64_t>(id.generation) << 32u) | id.index);
	}
};

#endif // EMULATOR_SRC_COMMON_SLOTVECTOR_H_
