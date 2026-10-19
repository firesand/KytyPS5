#include "common/hostException.h"
#include "common/slotVector.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/gpuWriteTicks.h"
#include "graphics/host_gpu/writeWatch.h"

#include <algorithm>
#include <functional>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach-o/dyld.h>
#include <csignal>
#include <limits.h>
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#include <csignal>
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using Libs::Graphics::GuestRange;
using Libs::Graphics::MemoryTracker;
using Libs::Graphics::PageManager;
using Libs::Graphics::RangeSet;
using Libs::Graphics::TRACKER_ADDRESS_SIZE;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "MemoryTrackerTests: failed: %s\n", text);
    std::abort();
  }
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
using DWORD = uint32_t;
constexpr uint32_t PAGE_NOACCESS = 1;
constexpr uint32_t PAGE_READONLY = 2;
constexpr uint32_t PAGE_READWRITE = 3;
constexpr uint32_t MEM_RESERVE = 0;
constexpr uint32_t MEM_COMMIT = 0;
constexpr uint32_t MEM_RELEASE = 0;

int ToHostProt(uint32_t protection) {
  switch (protection) {
  case PAGE_NOACCESS:
    return PROT_NONE;
  case PAGE_READONLY:
    return PROT_READ;
  default:
    return PROT_READ | PROT_WRITE;
  }
}

uint32_t Protection(const void *address) {
#if defined(__APPLE__)
  mach_vm_address_t region_address =
      reinterpret_cast<mach_vm_address_t>(address);
  mach_vm_size_t region_size = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object_name = MACH_PORT_NULL;
  Check(mach_vm_region(mach_task_self(), &region_address, &region_size,
                       VM_REGION_BASIC_INFO_64,
                       reinterpret_cast<vm_region_info_t>(&info), &info_count,
                       &object_name) == KERN_SUCCESS,
        "mach_vm_region failed");
  if (object_name != MACH_PORT_NULL) {
    mach_port_deallocate(mach_task_self(), object_name);
  }
  return (info.protection & VM_PROT_WRITE) != 0
             ? PAGE_READWRITE
             : (info.protection & VM_PROT_READ) != 0 ? PAGE_READONLY
                                                     : PAGE_NOACCESS;
#else
  const auto addr = reinterpret_cast<uintptr_t>(address);
  std::FILE *maps = std::fopen("/proc/self/maps", "r");
  Check(maps != nullptr, "open /proc/self/maps failed");
  char line[512];
  uint32_t result = 0;
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    unsigned long start = 0;
    unsigned long end = 0;
    char perms[8]{};
    if (std::sscanf(line, "%lx-%lx %7s", &start, &end, perms) != 3) {
      continue;
    }
    if (addr >= start && addr < end) {
      result = perms[1] == 'w'   ? PAGE_READWRITE
               : perms[0] == 'r' ? PAGE_READONLY
                                 : PAGE_NOACCESS;
      break;
    }
  }
  std::fclose(maps);
  return result;
#endif
}

std::map<void *, size_t> &AllocationSizes() {
  static std::map<void *, size_t> sizes;
  return sizes;
}

void *VirtualAlloc(void *address, size_t size, DWORD, uint32_t protection) {
#if defined(__APPLE__)
  mach_vm_address_t raw_address = reinterpret_cast<mach_vm_address_t>(address);
  const auto flags = address != nullptr ? VM_FLAGS_FIXED : VM_FLAGS_ANYWHERE;
  if (mach_vm_allocate(mach_task_self(), &raw_address, size, flags) !=
      KERN_SUCCESS) {
    return nullptr;
  }
  if (mach_vm_protect(mach_task_self(), raw_address, size, false,
                      static_cast<vm_prot_t>(ToHostProt(protection))) !=
      KERN_SUCCESS) {
    mach_vm_deallocate(mach_task_self(), raw_address, size);
    return nullptr;
  }
  void *raw = reinterpret_cast<void *>(raw_address);
  AllocationSizes()[raw] = size;
  return raw;
#else
  const int extra = address != nullptr ? MAP_FIXED_NOREPLACE : 0;
  void *raw = ::mmap(address, size, ToHostProt(protection),
                     MAP_PRIVATE | MAP_ANONYMOUS | extra, -1, 0);
  if (raw == MAP_FAILED) {
    return nullptr;
  }
  AllocationSizes()[raw] = size;
  return raw;
#endif
}

int VirtualFree(void *address, size_t, DWORD) {
  auto &sizes = AllocationSizes();
  auto it = sizes.find(address);
  if (it == sizes.end()) {
    return 0;
  }
#if defined(__APPLE__)
  const int ok = mach_vm_deallocate(mach_task_self(),
                                    reinterpret_cast<mach_vm_address_t>(address),
                                    it->second) == KERN_SUCCESS
                     ? 1
                     : 0;
#else
  const int ok = ::munmap(address, it->second) == 0 ? 1 : 0;
#endif
  sizes.erase(it);
  return ok;
}

int VirtualProtect(void *address, size_t size, uint32_t protection,
                   DWORD *old_protection) {
  if (old_protection != nullptr) {
    *old_protection = Protection(address);
  }
#if defined(__APPLE__)
  return mach_vm_protect(mach_task_self(),
                         reinterpret_cast<mach_vm_address_t>(address), size,
                         false, static_cast<vm_prot_t>(ToHostProt(protection))) ==
                 KERN_SUCCESS
             ? 1
             : 0;
#else
  return ::mprotect(address, size, ToHostProt(protection)) == 0 ? 1 : 0;
#endif
}
#else
uint32_t Protection(const void *address) {
  MEMORY_BASIC_INFORMATION info{};
  Check(VirtualQuery(address, &info, sizeof(info)) != 0, "VirtualQuery failed");
  return info.Protect;
}
#endif

bool IsWritable(const void *address) {
  return Protection(address) == PAGE_READWRITE;
}

uint64_t g_protection_calls = 0;

struct ProtectionCall {
  uint64_t address;
  uint64_t size;
  Common::VirtualMemory::Mode mode;
};

std::vector<ProtectionCall> g_protection_log;
std::mutex g_protection_log_mutex;

void ResetProtectionLog() {
  g_protection_calls = 0;
  g_protection_log.clear();
}

bool ProtectAddressSpace(uint64_t vaddr, uint64_t size,
                         Common::VirtualMemory::Mode mode) {
  uint32_t protection = PAGE_NOACCESS;
  if (mode == Common::VirtualMemory::Mode::Read) {
    protection = PAGE_READONLY;
  } else if (mode == Common::VirtualMemory::Mode::ReadWrite) {
    protection = PAGE_READWRITE;
  }
  DWORD old_protection = 0;
  {
    std::lock_guard lock(g_protection_log_mutex);
    g_protection_calls++;
    g_protection_log.push_back({vaddr, size, mode});
  }
  return VirtualProtect(reinterpret_cast<void *>(vaddr), size, protection,
                        &old_protection) != 0;
}

struct TrackerHarness {
  TrackerHarness() : tracker(page_manager) {}

  PageManager page_manager;
  MemoryTracker tracker;
};

uint8_t *AllocateFixedGuestRange(uint64_t size, uintptr_t offset) {
  // Keep the mapping inside the tracker's guest address space and preserve the
  // requested region alignment. A fixed address can be occupied by the host
  // process (notably by the macOS runner's ASLR layout).
  constexpr uintptr_t first_base = 0x0000000200000000ull;
  constexpr uintptr_t stride = 0x0000000100000000ull;
  for (uintptr_t attempt = 0; attempt < 256; attempt++) {
    auto *wanted = reinterpret_cast<void *>(first_base + offset + attempt * stride);
    auto *memory = static_cast<uint8_t *>(
        VirtualAlloc(wanted, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (memory == wanted) {
      return memory;
    }
    if (memory != nullptr) {
      Check(VirtualFree(memory, 0, MEM_RELEASE) != 0,
            "release unexpected fixed allocation failed");
    }
  }
  Check(false, "no free fixed guest address found");
  return nullptr;
}

uint8_t *Allocate(PageManager &manager, uint64_t pages) {
  const auto size = manager.GetPageSize() * pages;
  return AllocateFixedGuestRange(size, 0x10000);
}

void Release(uint8_t *memory) {
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestRangeSet() {
  RangeSet ranges;
  ranges.Add(0x1000, 0x80);
  ranges.Add(0x1080, 0x80);
  ranges.Add(0x1200, 0x40);
  Check(ranges.Contains(0x1010, 0xe0) && !ranges.Contains(0x1010, 0x200),
        "range set containment did not require full coverage");
  Check(ranges.Intersects(0x0fff, 2) && ranges.Intersects(0x11ff, 2) &&
            !ranges.Intersects(0x1100, 0x100) &&
            !ranges.Intersects(0x1240, 1),
        "range set intersection did not preserve half-open boundaries");
  std::vector<std::pair<uint64_t, uint64_t>> intersections;
  ranges.ForEachInRange(0x1070, 0x1b0, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1070 &&
            intersections[0].second == 0x1100 &&
            intersections[1].first == 0x1200 && intersections[1].second == 0x1220,
        "range set did not merge and intersect exact byte ranges");
  ranges.Subtract(0x1040, 0x1e0);
  intersections.clear();
  ranges.ForEachInRange(0x1000, 0x300, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1000 &&
            intersections[0].second == 0x1040 &&
            intersections[1].first == 0x1220 && intersections[1].second == 0x1240,
        "range set subtraction did not preserve both exact tails");
}

void TestGuestRange() {
  constexpr GuestRange empty{};
  constexpr GuestRange first_byte{1, 1};
  constexpr uint64_t extended_end = Libs::LibKernel::Memory::kExtendedMemoryBase +
                                    Libs::LibKernel::Memory::kExtendedMemorySize;
  constexpr GuestRange last_byte{extended_end - 1, 1};

  static_assert(empty.Empty() && !empty.Valid() && empty.ValidOrEmpty());
  static_assert(!first_byte.Empty() && first_byte.Valid() &&
                first_byte.ValidOrEmpty() && first_byte.End() == 2);
  static_assert(last_byte.Valid() && last_byte.End() == extended_end);

  Check(!GuestRange{0, 1}.Empty() && !GuestRange{0, 1}.ValidOrEmpty(),
        "zero-address nonempty guest range is rejected");
  Check(!GuestRange{1, 0}.Empty() && !GuestRange{1, 0}.ValidOrEmpty(),
        "nonzero-address empty guest range is rejected");
  Check(GuestRange{Libs::LibKernel::Memory::kExtendedMemoryBase, 1}.Valid() &&
            !GuestRange{Libs::Graphics::LOWER_ADDRESS_SIZE, 1}.Valid() &&
            !GuestRange{Libs::LibKernel::Memory::kExtendedMemoryBase - 1, 2}.Valid() &&
            !GuestRange{extended_end - 1, 2}.Valid(),
        "extended range and gap boundaries are enforced");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE, 1}.Valid(),
        "first address beyond the guest range is rejected");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE - 1, 2}.Valid(),
        "guest range crossing the address-space end is rejected");
  Check(!GuestRange{UINT64_MAX, 2}.Valid(), "wrapping guest range is rejected");
}

void TestQueriesDoNotRequireMappedOwnership() {
  constexpr uint64_t address = 0x0000000203000000ull;
  TrackerHarness harness;
  const auto page_size = harness.page_manager.GetPageSize();
  Check(harness.tracker.IsRegionCpuModified(address, page_size) &&
            !harness.tracker.IsRegionGpuModified(address, page_size),
        "unowned tracker range did not expose its initial CPU-dirty state");
}

void TestConcurrentRegionPublication() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  std::binary_semaphore start_first{0};
  std::binary_semaphore start_second{0};
  std::atomic_uint32_t cpu_dirty_results{0};
  std::jthread first([&] {
    start_first.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::jthread second([&] {
    start_second.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  start_first.release();
  start_second.release();
  first.join();
  second.join();

  tracker.UntrackMemory(address, page_size);
  Release(memory);
  Check(cpu_dirty_results.load(std::memory_order_relaxed) == 2,
        "concurrent region publication lost initial CPU ownership");
}

void TestCpuDirtyUpload() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  Check(tracker.IsRegionCpuModified(address + 16, 32),
        "new region was not CPU dirty");

  uint32_t ranges = 0;
  bool uploaded = false;
  tracker.ForEachUploadRange(
      address + 16, 32, false,
      [&](uint64_t upload_address, uint64_t upload_size, const void*) noexcept {
        Check(upload_address == address && upload_size == page_size,
              "upload range was not page aligned");
        ranges++;
      },
      [&]() noexcept { uploaded = true; });
  Check(ranges == 1 && uploaded &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "upload did not clear CPU dirty state and arm protection");

  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(tracker.IsRegionCpuModified(address, page_size) && IsWritable(memory),
        "explicit CPU dirtiness did not release write protection");
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestCleanUploadPreservesOwnership() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(region_size * 2, region_size);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(address, region_size * 2, false,
                             [](uint64_t, uint64_t, const void*) noexcept {},
                             []() noexcept {});

  for (const auto start : {address + 63 * page_size,
                           address + region_size - page_size}) {
    const auto before = start - page_size;
    const auto after = start + 2 * page_size;
    tracker.MarkRegionAsCpuModified(before, page_size);
    tracker.MarkRegionAsCpuModified(after, page_size);
    uint32_t ranges = 0;
    uint32_t completions = 0;
    ResetProtectionLog();
    tracker.ForEachUploadRange(
        start, 2 * page_size, false,
        [&](uint64_t, uint64_t, const void*) noexcept { ranges++; },
        [&]() noexcept { completions++; });
    Check(ranges == 0 && completions == 1 && g_protection_calls == 0 &&
              !tracker.IsRegionCpuModified(start, 2 * page_size),
          "clean read-only upload changed dirty state or protection");

    tracker.ForEachUploadRange(
        start, 2 * page_size, true,
        [&](uint64_t, uint64_t, const void*) noexcept { ranges++; },
        [&]() noexcept { completions++; });
    Check(ranges == 0 && completions == 2 &&
              tracker.IsRegionGpuModified(start, page_size) &&
              tracker.IsRegionGpuModified(start + page_size, page_size) &&
              Protection(reinterpret_cast<void *>(start)) == PAGE_NOACCESS &&
              Protection(reinterpret_cast<void *>(start + page_size)) ==
                  PAGE_NOACCESS &&
              tracker.IsRegionCpuModified(before, page_size) &&
              tracker.IsRegionCpuModified(after, page_size) &&
              !tracker.IsRegionGpuModified(before, page_size) &&
              !tracker.IsRegionGpuModified(after, page_size) &&
              IsWritable(reinterpret_cast<void *>(before)) &&
              IsWritable(reinterpret_cast<void *>(after)),
          "clean written upload lost GPU ownership or changed dirty neighbors");
    tracker.UnmarkRegionAsGpuModified(start, 2 * page_size);
  }
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

void BenchmarkCleanUploads() {
  constexpr uint64_t size = 256ull * 1024 * 1024;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(size, 0);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(address, size, false,
                             [](uint64_t, uint64_t, const void*) noexcept {},
                             []() noexcept {});
  std::puts("chunk_bytes,sweeps,calls,elapsed_ns,ns_per_sweep,ns_per_call");
  for (const uint64_t chunk : {size, Libs::Graphics::TRACKER_REGION_SIZE,
                               uint64_t{64 * 1024}}) {
    uint64_t ranges = 0;
    uint64_t completions = 0;
    uint64_t sweeps = 0;
    ResetProtectionLog();
    const auto start = std::chrono::steady_clock::now();
    std::chrono::nanoseconds elapsed{};
    do {
      for (uint64_t offset = 0; offset < size; offset += chunk) {
        tracker.ForEachUploadRange(
            address + offset, chunk, false,
            [&](uint64_t, uint64_t, const void*) noexcept { ranges++; },
            [&]() noexcept { completions++; });
      }
      sweeps++;
      elapsed = std::chrono::steady_clock::now() - start;
    } while (elapsed < std::chrono::milliseconds(250));
    Check(ranges == 0 && completions == sweeps * (size / chunk) &&
              g_protection_calls == 0,
          "clean upload benchmark changed ranges, completion, or protection");
    std::printf("%llu,%llu,%llu,%lld,%.2f,%.2f\n",
                static_cast<unsigned long long>(chunk),
                static_cast<unsigned long long>(sweeps),
                static_cast<unsigned long long>(completions),
                static_cast<long long>(elapsed.count()),
                static_cast<double>(elapsed.count()) / sweeps,
                static_cast<double>(elapsed.count()) / completions);
  }
  tracker.UntrackMemory(address, size);
  Release(memory);
}

void TestRangeInvalidation() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t size = Libs::Graphics::TRACKER_REGION_SIZE * 2;
  auto *memory = AllocateFixedGuestRange(size, 0x1000000);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, size, true, [](uint64_t, uint64_t, const void*) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, size) && !IsWritable(memory),
        "range invalidation setup did not establish GPU ownership");
  uint32_t flushes = 0;
  tracker.InvalidateRegion(address + 16, size - 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, size - 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, size - 32);
  });
  Check(flushes == 1 && !tracker.IsRegionGpuModified(address, size) &&
            tracker.IsRegionCpuModified(address, size) && IsWritable(memory) &&
            IsWritable(memory + size - 1),
        "range invalidation did not batch ownership transfer across regions");
  tracker.InvalidateRegion(address + 16, size - 32, [&] { flushes++; });
  Check(flushes == 1,
        "clean range invalidation unnecessarily requested a GPU flush");
  tracker.UntrackMemory(address, size);
  Release(memory);
}

void TestGpuReacquisitionAfterInvalidation() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t, const void*) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size),
        "reacquisition setup did not establish GPU ownership");

  uint32_t flushes = 0;
  uint32_t uploads = 0;
  std::binary_semaphore reacquire{0};
  std::binary_semaphore reacquired{0};
  std::jthread publisher([&] {
    reacquire.acquire();
    tracker.ForEachUploadRange(
        address + 16, 32, true,
        [&](uint64_t, uint64_t, const void*) noexcept { uploads++; }, []() noexcept {});
    reacquired.release();
  });
  tracker.InvalidateRegion(address + 16, 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, 32);
    reacquire.release();
    reacquired.acquire();
  });
  publisher.join();
  Check(flushes == 1 && uploads == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            !IsWritable(memory),
        "invalidation rejected a new generation of GPU ownership");

  tracker.UnmarkRegionAsGpuModified(address, page_size);
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDirtyBits() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t, const void*) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionGpuModified(address + page_size, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "GPU dirty state escaped the requested range");
  tracker.UnmarkRegionAsGpuModified(address, page_size);
  Check(!tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "GPU dirty state did not restore write-only tracking");
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestExactDirtyIntervalsSharingTrackerPage() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, false, [](uint64_t, uint64_t, const void*) noexcept {},
      []() noexcept {});
  RangeSet exact_dirty;
  exact_dirty.Add(address + 64, 16);
  exact_dirty.Add(address + 192, 32);

  ResetProtectionLog();
  tracker.MarkRegionAsGpuModified(address + 64, 16);
  tracker.MarkRegionAsGpuModified(address + 192, 32);
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "disjoint byte dirtiness duplicated the page watcher");

  exact_dirty.Subtract(address + 64, 16);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "draining one exact interval prematurely released its shared page");

  exact_dirty.Subtract(address + 192, 32);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 2 &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "draining the final exact interval did not release its tracker page");

  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDownloadProtectionMirrors() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size * 4, false, [](uint64_t, uint64_t, const void*) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(address + 16, 32);
  tracker.MarkRegionAsGpuModified(address + page_size * 2 + 16, 32);

  std::vector<std::pair<uint64_t, uint64_t>> visited;
  ResetProtectionLog();
  tracker.ForEachDownloadRange<false>(
      address, page_size * 3,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.size() == 2 && visited[0].first == address &&
            visited[0].second == page_size &&
            visited[1].first == address + page_size * 2 &&
            visited[1].second == page_size && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address, page_size * 3),
        "non-clearing download changed protection or lost sparse ranges");

  visited.clear();
  bool protected_during_download = false;
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        protected_during_download = Protection(memory) == PAGE_NOACCESS;
        visited.push_back({range_address, range_size});
      });
  Check(protected_during_download && visited.size() == 1 &&
            visited[0].first == address &&
            visited[0].second == page_size && g_protection_log.size() == 1 &&
            g_protection_log[0].address == address &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size) &&
            Protection(memory) == PAGE_READONLY &&
            Protection(memory + page_size * 2) == PAGE_NOACCESS,
        "partial download did not preserve the CPU/GPU protection mirrors");

  visited.clear();
  ResetProtectionLog();
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.empty() && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size),
        "idempotent partial download disturbed another GPU-owned page");

  tracker.UnmarkRegionAsGpuModified(address, page_size * 3);
  Check(!tracker.IsRegionGpuModified(address, page_size * 3) &&
            Protection(memory + page_size * 2) == PAGE_READONLY,
        "broad final unmark did not restore write-only tracking");
  ResetProtectionLog();
  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(
      g_protection_log.size() == 1 && g_protection_log[0].address == address &&
          g_protection_log[0].size == page_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::ReadWrite &&
          IsWritable(memory) && !IsWritable(memory + page_size),
      "CPU-dirty transition did not release only its write watcher");

  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

void TestCrossRegionUpload() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  auto *memory = AllocateFixedGuestRange(region_size * 2, 0x10000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto boundary = (address + region_size - 1) & ~(region_size - 1);
  for (const bool is_written : {false, true}) {
    TrackerHarness harness;
    auto &tracker = harness.tracker;
    uint32_t ranges = 0;
    tracker.ForEachUploadRange(
        boundary - page_size, page_size * 2, is_written,
        [&](uint64_t start, uint64_t size, const void*) noexcept {
          Check(start == boundary - page_size + ranges * page_size &&
                    size == page_size,
                "cold upload did not visit both region boundaries exactly");
          ranges++;
        },
        [&]() noexcept {
          Check(ranges == 2 &&
                    Protection(reinterpret_cast<void *>(boundary - page_size)) ==
                        PAGE_READONLY &&
                    Protection(reinterpret_cast<void *>(boundary)) == PAGE_READONLY,
                "cold upload completed before clearing and protecting both regions");
        });
    for (const auto page : {boundary - page_size, boundary}) {
      Check(!tracker.IsRegionCpuModified(page, page_size) &&
                tracker.IsRegionGpuModified(page, page_size) == is_written &&
                Protection(reinterpret_cast<void *>(page)) ==
                    (is_written ? PAGE_NOACCESS : PAGE_READONLY),
            "cross-region upload did not preserve final ownership and protection");
    }
    Check(IsWritable(reinterpret_cast<void *>(boundary - 2 * page_size)) &&
              IsWritable(reinterpret_cast<void *>(boundary + page_size)),
          "cross-region upload changed neighboring pages");
    tracker.UnmarkRegionAsGpuModified(boundary - page_size, 2 * page_size);
    tracker.UntrackMemory(address, region_size * 2);
  }
  Release(memory);
}

void TestConcurrentColdUploads() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(region_size * 2, region_size);
  const auto address = reinterpret_cast<uint64_t>(memory);
  std::counting_semaphore<2> start{0};
  std::counting_semaphore<2> upload_entered{0};
  std::counting_semaphore<2> finish_upload{0};
  std::vector<std::jthread> workers;
  for (const auto page : {address, address + region_size}) {
    workers.emplace_back([&, page] {
      start.acquire();
      uint32_t ranges = 0;
      tracker.ForEachUploadRange(
          page, page_size, true,
          [&](uint64_t upload_address, uint64_t upload_size, const void*) noexcept {
            Check(upload_address == page && upload_size == page_size,
                  "concurrent cold upload lost its dirty page");
            ranges++;
          },
          [&]() noexcept {
            Check(ranges == 1, "concurrent cold upload skipped its dirty page");
            upload_entered.release();
            finish_upload.acquire();
          });
    });
  }
  start.release(2);
  const bool first_entered = upload_entered.try_acquire_for(std::chrono::seconds(5));
  const bool second_entered = upload_entered.try_acquire_for(std::chrono::seconds(5));
  finish_upload.release(2);
  for (auto &worker : workers) {
    worker.join();
  }
  Check(first_entered && second_entered,
        "cold writable uploads serialized disjoint regions");
  for (const auto page : {address, address + region_size}) {
    Check(!tracker.IsRegionCpuModified(page, page_size) &&
              tracker.IsRegionGpuModified(page, page_size) &&
              Protection(reinterpret_cast<void *>(page)) == PAGE_NOACCESS,
          "concurrent cold upload did not retain GPU ownership");
    tracker.UnmarkRegionAsGpuModified(page, page_size);
  }
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

void TestUploadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  // Publish both managers before the concurrent section so this test measures
  // tracker access serialization rather than manager allocation.
  Check(tracker.IsRegionCpuModified(allocation_base, page_size) &&
            tracker.IsRegionCpuModified(second_region, page_size),
        "disjoint upload setup did not initialize both regions");

  std::binary_semaphore upload_entered{0};
  std::binary_semaphore finish_upload{0};
  std::binary_semaphore query_finished{0};
  std::atomic_bool query_result{false};
  std::jthread uploader([&] {
    tracker.ForEachUploadRange(
        allocation_base, page_size, true, [](uint64_t, uint64_t, const void*) noexcept {},
        [&]() noexcept {
          upload_entered.release();
          finish_upload.acquire();
        });
  });
  upload_entered.acquire();
  std::jthread query([&] {
    query_result.store(tracker.IsRegionCpuModified(second_region, page_size),
                       std::memory_order_relaxed);
    query_finished.release();
  });

  const bool completed_while_upload_blocked =
      query_finished.try_acquire_for(std::chrono::seconds(5));
  finish_upload.release();
  uploader.join();
  query.join();

  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_upload_blocked &&
            query_result.load(std::memory_order_relaxed),
        "upload callback serialized an unrelated tracker region");
}

void TestDownloadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  tracker.ForEachUploadRange(
      allocation_base, page_size, true, [](uint64_t, uint64_t, const void*) noexcept {},
      []() noexcept {});
  tracker.ForEachUploadRange(
      second_region, page_size, false, [](uint64_t, uint64_t, const void*) noexcept {},
      []() noexcept {});

  std::binary_semaphore download_entered{0};
  std::binary_semaphore finish_download{0};
  std::binary_semaphore mutation_finished{0};
  std::jthread downloader([&] {
    tracker.ForEachDownloadRange<false>(
        allocation_base, second_region + page_size - allocation_base,
        [&](uint64_t address, uint64_t) noexcept {
          if (address == allocation_base) {
            download_entered.release();
            finish_download.acquire();
          }
        });
  });
  download_entered.acquire();
  std::jthread mutation([&] {
    tracker.MarkRegionAsGpuModified(second_region, page_size);
    mutation_finished.release();
  });

  const bool completed_while_download_blocked =
      mutation_finished.try_acquire_for(std::chrono::seconds(5));
  finish_download.release();
  downloader.join();
  mutation.join();

  const bool both_gpu_owned =
      tracker.IsRegionGpuModified(allocation_base, page_size) &&
      tracker.IsRegionGpuModified(second_region, page_size);
  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.UnmarkRegionAsGpuModified(second_region, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(second_region, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_download_blocked && both_gpu_owned,
        "download callback serialized an unrelated tracker region");
}

void TestGpuUnmarkUsesRegionMask() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size + page_size <=
            allocation_base + region_size * 2,
        "test allocation does not span two complete tracker regions");

  const auto sparse_begin = region_base + page_size;
  tracker.ForEachUploadRange(
      sparse_begin, page_size * 3, false, [](uint64_t, uint64_t, const void*) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(sparse_begin, page_size);
  tracker.MarkRegionAsGpuModified(sparse_begin + page_size * 2, page_size);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(
      g_protection_calls == 1 && g_protection_log.size() == 1 &&
          g_protection_log[0].address == sparse_begin &&
          g_protection_log[0].size == page_size * 3 &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(sparse_begin, page_size * 3) &&
          Protection(reinterpret_cast<void *>(sparse_begin)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size)) ==
              PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size * 2)) ==
              PAGE_READONLY,
      "GPU unmark did not coalesce a sparse 4 MiB region mask");
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(g_protection_calls == 0,
        "idempotent GPU unmark performed a protection call");

  const auto boundary = region_base + region_size;
  const auto cross_begin = boundary - page_size;
  tracker.ForEachUploadRange(
      cross_begin, page_size * 2, false, [](uint64_t, uint64_t, const void*) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(cross_begin, page_size * 2);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(cross_begin, page_size * 2);
  Check(g_protection_calls == 2 && g_protection_log.size() == 2 &&
            g_protection_log[0].address == cross_begin &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            g_protection_log[1].address == boundary &&
            g_protection_log[1].size == page_size &&
            g_protection_log[1].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(cross_begin, page_size * 2),
        "cross-region GPU unmark did not use one update per 4 MiB region");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

void TestFullRegionGpuUnmarkBatching() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size <= allocation_base + region_size * 2,
        "test allocation does not contain a complete tracker region");

  tracker.ForEachUploadRange(
      region_base, region_size, false, [](uint64_t, uint64_t, const void*) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(region_base, region_size);
  Check(tracker.IsRegionGpuModified(region_base, region_size) &&
            Protection(reinterpret_cast<void *>(region_base)) ==
                PAGE_NOACCESS &&
            Protection(reinterpret_cast<void *>(region_base + region_size -
                                                page_size)) == PAGE_NOACCESS,
        "full-region setup did not establish GPU read protection");

  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(region_base, region_size);
  Check(
      g_protection_log.size() == 1 &&
          g_protection_log[0].address == region_base &&
          g_protection_log[0].size == region_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(region_base, region_size) &&
          Protection(reinterpret_cast<void *>(region_base)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(region_base + region_size -
                                              page_size)) == PAGE_READONLY,
      "full-region GPU unmark did not use one exact 4 MiB protection request");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

[[noreturn]] void RunDeathCase(const char *name) {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);
  if (std::strcmp(name, "gpu-dirty-explicit-cpu") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t, const void*) noexcept {},
        []() noexcept {});
    tracker.MarkRegionAsCpuModified(address, page_size);
  } else if (std::strcmp(name, "reentrant-upload") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t, const void*) noexcept {},
        [&]() noexcept {
          (void)tracker.IsRegionCpuModified(address, page_size);
        });
  } else if (std::strcmp(name, "recursive-tracking-lock") == 0) {
    Libs::Graphics::TrackingSpinLock lock;
    lock.lock();
    lock.lock();
  } else if (std::strcmp(name, "non-owner-tracking-unlock") == 0) {
    Libs::Graphics::TrackingSpinLock lock;
    lock.lock();
    std::thread worker([&] { lock.unlock(); });
    worker.join();
  }
  std::_Exit(0x7f);
}

void CheckDeathCase(const char *name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  char path[MAX_PATH]{};
  Check(GetModuleFileNameA(nullptr, path, MAX_PATH) != 0,
        "GetModuleFileName failed");
  std::string command = std::string("\"") + path + "\" --death " + name;
  std::vector<char> mutable_command(command.begin(), command.end());
  mutable_command.push_back('\0');
  STARTUPINFOA startup{sizeof(startup)};
  PROCESS_INFORMATION process{};
  Check(CreateProcessA(nullptr, mutable_command.data(), nullptr, nullptr, FALSE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                       &process) != 0,
        "CreateProcess failed");
  Check(WaitForSingleObject(process.hProcess, 10000) == WAIT_OBJECT_0,
        "MemoryTracker death test timed out");
  DWORD exit_code = 0;
  Check(
      GetExitCodeProcess(process.hProcess, &exit_code) != 0 &&
          (exit_code == 321 || exit_code == EXCEPTION_NONCONTINUABLE_EXCEPTION),
      "MemoryTracker death path used the wrong exit");
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
#else
#if defined(__APPLE__)
  std::vector<char> path(PATH_MAX);
  uint32_t path_size = static_cast<uint32_t>(path.size());
  if (_NSGetExecutablePath(path.data(), &path_size) != 0) {
    path.resize(path_size);
    Check(_NSGetExecutablePath(path.data(), &path_size) == 0,
          "_NSGetExecutablePath failed");
  }
#endif
  const pid_t pid = ::fork();
  Check(pid >= 0, "fork failed");
  if (pid == 0) {
#if defined(__APPLE__)
    ::execl(path.data(), "MemoryTrackerTests", "--death", name, nullptr);
#else
    ::execl("/proc/self/exe", "MemoryTrackerTests", "--death", name, nullptr);
#endif
    std::_Exit(0x7e);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid, "waitpid failed");
  const bool fatal_exit =
      WIFEXITED(status) && WEXITSTATUS(status) == (321 & 0xff);
  Check(fatal_exit || WIFSIGNALED(status),
        "MemoryTracker death path used the wrong exit");
#endif
}

void TestFatalPaths() {
  for (const char *name : {"gpu-dirty-explicit-cpu", "reentrant-upload",
                           "recursive-tracking-lock", "non-owner-tracking-unlock"}) {
    CheckDeathCase(name);
  }
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
void *g_fault_stack = nullptr;
constexpr size_t FAULT_STACK_SIZE = 64 * 1024;
volatile sig_atomic_t g_stack_faults = 0;

bool HandleStackFault(const Common::HostException::ExceptionInfo &info) {
  using namespace Common::HostException;
  stack_t active_stack{};
  const auto fault_address = reinterpret_cast<uintptr_t>(g_fault_stack) +
                             FAULT_STACK_SIZE - sizeof(uintptr_t);
  if (info.type != ExceptionType::AccessViolation ||
      info.access_violation_type != AccessViolationType::Write ||
      info.access_violation_vaddr != fault_address ||
      ::sigaltstack(nullptr, &active_stack) != 0 ||
      (active_stack.ss_flags & SS_ONSTACK) == 0) {
    std::_Exit(1);
  }
  g_stack_faults = 1;
  return ::mprotect(g_fault_stack, FAULT_STACK_SIZE,
                    PROT_READ | PROT_WRITE) == 0;
}

// A stack write must fault before any signal frame can use the protected stack.
__attribute__((naked)) void WriteProtectedStack(void *) {
  asm volatile("mov %rsp, %rax\n"
               "mov %rdi, %rsp\n"
               "push %rax\n"
               "pop %rsp\n"
               "ret\n");
}

void TestFaultOnProtectedStack() {
  const pid_t pid = ::fork();
  Check(pid >= 0, "stack fault fork failed");
  if (pid == 0) {
    std::thread worker([] {
      Check(Common::HostException::InitializeThreadSignalStack(),
            "initialize thread signal stack failed");
      Check(Common::HostException::InstallHandler(HandleStackFault),
            "install stack fault handler failed");
      g_fault_stack = ::mmap(nullptr, FAULT_STACK_SIZE, PROT_READ,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      Check(g_fault_stack != MAP_FAILED, "allocate protected stack failed");
      WriteProtectedStack(static_cast<char *>(g_fault_stack) + FAULT_STACK_SIZE);
      Check(g_stack_faults == 1, "protected stack write did not resume");
      struct sigaction action{};
      Check(::sigaction(SIGSEGV, nullptr, &action) == 0 &&
                action.sa_handler != SIG_DFL,
            "stack fault reset the process handler");
      Check(::munmap(g_fault_stack, FAULT_STACK_SIZE) == 0,
            "release protected stack failed");
    });
    worker.join();
    std::_Exit(0);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
            WEXITSTATUS(status) == 0,
        "fault on a protected stack did not recover");
}
#endif

} // namespace

namespace Libs::LibKernel::Memory {

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size,
                            Common::VirtualMemory::Mode mode) {
  return ProtectAddressSpace(vaddr, size, mode);
}

} // namespace Libs::LibKernel::Memory

// Hot-page content hashing: the remembered hash must describe the bytes the GPU holds. Covers the
// A -> B -> A interleaving (guest write between hashing and copying), demotion followed by an
// unhashed upload and re-promotion with the old content, and the upload-from-snapshot contract
// (second review, 2026-10-12).
void TestHotPageSnapshotHash() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t page = Libs::Graphics::TRACKER_PAGE_SIZE;
  uint8_t *memory = Allocate(harness.page_manager, 16);
  const auto address = reinterpret_cast<uint64_t>(memory);
  std::vector<uint8_t> gpu(page, 0);
  struct Emitted { uint64_t address; uint64_t size; const void *snapshot; };
  std::vector<Emitted> emitted;
  // One sync: the "GPU" copies each emitted range from the snapshot when given, else from guest
  // memory. `between` runs after the ranges were emitted and before the copy (a racing writer).
  auto sync = [&](const std::function<void()> &between = {}) {
    emitted.clear();
    tracker.ForEachUploadRange(
        address, page, false,
        [&](uint64_t upload_address, uint64_t upload_size, const void *snapshot) noexcept {
          emitted.push_back({upload_address, upload_size, snapshot});
        },
        [&]() noexcept {
          if (between) between();
          for (const auto &range : emitted) {
            const auto *source = range.snapshot != nullptr
                                     ? static_cast<const uint8_t *>(range.snapshot)
                                     : reinterpret_cast<const uint8_t *>(range.address);
            std::memcpy(gpu.data() + (range.address - address), source, range.size);
          }
        });
  };
  auto fill = [&](uint8_t value) { std::memset(memory, value, page); };
  // The real CPU write-fault path (what the host fault handler runs), not MarkRegionAsCpuModified:
  // that helper also invalidates the page hashes and would mask a missing demotion invalidation.
  auto fault = [&] { tracker.InvalidateRegion(address, page, [] {}); };
  auto gpu_is = [&](uint8_t value) {
    return std::all_of(gpu.begin(), gpu.end(), [value](uint8_t b) { return b == value; });
  };

  // Track the page first (what buffer creation does), then promote it through the fault path:
  // KYTY_HOT_PAGE_FAULTS (default 8) CPU-dirty transitions.
  fill(0xA1);
  tracker.MarkRegionAsCpuModified(address, page);
  sync();
  for (int i = 0; i < 8; i++) {
    fault();
    sync();
  }
  Check(gpu_is(0xA1), "hot promotion uploads did not deliver the page");
  // Hot pages stay dirty: an unchanged page is skipped, a changed one uploaded.
  sync();
  Check(emitted.empty() && gpu_is(0xA1), "unchanged hot page was uploaded again");
  fill(0xB2);
  sync();
  Check(emitted.size() == 1 && emitted[0].snapshot != nullptr && gpu_is(0xB2),
        "changed hot page was not uploaded from a snapshot");

  // A -> B -> A: a guest write between the hash and the copy must not leave the GPU with bytes
  // the remembered hash does not describe. The snapshot was taken at hash time, so the GPU gets
  // A (what was hashed); the later guest content B is uploaded by the next sync.
  fill(0xA1);
  sync([&] { fill(0xC3); });
  Check(emitted.size() == 1 && gpu_is(0xA1), "snapshot upload did not deliver the hashed bytes");
  sync();
  Check(emitted.size() == 1 && gpu_is(0xC3), "content written during the copy was not uploaded");
  fill(0xA1);
  sync();
  Check(emitted.size() == 1 && gpu_is(0xA1), "return to an older content was skipped (stale hash)");

  // Demotion (KYTY_HOT_DEMOTE_SYNCS=3 set in main) forgets the hash. A clean page is
  // write-protected, so, like the fault handler would, mark it CPU-dirty before writing to it.
  sync();
  sync();
  sync();
  Check(emitted.empty() && gpu_is(0xA1), "unchanged syncs before demotion uploaded");
  fault();
  fill(0xB2);
  sync(); // the demotion sync: the page leaves the hot set and is uploaded as a plain range
  Check(emitted.size() == 1 && emitted[0].snapshot == nullptr && gpu_is(0xB2),
        "demoted page was not uploaded as a plain (unhashed) range");
  for (int i = 0; i < 7; i++) {
    fault();
    sync();
    Check(gpu_is(0xB2), "plain re-upload of the demoted page failed");
  }
  // The 8th transition re-promotes the page; its content is the old hashed A again. Without the
  // demotion invalidation the stale H(A) would match and the upload would be skipped (GPU = B).
  fault();
  fill(0xA1);
  sync();
  Check(emitted.size() == 1 && emitted[0].snapshot != nullptr && gpu_is(0xA1),
        "re-promotion with the previously hashed content skipped the upload (stale hash)");
  sync();
  Check(emitted.empty() && gpu_is(0xA1), "re-promoted page was not skipped when unchanged");
  Release(memory);
}

// Page-driven range synchronization (BufferCache::SynchronizeBuffersInRange, 177d31c): the tracker
// lists the CPU-dirty and hot runs of a range and only their owners are synchronized. Synchronizing
// exactly those runs must upload what one synchronization of the whole range (the historical walk
// over every buffer) uploads, and must leave nothing behind for it.
void TestSyncCandidateRunsMatchWholeRangeUpload() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t page = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr uint64_t pages = 64;
  uint8_t *memory = Allocate(harness.page_manager, pages);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto size = pages * page;

  using Run = std::pair<uint64_t, uint64_t>; // first page, page count
  auto candidates = [&](uint64_t vaddr, uint64_t bytes) {
    std::vector<Run> runs;
    tracker.ForEachSyncCandidateRun(
        vaddr, bytes, [&](uint64_t run_address, uint64_t run_size) noexcept {
          runs.emplace_back((run_address - address) / page, run_size / page);
        });
    return runs;
  };
  // Pages one synchronization of [vaddr, vaddr + bytes) uploads (plain runs and hot snapshots).
  auto upload = [&](uint64_t vaddr, uint64_t bytes) {
    std::vector<uint64_t> uploaded;
    tracker.ForEachUploadRange(
        vaddr, bytes, false,
        [&](uint64_t upload_address, uint64_t upload_size, const void *) noexcept {
          const auto first = (upload_address - address) / page;
          for (uint64_t p = first; p < first + upload_size / page; ++p) {
            uploaded.push_back(p);
          }
        },
        []() noexcept {});
    std::sort(uploaded.begin(), uploaded.end());
    return uploaded;
  };
  // The CPU write-fault path (what the host fault handler runs).
  auto fault = [&](uint64_t first, uint64_t count) {
    tracker.InvalidateRegion(address + first * page, count * page, [] {});
  };

  // Buffer creation synchronizes its bound range (creating the region, which starts fully
  // CPU-dirty); the rest of the region stays a candidate until one sync drains it.
  Check(upload(address, page).size() == 1, "first sync did not upload the bound page");
  Check(candidates(address, size) == std::vector<Run>{{1, pages - 1}},
        "fresh region remainder was not one candidate run");
  Check(upload(address, size).size() == pages - 1, "drain did not upload the fresh remainder");
  Check(candidates(address, size).empty(), "clean range still had sync candidates");

  // Page 10 becomes hot through KYTY_HOT_PAGE_FAULTS (default 8) fault/sync rounds; page 30 is
  // GPU-owned: neither CPU-dirty nor hot, never a candidate.
  for (int i = 0; i < 8; i++) {
    fault(10, 1);
    (void)upload(address + 10 * page, page);
  }
  tracker.MarkRegionAsGpuModified(address + 30 * page, page);

  uint8_t stamp = 0x11;
  auto arrange = [&] {
    fault(3, 3);  // pages 3-5 CPU-dirty
    fault(20, 2); // pages 20-21 CPU-dirty
    std::memset(memory + 10 * page, stamp++, page); // hot page: writable, new content, no fault
  };
  const std::vector<uint64_t> expected = {3, 4, 5, 10, 20, 21};

  // Page-driven: the candidate runs are the dirty and hot runs, clipped to the page-rounded range;
  // synchronizing only those runs uploads the expected pages and leaves nothing for a whole-range
  // synchronization. The hot page stays a candidate (it stays CPU-dirty) but is not re-uploaded
  // while unchanged.
  arrange();
  const auto runs = candidates(address, size);
  Check(runs == std::vector<Run>{{3, 3}, {10, 1}, {20, 2}},
        "sync candidate runs did not match the dirty and hot pages");
  Check(candidates(address + 4 * page + 16, 17 * page) ==
            std::vector<Run>{{4, 2}, {10, 1}, {20, 2}},
        "sync candidate runs were not clipped to the page-rounded range");
  std::vector<uint64_t> page_driven;
  for (const auto [first, count] : runs) {
    const auto part = upload(address + first * page, count * page);
    page_driven.insert(page_driven.end(), part.begin(), part.end());
  }
  std::sort(page_driven.begin(), page_driven.end());
  Check(page_driven == expected, "page-driven synchronization did not upload the dirty and hot pages");
  Check(upload(address, size).empty(),
        "page-driven synchronization left pages for the whole-range synchronization");
  Check(candidates(address, size) == std::vector<Run>{{10, 1}},
        "the hot page did not stay a sync candidate");

  // Buffer walk: the same arrangement synchronized as one range uploads the same page set.
  arrange();
  Check(upload(address, size) == expected,
        "whole-range synchronization uploaded a different page set than the page-driven one");
  Check(!tracker.IsRegionCpuModified(address + 30 * page, page) &&
            tracker.IsRegionGpuModified(address + 30 * page, page),
        "the GPU-owned page was disturbed by the synchronizations");

  tracker.UnmarkRegionAsGpuModified(address + 30 * page, page);
  tracker.UntrackMemory(address, size);
  Release(memory);
}

// F1 (KYTY_SYNC_REPASS) relies on these semantics: a page that becomes CPU-dirty after the candidate
// snapshot, outside the runs already uploaded, stays CPU-dirty and re-arms the region's pending flag;
// a second candidate listing minus the synced hull is exactly that page; a hot page reappears in the
// second listing (it stays CPU-dirty) but lies inside the synced hull and is subtracted.
void TestPostSnapshotDirtyStaysPending() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t page = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr uint64_t pages = 64;
  uint8_t *memory = Allocate(harness.page_manager, pages);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto size = pages * page;
  constexpr uint64_t kA = 2, kB = 40;

  auto candidate_pages = [&] {
    std::vector<uint64_t> result;
    tracker.ForEachSyncCandidateRun(
        address, size, [&](uint64_t run_address, uint64_t run_size) noexcept {
          for (uint64_t p = (run_address - address) / page; p < (run_address - address + run_size) / page; ++p) {
            result.push_back(p);
          }
        });
    return result;
  };
  auto upload = [&](uint64_t first, uint64_t count) {
    tracker.ForEachUploadRange(
        address + first * page, count * page, false,
        [](uint64_t, uint64_t, const void *) noexcept {}, []() noexcept {});
  };
  auto take_pending = [&] {
    int regions = 0;
    tracker.ForEachBdaSyncPendingRegion([&](uint64_t, uint64_t) { ++regions; });
    return regions;
  };
  auto fault = [&](uint64_t first) { tracker.InvalidateRegion(address + first * page, page, [] {}); };
  auto minus = [](std::vector<uint64_t> from, uint64_t hull_first, uint64_t hull_count) {
    from.erase(std::remove_if(from.begin(), from.end(),
                              [&](uint64_t p) { return p >= hull_first && p < hull_first + hull_count; }),
               from.end());
    return from;
  };

  // Create the region and drain its fresh-region dirtiness.
  upload(0, pages);
  Check(candidate_pages().empty(), "drained region still had sync candidates");
  (void)take_pending();

  // 1-5: page A dirty at the snapshot; S3 already took the pending flag.
  fault(kA);
  (void)take_pending();
  Check(candidate_pages() == std::vector<uint64_t>{kA}, "snapshot did not list exactly page A");
  // 2: page B faults after the snapshot.
  fault(kB);
  // 3: only A's run is uploaded.
  upload(kA, 1);
  // 4: B is still CPU-dirty and the region is pending again; A is clean.
  Check(tracker.IsRegionCpuModified(address + kB * page, page), "page B lost its CPU-dirty state");
  Check(!tracker.IsRegionCpuModified(address + kA * page, page), "page A stayed CPU-dirty");
  Check(take_pending() == 1, "the post-snapshot fault did not re-arm the pending flag");
  // 5: second listing minus A's hull is exactly B.
  Check(minus(candidate_pages(), kA, 1) == std::vector<uint64_t>{kB},
        "second candidate pass minus A's hull was not exactly page B");
  upload(kB, 1);
  Check(candidate_pages().empty(), "uploading B left candidates");

  // Hot A: KYTY_HOT_PAGE_FAULTS (default 8) fault/sync rounds. It stays CPU-dirty and a candidate.
  for (int i = 0; i < 8; i++) {
    fault(kA);
    upload(kA, 1);
  }
  (void)take_pending();
  Check(candidate_pages() == std::vector<uint64_t>{kA}, "hot page A was not a candidate");
  fault(kB);
  upload(kA, 1); // pass 1 syncs A's hull only
  const auto second = candidate_pages();
  Check(second == std::vector<uint64_t>({kA, kB}), "hot A did not reappear next to B in pass 2");
  Check(minus(second, kA, 1) == std::vector<uint64_t>{kB}, "hot A was not subtracted by its hull");

  upload(kB, 1);
  tracker.UntrackMemory(address, size);
  Release(memory);
}

// GpuWriteTicks (paged array) must behave exactly like the unordered_map<page, tick> it replaced:
// stamped-or-not, monotonic max, tick 0 still counts as stamped, out-of-range pages are rejected.
void TestGpuWriteTicksMatchesMap() {
  using Ticks = Libs::Graphics::GpuWriteTicks<14>;
  Ticks ticks;
  std::unordered_map<uint64_t, uint64_t> reference;
  uint64_t state = 0x9E3779B97F4A7C15ull;
  const auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  constexpr uint64_t kPages = uint64_t{1} << 30;
  std::vector<uint64_t> touched;
  for (int i = 0; i < 20000; ++i) {
    // Dense window (cheap) plus a few far-apart pages that exercise distinct buckets.
    const uint64_t page = (i % 3 == 0 && !touched.empty()) ? touched[next() % touched.size()]
                          : (i % 500 == 0)                  ? next() % kPages
                                                            : 0x40000 + next() % 0x10000;
    const uint64_t tick = next() % 5 == 0 ? 0 : next() % 1000;
    Check(ticks.Note(page, tick), "valid page rejected");
    auto &stamp = reference[page];
    stamp = std::max(stamp, tick);
    touched.push_back(page);
  }
  for (const auto page : touched) {
    Check(ticks.Contains(page) && ticks.Get(page) == reference[page], "stamp differs from map");
  }
  for (int i = 0; i < 20000; ++i) {
    const uint64_t page = i % 2 ? 0x40000 + next() % 0x12000 : next() % kPages;
    const auto it = reference.find(page);
    Check(ticks.Contains(page) == (it != reference.end()), "presence differs from map");
    Check(ticks.Get(page) == (it != reference.end() ? it->second : 0), "value differs from map");
  }
  Check(!ticks.Note(kPages, 5) && !ticks.Contains(kPages) && ticks.Get(kPages) == 0,
        "out-of-range page must be rejected and unstamped");
  Ticks zero;
  Check(zero.Note(7, 0) && zero.Contains(7) && zero.Get(7) == 0, "tick 0 must still be a stamp");
  Check(zero.Note(7, 9) && zero.Note(7, 3) && zero.Get(7) == 9, "stamp must be monotonic");
}

// SlotVector (paged storage): stable addresses across growth, generation checks, free-list reuse and
// index-ordered ForEach, across several 4096-entry pages.
void TestSlotVectorPaged() {
  struct Item {
    explicit Item(int v) : value(v) {}
    Item(const Item &) = delete;
    int value;
  };
  Common::SlotVector<Item> slots;
  std::vector<Common::SlotId> ids;
  std::vector<const Item *> addresses;
  constexpr int kCount = 3 * 4096 + 17;
  for (int i = 0; i < kCount; ++i) {
    ids.push_back(slots.insert(i));
    addresses.push_back(&slots[ids.back()]);
  }
  Check(slots.size() == static_cast<size_t>(kCount) && slots.capacity() == static_cast<size_t>(kCount),
        "slot count after inserts");
  for (int i = 0; i < kCount; ++i) {
    Check(slots.try_get(ids[i]) == addresses[i] && slots[ids[i]].value == i,
          "slot address moved or value changed while growing");
  }
  for (int i = 0; i < kCount; i += 3) slots.erase(ids[i]);
  for (int i = 0; i < kCount; ++i) {
    Check((slots.try_get(ids[i]) != nullptr) == (i % 3 != 0), "erased slot still allocated");
  }
  Check(slots.try_get(Common::SlotId{}) == nullptr &&
            slots.try_get(Common::SlotId{static_cast<uint32_t>(kCount), 1}) == nullptr,
        "invalid ids must not resolve");
  const auto reused = slots.insert(-1);
  Check(reused.index == ids[kCount - 1 - ((kCount - 1) % 3)].index && reused.generation == 2,
        "free list must reuse the last erased index with a new generation");
  Check(slots.try_get(ids[reused.index]) == nullptr, "stale generation must not resolve");
  int previous = -1;
  size_t visited = 0;
  slots.ForEach([&](Common::SlotId id, const Item &) {
    Check(static_cast<int>(id.index) > previous, "ForEach must visit in index order");
    previous = static_cast<int>(id.index);
    ++visited;
  });
  Check(visited == slots.size(), "ForEach visit count");
}

// The fused stream-path query must equal !IsRegionGpuModified && IsRegionCpuModified in every
// CPU/GPU state combination, for one page and for a range spanning two tracker regions.
void TestFusedCpuNotGpuQuery() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto agree = [&](uint64_t begin, uint64_t size, const char *state) {
    const bool old_form =
        !tracker.IsRegionGpuModified(begin, size) && tracker.IsRegionCpuModified(begin, size);
    Check(tracker.IsRegionCpuModifiedNotGpuModified(begin, size) == old_form, state);
    return old_form;
  };
  // Fresh range: CPU-dirty, never GPU-owned.
  Check(agree(address, page_size, "fused query disagrees on a fresh range"),
        "a fresh range must be CPU-modified");
  // Uploaded clean: neither.
  tracker.ForEachUploadRange(
      address, page_size * 4, false, [](uint64_t, uint64_t, const void *) noexcept {},
      []() noexcept {});
  Check(!agree(address, page_size, "fused query disagrees on a clean range"),
        "a clean range must not be CPU-modified");
  // GPU-owned page.
  tracker.ForEachUploadRange(
      address + page_size, page_size, true, [](uint64_t, uint64_t, const void *) noexcept {},
      []() noexcept {});
  Check(!agree(address + page_size, page_size, "fused query disagrees on a GPU page"),
        "a GPU-owned page must not take the stream path");
  // Mixed range: one GPU-owned page inside a wider range is GPU-modified overall.
  Check(!agree(address, page_size * 4, "fused query disagrees on a mixed range"),
        "a range holding a GPU-owned page must not take the stream path");
  // CPU-dirty page next to the GPU one.
  tracker.MarkRegionAsCpuModified(address + 2 * page_size, page_size);
  Check(agree(address + 2 * page_size, page_size, "fused query disagrees on a CPU page"),
        "a CPU-dirty page must take the stream path");
  Check(!agree(address, page_size * 4, "fused query disagrees on CPU+GPU range"),
        "CPU+GPU range must stay off the stream path");
  tracker.UnmarkRegionAsGpuModified(address + page_size, page_size);
  Check(agree(address, page_size * 4, "fused query disagrees after GPU release"),
        "after the GPU page is released the CPU-dirty page qualifies");
  tracker.MarkRegionAsCpuModified(address, page_size * 4);
  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
// Kernel write-watch (FPS Stage 1): views of a memfd registered with userfaultfd async
// write-protect, as GuestBackingStore::MapView does. CPU writes reach the tracker only through
// CollectWrittenPages. Runs in a child process (--write-watch): KYTY_WRITE_WATCH is read once and
// turns hot pages off for the whole process.
using Libs::Graphics::TRACKER_PAGE_SIZE;
using Libs::Graphics::WriteWatch;

struct WriteWatchView {
  int fd = -1;
  uint8_t *memory = nullptr;
  uint64_t size = 0;
  [[nodiscard]] uint64_t Address() const { return reinterpret_cast<uint64_t>(memory); }
};

WriteWatchView MapWriteWatchView(uint64_t pages, uintptr_t offset = 0x40000) {
  WriteWatchView view;
  view.size = pages * TRACKER_PAGE_SIZE;
  view.fd = ::memfd_create("WriteWatchTest", MFD_CLOEXEC);
  Check(view.fd >= 0 && ::ftruncate(view.fd, static_cast<off_t>(view.size)) == 0,
        "memfd creation failed");
  // A fixed guest-range placeholder, then the shared view over it.
  view.memory = AllocateFixedGuestRange(view.size, offset);
  Check(::mmap(view.memory, view.size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, view.fd,
               0) == view.memory,
        "memfd view mapping failed");
  WriteWatch::Get()->RegisterView(view.Address(), view.size);
  Check(WriteWatch::Get()->IsCovered(view.Address(), view.size), "view was not registered");
  return view;
}

void UnmapWriteWatchView(WriteWatchView &view) {
  WriteWatch::Get()->UnregisterView(view.Address(), view.size);
  Release(view.memory);
  ::close(view.fd);
}

// RenderContext::CollectCpuWrites without the texture cache: each collected run is applied as the
// write fault it replaces. Returns the pages collected.
uint64_t CollectAndApply(MemoryTracker &tracker, uint64_t vaddr, uint64_t size) {
  uint64_t pages = 0;
  tracker.CollectWrittenPages(vaddr, size, [&](uint64_t start, uint64_t bytes) {
    pages += bytes / TRACKER_PAGE_SIZE;
    tracker.InvalidateWriteFault(start, bytes, [] { Check(false, "unexpected GPU download"); });
  });
  return pages;
}

void UploadAll(MemoryTracker &tracker, uint64_t vaddr, uint64_t size) {
  tracker.ForEachUploadRange(vaddr, size, false, [](uint64_t, uint64_t, const void *) noexcept {},
                             []() noexcept {});
}

void TestWriteWatchCollectsWithoutWriteProtection() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  auto view = MapWriteWatchView(8);
  const auto address = view.Address();
  Check(tracker.IsRegionCpuModified(address, view.size), "new region was not CPU dirty");
  ResetProtectionLog();
  UploadAll(tracker, address, view.size);
  Check(!tracker.IsRegionCpuModified(address, view.size), "upload left CPU-dirty pages");
  Check(g_protection_calls == 0 && IsWritable(view.memory + 3 * page),
        "a write-watched view was write-protected");

  // Plain stores: no fault, recorded by the kernel only.
  view.memory[3 * page + 5] = 1;
  view.memory[6 * page] = 2;
  Check(!tracker.IsRegionCpuModified(address, view.size),
        "a CPU write reached the tracker before its collection");
  Check(CollectAndApply(tracker, address, view.size) == 2,
        "collection did not report exactly the two written pages");
  Check(tracker.IsRegionCpuModified(address + 3 * page, page) &&
            tracker.IsRegionCpuModified(address + 6 * page, page) &&
            !tracker.IsRegionCpuModified(address, 3 * page) &&
            !tracker.IsRegionCpuModified(address + 4 * page, 2 * page),
        "collected writes did not mark exactly their pages CPU-dirty");
  Check(CollectAndApply(tracker, address, view.size) == 0,
        "a second collection reported pages again");

  // Whole-tracker collection (size 0) walks the spans that hold write-watched pages.
  UploadAll(tracker, address, view.size);
  view.memory[page] = 3;
  Check(CollectAndApply(tracker, 0, 0) == 1 && tracker.IsRegionCpuModified(address + page, page),
        "span collection missed a write");
  Check(g_protection_calls == 0, "write-watch transitions called mprotect");

  tracker.UntrackMemory(address, view.size);
  UnmapWriteWatchView(view);
}

void TestWriteWatchWriteBetweenScanAndUpload() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  auto view = MapWriteWatchView(4);
  const auto address = view.Address();
  UploadAll(tracker, address, view.size);

  view.memory[2 * page] = 1;
  Check(CollectAndApply(tracker, address, view.size) == 1, "first write was not collected");
  // The scan re-protected the page; the guest writes it again while its upload copies it.
  bool wrote = false;
  tracker.ForEachUploadRange(
      address, view.size, false,
      [&](uint64_t start, uint64_t bytes, const void *) noexcept {
        if (start <= address + 2 * page && address + 2 * page < start + bytes) {
          view.memory[2 * page + 7] = 9;
          wrote = true;
        }
      },
      []() noexcept {});
  Check(wrote && !tracker.IsRegionCpuModified(address, view.size),
        "the upload did not consume the collected page");
  Check(CollectAndApply(tracker, address, view.size) == 1 &&
            tracker.IsRegionCpuModified(address + 2 * page, page),
        "a write between scan and upload was lost");

  tracker.UntrackMemory(address, view.size);
  UnmapWriteWatchView(view);
}

void TestWriteWatchCpuWriteWinsOverGpuMark() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  auto view = MapWriteWatchView(8);
  const auto address = view.Address();
  UploadAll(tracker, address, view.size);
  std::vector<std::pair<uint64_t, uint64_t>> gpu_parts;
  // BufferCache::CollectCpuWritesInline: what MarkPageWritten does before TryMark.
  const auto collect_inline = [&](uint64_t vaddr, uint64_t size) {
    tracker.CollectWrittenPages(vaddr, size, [&](uint64_t start, uint64_t bytes) {
      tracker.MarkCpuWrittenUnlessGpuModified(start, bytes, [&](uint64_t gpu, uint64_t length) {
        gpu_parts.emplace_back(gpu, length);
      });
    });
  };

  // An uncollected CPU write wins over a later GPU write-fault mark, as a faulted one does.
  view.memory[4 * page] = 1;
  collect_inline(address + 4 * page, page);
  Check(gpu_parts.empty() && tracker.IsRegionCpuModified(address + 4 * page, page),
        "inline collection did not mark the written page CPU-dirty");
  Check(!tracker.TryMarkRegionAsGpuModified(address + 4 * page, page),
        "a GPU mark won over an earlier CPU write");

  // Without a CPU write the GPU takes the page, and read protection still applies.
  collect_inline(address + 5 * page, page);
  Check(tracker.TryMarkRegionAsGpuModified(address + 5 * page, page),
        "GPU mark of an unwritten page failed");
  Check(Protection(view.memory + 5 * page) == PAGE_NOACCESS,
        "a GPU-owned page lost its read protection under write-watch");

  // A CPU write that landed after the collection but before the mark's read protection.
  view.memory[6 * page] = 1;
  Check(tracker.TryMarkRegionAsGpuModified(address + 6 * page, page),
        "GPU mark in the window failed");
  // BufferCache::MarkPageWritten verifies after marking: the write predates the GPU's ownership,
  // so the mark is undone and the CPU wins.
  std::vector<std::pair<uint64_t, uint64_t>> written;
  tracker.CollectWrittenPages(address + 6 * page, page, [&](uint64_t start, uint64_t bytes) {
    written.emplace_back(start, bytes);
  });
  Check(written.size() == 1 && written[0].first == address + 6 * page,
        "the verify collection after the GPU mark missed the window write");
  tracker.UnmarkRegionAsGpuModified(address + 6 * page, page);
  for (const auto &[start, bytes] : written) {
    tracker.MarkCpuWrittenUnlessGpuModified(start, bytes, [&](uint64_t gpu, uint64_t length) {
      gpu_parts.emplace_back(gpu, length);
    });
  }
  Check(gpu_parts.empty() && tracker.IsRegionCpuModified(address + 6 * page, page) &&
            !tracker.IsRegionGpuModified(address + 6 * page, page) && IsWritable(view.memory + 6 * page),
        "the CPU write in the marking window did not win");

  // Without the verify step the same write comes back as a GPU part (download first).
  view.memory[7 * page] = 1;
  Check(tracker.TryMarkRegionAsGpuModified(address + 7 * page, page), "GPU mark of page 7 failed");
  collect_inline(address + 7 * page, page);
  Check(gpu_parts.size() == 1 && gpu_parts[0].first == address + 7 * page &&
            tracker.IsRegionGpuModified(address + 7 * page, page) &&
            !tracker.IsRegionCpuModified(address + 7 * page, page),
        "a write under a GPU-owned page was not handed back as a GPU part");

  tracker.UnmarkRegionAsGpuModified(address + 5 * page, page);
  tracker.UnmarkRegionAsGpuModified(address + 7 * page, page);
  tracker.UntrackMemory(address, view.size);
  UnmapWriteWatchView(view);
}

// Anonymous guest memory (flexible commits) is never registered: it keeps mprotect tracking.
void TestWriteWatchUncoveredMemoryKeepsProtection() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  auto *memory = Allocate(harness.page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  Check(!WriteWatch::Get()->IsCovered(address, page), "anonymous memory counted as covered");
  Check(tracker.IsRegionCpuModified(address, 2 * page), "new region was not CPU dirty");
  UploadAll(tracker, address, 2 * page);
  Check(Protection(memory) == PAGE_READONLY && Protection(memory + page) == PAGE_READONLY,
        "an uncovered page was not write-protected");
  tracker.UntrackMemory(address, 2 * page);
  Check(IsWritable(memory), "untracking left an uncovered page protected");
  Release(memory);
}

// A page written while nobody watched it (CPU-dirty, or no buffer over it yet) is re-protected
// when it gains its first write watcher: the old write is no news, as under mprotect.
void TestWriteWatchFirstWatcherDropsOldWrites() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  auto view = MapWriteWatchView(4);
  const auto address = view.Address();
  Check(tracker.IsRegionCpuModified(address, view.size), "new region was not CPU dirty");
  view.memory[2 * page] = 7; // CPU-dirty page: no watcher yet
  ResetProtectionLog();
  UploadAll(tracker, address, view.size);
  Check(g_protection_calls == 0, "re-arming a first watcher fell back to mprotect");
  Check(CollectAndApply(tracker, address, view.size) == 0,
        "a write from before the page was watched came back as a new one");
  Check(tracker.TryMarkRegionAsGpuModified(address + 2 * page, page),
        "a stale write blocked a later GPU mark");
  tracker.UnmarkRegionAsGpuModified(address + 2 * page, page);
  tracker.UntrackMemory(address, view.size);
  UnmapWriteWatchView(view);
}

// Bindings start anywhere: an unaligned collection range must scan its pages, not fail
// (PAGEMAP_SCAN rejects an unaligned start, and a failed scan counts as fully written).
void TestWriteWatchUnalignedCollection() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  auto view = MapWriteWatchView(4);
  const auto address = view.Address();
  UploadAll(tracker, address, view.size);
  (void)WriteWatch::TakeStats();
  Check(CollectAndApply(tracker, address + page - 4, 12) == 0,
        "an unaligned collection over clean pages reported writes");
  view.memory[2 * page + 100] = 1;
  Check(CollectAndApply(tracker, address + page + 8, page) == 1 &&
            tracker.IsRegionCpuModified(address + 2 * page, page) &&
            !tracker.IsRegionCpuModified(address + page, page),
        "an unaligned collection did not report exactly the written page");
  Check(WriteWatch::TakeStats().errors == 0, "an unaligned collection made a scan fail");
  tracker.UntrackMemory(address, view.size);
  UnmapWriteWatchView(view);
}

// A full collection scans merged ranges in one pass: it must cross the unregistered anonymous
// mapping between two views (a placeholder) without failing, and report the writes in both.
void TestWriteWatchFullCollectionCrossesPlaceholders() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  auto first = MapWriteWatchView(4, 0x40000);
  auto *gap = AllocateFixedGuestRange(4 * page, 0x50000); // anonymous, never registered
  auto second = MapWriteWatchView(4, 0x60000);
  Check(reinterpret_cast<uint64_t>(gap) / Libs::Graphics::TRACKER_REGION_SIZE ==
            first.Address() / Libs::Graphics::TRACKER_REGION_SIZE &&
            second.Address() / Libs::Graphics::TRACKER_REGION_SIZE ==
                first.Address() / Libs::Graphics::TRACKER_REGION_SIZE,
        "the two views and the gap were not placed in one tracker region");
  UploadAll(tracker, first.Address(), first.size);
  UploadAll(tracker, second.Address(), second.size);
  (void)WriteWatch::TakeStats();
  first.memory[page] = 1;
  second.memory[3 * page] = 2;
  Check(CollectAndApply(tracker, 0, 0) == 2 &&
            tracker.IsRegionCpuModified(first.Address() + page, page) &&
            tracker.IsRegionCpuModified(second.Address() + 3 * page, page),
        "a full collection across a placeholder missed a view's write");
  const auto stats = WriteWatch::TakeStats();
  Check(stats.errors == 0 && stats.range_passes == 1,
        "a full collection failed on the placeholder or was not one pass");
  tracker.UntrackMemory(first.Address(), first.size);
  tracker.UntrackMemory(second.Address(), second.size);
  UnmapWriteWatchView(second);
  Release(gap);
  UnmapWriteWatchView(first);
}

void TestWriteWatchLabels() {
  auto *write_watch = WriteWatch::Get();
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  auto view = MapWriteWatchView(4);
  const auto label = view.Address() + page + 8;
  Check(write_watch->WasLabelWritten(label, 8), "an unknown label must count as written");
  write_watch->CollectLabels([](uint64_t, uint64_t) {});
  Check(!write_watch->WasLabelWritten(label, 8), "an unwritten collected label counted as written");
  *reinterpret_cast<volatile uint64_t *>(label) = 1;
  Check(write_watch->WasLabelWritten(label, 8), "a label write after collection was missed");
  // Another collection re-protects the label page; the label must still read as written.
  write_watch->Collect(view.Address(), view.size, [](uint64_t, uint64_t) {});
  Check(write_watch->WasLabelWritten(label, 8), "re-protecting the label page hid its write");
  write_watch->CollectLabels([](uint64_t, uint64_t) {});
  Check(!write_watch->WasLabelWritten(label, 8), "CollectLabels did not reset the label");
  UnmapWriteWatchView(view);
  Check(write_watch->WasLabelWritten(label, 8), "a label in an unregistered range must count as written");
}

// The label pass scans nearby label pages as one range; another written page it meets there is
// reported like any collected run.
void TestWriteWatchLabelPassMergesRanges() {
  auto *write_watch = WriteWatch::Get();
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  auto view = MapWriteWatchView(8);
  const auto first_label = view.Address() + page;
  const auto second_label = view.Address() + 6 * page;
  (void)write_watch->WasLabelWritten(first_label, 8); // both join the label set
  (void)write_watch->WasLabelWritten(second_label, 8);
  write_watch->CollectLabels([](uint64_t, uint64_t) {});
  view.memory[3 * page] = 1; // between the labels
  (void)WriteWatch::TakeStats();
  std::vector<std::pair<uint64_t, uint64_t>> runs;
  write_watch->CollectLabels(
      [&](uint64_t start, uint64_t bytes) { runs.emplace_back(start, start + bytes); });
  const auto stats = WriteWatch::TakeStats();
  const auto &labels = stats.sources[static_cast<size_t>(WriteWatch::Source::Labels)];
  Check(labels.calls == 1 && labels.ranges == 1 && labels.ioctls == 1,
        "nearby label pages were not scanned as one range");
  Check(runs.size() == 1 && runs[0].first == view.Address() + 3 * page &&
            runs[0].second == view.Address() + 4 * page,
        "the label pass did not report exactly the page written between the labels");
  Check(!write_watch->WasLabelWritten(first_label, 8) && !write_watch->WasLabelWritten(second_label, 8),
        "a write next to the labels made them count as written");
  UnmapWriteWatchView(view);
}

// A BDA sync's batch: one pass over the regions it is about to upload (PrearmWrittenPages), then
// uploads inside a PrearmScope re-arm nothing of their own. A CPU-dirty page's write before the
// pass is the upload's to copy, a watched page's write is collected by the pass, and a write after
// the pass comes back at the next collection.
void TestWriteWatchPrearmedUpload() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  auto view = MapWriteWatchView(8);
  const auto address = view.Address();
  UploadAll(tracker, address, view.size);
  view.memory[2 * page] = 1;
  Check(CollectAndApply(tracker, address, view.size) == 1, "first write was not collected");
  view.memory[2 * page + 8] = 2; // CPU-dirty page written again: the upload copies it
  view.memory[5 * page] = 3;     // watched page: the pass must collect it
  (void)WriteWatch::TakeStats();
  std::vector<std::pair<uint64_t, uint64_t>> scanned;
  uint64_t collected = 0;
  tracker.PrearmWrittenPages(
      {{address, address + view.size}},
      [&](uint64_t start, uint64_t bytes) {
        collected += bytes / page;
        tracker.InvalidateWriteFault(start, bytes, [] { Check(false, "unexpected GPU download"); });
      },
      &scanned);
  Check(collected == 1 && tracker.IsRegionCpuModified(address + 5 * page, page),
        "the pre-arm pass did not collect exactly the watched page's write");
  Check(scanned.size() == 1 && scanned[0].first <= address && address + view.size <= scanned[0].second,
        "the pre-arm pass did not return the range it covered");
  bool wrote = false;
  {
    PageManager::PrearmScope scope(std::move(scanned));
    tracker.ForEachUploadRange(
        address, view.size, false,
        [&](uint64_t start, uint64_t bytes, const void *) noexcept {
          if (start <= address + 5 * page && address + 5 * page < start + bytes) {
            view.memory[5 * page + 8] = 4; // after the pass, while the upload runs
            wrote = true;
          }
        },
        []() noexcept {});
  }
  Check(wrote && !tracker.IsRegionCpuModified(address, view.size),
        "the upload did not consume the dirty pages");
  const auto stats = WriteWatch::TakeStats();
  Check(stats.rearms == 0 && stats.rearm_skipped == 2,
        "uploads inside a pre-arm scope re-armed their pages one by one");
  Check(CollectAndApply(tracker, address, view.size) == 1 &&
            tracker.IsRegionCpuModified(address + 5 * page, page) &&
            !tracker.IsRegionCpuModified(address + 2 * page, page),
        "a write before the pass came back, or the write after it was lost");
  // Outside a scope, a page gaining its first watcher is re-armed again.
  UploadAll(tracker, address, view.size);
  Check(WriteWatch::TakeStats().rearms == 1, "an upload outside the scope did not re-arm");
  tracker.UntrackMemory(address, view.size);
  UnmapWriteWatchView(view);
}

void RunWriteWatchCases() {
  setenv("KYTY_WRITE_WATCH", "1", 1);
  if (WriteWatch::Get() == nullptr) {
    std::puts("MemoryTrackerTests: write-watch cases skipped (no userfaultfd WP_ASYNC / PAGEMAP_SCAN)");
    std::exit(0);
  }
  TestWriteWatchCollectsWithoutWriteProtection();
  TestWriteWatchWriteBetweenScanAndUpload();
  TestWriteWatchCpuWriteWinsOverGpuMark();
  TestWriteWatchUncoveredMemoryKeepsProtection();
  TestWriteWatchFirstWatcherDropsOldWrites();
  TestWriteWatchUnalignedCollection();
  TestWriteWatchFullCollectionCrossesPlaceholders();
  TestWriteWatchLabels();
  TestWriteWatchLabelPassMergesRanges();
  TestWriteWatchPrearmedUpload();
  std::puts("MemoryTrackerTests: write-watch cases passed");
  std::exit(0);
}

void TestWriteWatchInChildProcess() {
  const pid_t pid = ::fork();
  Check(pid >= 0, "fork failed");
  if (pid == 0) {
    ::execl("/proc/self/exe", "MemoryTrackerTests", "--write-watch", nullptr);
    std::_Exit(0x7e);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid, "waitpid failed");
  Check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "write-watch cases failed");
}
#endif

int main(int argc, char **argv) {
  if (argc == 3 && std::strcmp(argv[1], "--death") == 0) {
    RunDeathCase(argv[2]);
  }
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
  if (argc == 2 && std::strcmp(argv[1], "--write-watch") == 0) {
    RunWriteWatchCases();
  }
#endif
  if (argc == 2 && std::strcmp(argv[1], "--benchmark-clean-upload") == 0) {
    BenchmarkCleanUploads();
    return 0;
  }
  setenv("KYTY_HOT_DEMOTE_SYNCS", "3", 1); // read once, by the first sync that sees a hot page
  TestGpuWriteTicksMatchesMap();
  TestSlotVectorPaged();
  TestFusedCpuNotGpuQuery();
  TestGuestRange();
  TestRangeSet();
  TestHotPageSnapshotHash();
  TestSyncCandidateRunsMatchWholeRangeUpload();
  TestPostSnapshotDirtyStaysPending();
  TestQueriesDoNotRequireMappedOwnership();
  TestConcurrentRegionPublication();
  TestCpuDirtyUpload();
  TestCleanUploadPreservesOwnership();
  TestRangeInvalidation();
  TestGpuReacquisitionAfterInvalidation();
  TestGpuDirtyBits();
  TestExactDirtyIntervalsSharingTrackerPage();
  TestGpuDownloadProtectionMirrors();
  TestCrossRegionUpload();
  TestConcurrentColdUploads();
  TestUploadDoesNotSerializeDisjointRegion();
  TestDownloadDoesNotSerializeDisjointRegion();
  TestGpuUnmarkUsesRegionMask();
  TestFullRegionGpuUnmarkBatching();
  TestFatalPaths();
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
  TestFaultOnProtectedStack();
  TestWriteWatchInChildProcess();
#endif
  std::puts("MemoryTrackerTests: all cases passed");
  return 0;
}
