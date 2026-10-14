#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>

namespace deep_gemm::layout {

constexpr uint32_t kNumMaxRanks = 32;

// Same address mapping contract as upstream: passed by value, not a GPU table.
template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    int64_t base{};
    int64_t offsets[kNumRanks]{};
    uint32_t rank_idx{};

    CUTLASS_HOST_DEVICE SymBuffer() = default;

    template <typename Container>
    explicit SymBuffer(const Container& pointers, uint32_t rank) : base(pointers[rank]), rank_idx(rank) {
        for (uint32_t peer = 0; peer < kNumRanks; ++peer)
            offsets[peer] = peer < pointers.size() ? pointers[peer] - base : 0;
    }

    template <typename Pointer = void*>
    CUTLASS_HOST_DEVICE Pointer get_base_ptr() const {
        return reinterpret_cast<Pointer>(base);
    }

    template <typename Pointer>
    CUTLASS_HOST_DEVICE Pointer map(Pointer pointer, uint32_t peer) const {
        return reinterpret_cast<Pointer>(reinterpret_cast<int64_t>(pointer) + offsets[peer]);
    }
};

}  // namespace deep_gemm::layout
