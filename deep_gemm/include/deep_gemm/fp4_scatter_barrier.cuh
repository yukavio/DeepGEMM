#pragma once

#include <cstdint>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm::comm {

// The FP4 scatter path needs one intra-rank grid counter and one alternating
// cross-rank signal pair. The workspace must live in symmetric device memory.
struct Fp4ScatterBarrierWorkspace {
    int32_t grid_counter;
    int32_t phase_counter;
    int32_t phase_signal[2];
};

__device__ __forceinline__ void scatter_red_add_release_system(
        int32_t* address, int32_t value) {
    asm volatile("ppu.atom.add.release.sys.global.s32 _, [%0], %1;"
                 :: "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t scatter_load_acquire_device(
        const uint32_t* address) {
    uint32_t value;
    asm volatile("ppu.ld.acquire.gpu.global.b32 %0, [%1];"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ int32_t scatter_load_acquire_system(
        const int32_t* address) {
    int32_t value;
    asm volatile("ppu.ld.acquire.sys.global.s32 %0, [%1];"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ uint32_t scatter_atomic_add_release_device(
        uint32_t* address, uint32_t value) {
    uint32_t old_value;
    asm volatile("ppu.atom.add.release.gpu.global.u32 %0, [%1], %2;"
                 : "=r"(old_value) : "l"(address), "r"(value) : "memory");
    return old_value;
}

__device__ __forceinline__ void scatter_red_add_device(
        int32_t* address, int32_t value) {
    asm volatile("ppu.atom.add.relaxed.gpu.global.s32 _, [%0], %1;"
                 :: "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ void fp4_scatter_grid_sync(
        Fp4ScatterBarrierWorkspace& workspace,
        uint32_t num_blocks, uint32_t block_idx, uint32_t thread_idx) {
    constexpr uint32_t kFinishTag = 0x80000000u;
    __syncthreads();
    if (thread_idx == 0) {
        auto* counter = reinterpret_cast<uint32_t*>(&workspace.grid_counter);
        const uint32_t old_value = scatter_atomic_add_release_device(
            counter, block_idx == 0 ? kFinishTag - (num_blocks - 1) : 1);
        uint32_t new_value;
        do {
            new_value = scatter_load_acquire_device(counter);
        } while (((new_value ^ old_value) & kFinishTag) == 0);
    }
    __syncthreads();
}

// Release all local P2P stores, then acquire peer completion before reduce.
template <uint32_t kWorldSize>
__device__ __forceinline__ void fp4_scatter_release(
        Fp4ScatterBarrierWorkspace& workspace,
        const layout::SymBuffer<kWorldSize>& sym_buffer,
        uint32_t num_blocks, uint32_t block_idx, uint32_t thread_idx) {
    // PPU fences drain only the calling warp; each warp leader releases its stores.
    // The following grid sync joins these releases before peer notification.
    __syncwarp();
    if ((thread_idx & 31u) == 0) {
        __ppu_threadfence_system_uncache();
    }

    fp4_scatter_grid_sync(
        workspace, num_blocks, block_idx, thread_idx);

    if (block_idx == 0) {
        const int32_t status = workspace.phase_counter & 3;
        const uint32_t phase = status & 1;
        const bool negative = (status >> 1) != 0;

        if (thread_idx < kWorldSize) {
            auto* remote_workspace = sym_buffer.map(&workspace, thread_idx);
            scatter_red_add_release_system(
                &remote_workspace->phase_signal[phase], negative ? -1 : 1);
        }
        __syncthreads();

        if (thread_idx == 0) {
            scatter_red_add_device(&workspace.phase_counter, 1);
            const int32_t target = negative ? 0 : static_cast<int32_t>(kWorldSize);
            while (scatter_load_acquire_system(
                       &workspace.phase_signal[phase]) != target) {
            }
        }
    }

    // Multi-node ranks join all CTA leaders after the peer acquire.
    if constexpr (kWorldSize != 8) {
        fp4_scatter_grid_sync(
            workspace, num_blocks, block_idx, thread_idx);
    }
}

}  // namespace deep_gemm::comm
