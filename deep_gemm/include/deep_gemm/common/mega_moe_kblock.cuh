#pragma once
#include <stdint.h>
#ifndef DG_MOE_STREAMED
#define DG_MOE_STREAMED 0
#endif
#ifndef DG_MOE_KBLOCK
#define DG_MOE_KBLOCK 0
#endif
#if DG_MOE_STREAMED
namespace deep_gemm {
// Isolate each G1 BM64 scale tile in complete cache lines. G2 AIU prefetch
// must never cache another producer's not-yet-published scale values.
__device__ __forceinline__ uint16_t* stream_moe_scales(uint32_t* ws) {
    constexpr int offset = (16 + 32 * DG_MOE_STREAM_SLOTS + 31) / 32 * 32;
    return reinterpret_cast<uint16_t*>(ws + offset);
}
__device__ __forceinline__ uint32_t stream_moe_load(const uint32_t* p) {
    uint32_t v;
    asm volatile("ppu.ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t stream_moe_add(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("ppu.atom.add.release.gpu.global.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
__device__ __forceinline__ void stream_moe_prepare(uint32_t* ws, uint32_t generation) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __ppu_threadfence_uncache();
        asm volatile("ppu.st.release.gpu.global.u32 [%0], %1;" :: "l"(ws), "r"(generation) : "memory");
    }
    if (threadIdx.x == 0) while (stream_moe_load(ws) != generation) {}
    __syncthreads();
    if (threadIdx.x == 0 && stream_moe_add(ws + 1, 1) + 1 == gridDim.x) {
        asm volatile("ppu.st.release.gpu.global.u32 [%0], %1;" :: "l"(ws + 2), "r"(1) : "memory");
    }
}
__device__ __forceinline__ void stream_moe_finish_g1(uint32_t* ws, uint32_t expert, uint32_t m, uint32_t n) {
    // All payload and scale writers retire before the release publication.
    __ppu_threadfence_uncache();
    __syncthreads();
    if (threadIdx.x == 0) {
        auto* ready = ws + 16 + expert * DG_MOE_STREAM_SLOTS + m;
#if DG_MOE_KBLOCK
        uint32_t old;
        asm volatile("ppu.atom.or.release.gpu.global.b32 %0, [%1], %2;"
                     : "=r"(old) : "l"(ready), "r"(1u << n) : "memory");
#else
        stream_moe_add(ready, 1);
#endif
    }
}
}
#endif
