#pragma once

#include <cstdint>
#include <hggc.h>
#include <hggc_bf16.h>
#include <hggc_runtime.h>

constexpr int kMoeReduceThreads = 512;

template <bool kCheckInvalid, int kTopK, int kHidden, int kThreads = kMoeReduceThreads,
          bool kBulkScatter = true>
__device__ __forceinline__ void moe_reduce_token(
        const __ppu_bfloat16* __restrict__ input,
        __ppu_bfloat16* __restrict__ output,
        const int64_t (&slots)[kTopK], const float (&weights)[kTopK],
        const bool (&remote_slots)[kTopK], int token) {
    constexpr int kVector = sizeof(int4) / sizeof(__ppu_bfloat16);
    constexpr int kStride = kThreads * kVector;
    for (int column = threadIdx.x * kVector;
         column < kHidden; column += kStride) {
        float accumulator[kVector] = {0.0f};
        #pragma unroll
        for (int index = 0; index < kTopK; ++index) {
            if constexpr (kCheckInvalid) {
                if (slots[index] < 0)
                    continue;
            }
            const auto* source = input +
                slots[index] * kHidden + column;
            int4 packed;
            // Match the remote bulk and local vector layouts written by scatter.
            if (kBulkScatter && remote_slots[index]) {
                const uint4 bulk = __ppu_global_ldcg_bulk_b32x4(source);
                packed = *reinterpret_cast<const int4*>(&bulk);
            } else {
                packed = __ldcg(reinterpret_cast<const int4*>(source));
            }
            const auto* values =
                reinterpret_cast<const __ppu_bfloat16*>(&packed);
            #pragma unroll
            for (int lane = 0; lane < kVector; ++lane) {
                accumulator[lane] +=
                    __bfloat162float(values[lane]) * weights[index];
            }
        }

        int4 packed_output;
        auto* values = reinterpret_cast<__ppu_bfloat16*>(&packed_output);
        #pragma unroll
        for (int lane = 0; lane < kVector; ++lane) {
            values[lane] = __float2bfloat16_rn(accumulator[lane]);
        }
        *reinterpret_cast<int4*>(output + token * kHidden + column) =
            packed_output;
    }
}

// GEMM2 scatter completes peer synchronization before this weighted reduction.
template <int kTopK, int kHidden, int kThreads = kMoeReduceThreads,
          bool kBulkScatter = true>
__device__ __forceinline__ void moe_reduce_impl(
        const __ppu_bfloat16* __restrict__ input,
        const int32_t* __restrict__ topk_indices,
        const float* __restrict__ topk_weights,
        __ppu_bfloat16* __restrict__ output,
        int max_m, int local_experts, int rank_idx, int token = -1) {
    static_assert(kHidden % 256 == 0,
                  "remote bulk reduction requires complete 256-element warp rows");
    if (token < 0) token = blockIdx.x;

    int64_t slots[kTopK];
    float weights[kTopK];
    bool remote_slots[kTopK];
    bool all_valid = true;
    #pragma unroll
    for (int index = 0; index < kTopK; ++index) {
        const int32_t expert =
            __ldg(topk_indices + token * kTopK + index);
        const bool valid = expert >= 0;
        all_valid &= valid;
        remote_slots[index] = valid &&
            expert / local_experts != rank_idx;
        slots[index] = valid
            ? index * static_cast<int64_t>(max_m) + token
            : -1;
        weights[index] = valid
            ? __ldg(topk_weights + token * kTopK + index)
            : 0.0f;
    }

    // Skip invalid slots before loading scatter data, even when their weight is zero.
    if (all_valid) {
        moe_reduce_token<false, kTopK, kHidden, kThreads, kBulkScatter>(
            input, output, slots, weights, remote_slots, token);
    } else {
        moe_reduce_token<true, kTopK, kHidden, kThreads, kBulkScatter>(
            input, output, slots, weights, remote_slots, token);
    }
}
