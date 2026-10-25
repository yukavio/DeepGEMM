#pragma once
#include <deep_gemm/layout/sym_buffer.cuh>

#include <cstdint>
#include <cstdlib>
#include <hggc_bf16.h>
#include <deep_gemm/dispatch_layout.cuh>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// Convert two FP32 values to packed E2M1x2 (two FP4 values in 1 byte)
__device__ __forceinline__ uint8_t cvt_f32x2_to_fp4x2(float hi, float lo) {
    uint8_t b;
    asm volatile("ppu.cvt.rtte.satfinite.e2m1x2.f32 %0, %1, %2;\n"
                 : "=c"(b)
                 : "f"(hi), "f"(lo));
    return b;
}

__device__ __forceinline__ __ppu_bfloat16 fast_pow2_bf16(int x) {
    uint16_t bits = static_cast<uint16_t>((x + 127) << 7);
    return *reinterpret_cast<__ppu_bfloat16*>(&bits);
}

// Returns: scale (BF16 multiplier for quantization), scale_inv (UE8M0 for dequantization)
__device__ __forceinline__ void calculate_mxfp4_scales_bf16(
    __ppu_bfloat16 amax, __ppu_bfloat16& scale, uint8_t& scale_inv) {
    constexpr float kAmaxInvMXFP4 = 0.16666667f;  // 1/6.0, max representable E2M1 value
    __ppu_bfloat16 scale_inv_bf16 = __hmul(amax, __float2bfloat16(kAmaxInvMXFP4));
    uint16_t scale_inv_bits = *reinterpret_cast<uint16_t*>(&scale_inv_bf16);
    // Round the positive BF16 magnitude up to its UE8M0 exponent, ignoring the sign bit.
    uint16_t biased_exp_scale_inv = static_cast<uint16_t>(
        ((scale_inv_bits & 0x7fffu) + 0x7fu) >> 7);
    int exp_scale_inv = static_cast<int>(biased_exp_scale_inv) - 127;
    scale = fast_pow2_bf16(-exp_scale_inv);
    scale_inv = static_cast<uint8_t>(biased_exp_scale_inv);
}

// Quantize each BF16 token once into shared MXFP4/SFA, then scatter it to its top-k expert slots.
template <int HIDDEN, int TOPK, uint32_t MAX_TOKENS_PER_EXPERT_T,
          uint32_t NUM_RANKS_T, uint32_t NUM_TOTAL_EXPERTS_T,
          typename InputIndex = int32_t, bool HasActiveMask = false>
__device__ __forceinline__ void mxfp4_quantize_impl(
    const __ppu_bfloat16* __restrict__ input,    // [num_tokens, HIDDEN]
    const InputIndex* __restrict__ topk_ids,       // [num_tokens, topk]
    const float* __restrict__ topk_weights,
    int32_t* __restrict__ routing_topk_ids,
    float* __restrict__ routing_topk_weights,
    void* __restrict__ sym_buf_base,            // symmetric buffer base pointer
    uint32_t num_tokens,
    // The last block signals every consumer at the end of the active round.
    const layout::SymBuffer<>& sym_buffer,
    // Local staging base; peer staging is mapped through sym_buffer.
    uint8_t* __restrict__ local_staging,
    volatile uint32_t* __restrict__ preprocess_ready_flags,
    const bool* __restrict__ active_token_mask = nullptr) {

    static_assert(HIDDEN >= 2048 && HIDDEN % 1024 == 0,
                  "MegaMoE hidden must be a multiple of 1024 and at least 2048");
    constexpr int ELEMS_PER_THREAD = 16; // 16 BF16 values = 2 int4 loads
    constexpr int SCALE_GROUP_SIZE = 32;
    constexpr int LANES_PER_GROUP = SCALE_GROUP_SIZE / ELEMS_PER_THREAD;  // 2
    constexpr int QUANT_CHUNKS = HIDDEN / ELEMS_PER_THREAD;
    constexpr int FP4_INTS = HIDDEN / 8;  // number of int-sized FP4 output words
    constexpr int K_BLOCKS = HIDDEN / 32;
    constexpr int K_SCALE_BLOCKS = K_BLOCKS / 2;
    // For HIDDEN=7168, seven warps quantize two chunks per lane while warp 7 claims slots.
    constexpr int QUANT_WARPS = 7;
    constexpr int QUANT_THREADS = QUANT_WARPS * 32;
    constexpr bool SPECIALIZE_ATOMIC_WARP = (QUANT_CHUNKS % QUANT_THREADS) == 0;
    static_assert(MAX_TOKENS_PER_EXPERT_T > 0,
                  "max_tokens_per_expert must be positive");
    static_assert(TOPK >= 1 && TOPK <= 16,
                  "TOPK must be in [1, 16]");
    static_assert(is_supported_megamoe_rank_count(NUM_RANKS_T),
                  "quant requires a power-of-two rank count in [1, 32]");
    static_assert(NUM_TOTAL_EXPERTS_T > 0 &&
                      NUM_TOTAL_EXPERTS_T % NUM_RANKS_T == 0,
                  "quant requires a positive uniform expert shard");
    constexpr uint32_t kNumLocalExperts =
        NUM_TOTAL_EXPERTS_T / NUM_RANKS_T;
    constexpr uint32_t max_tokens_per_expert =
        MAX_TOKENS_PER_EXPERT_T;
    constexpr uint32_t num_ranks = NUM_RANKS_T;
    constexpr uint32_t num_total_experts = NUM_TOTAL_EXPERTS_T;
    const uint32_t rank_idx = sym_buffer.rank_idx;
    constexpr uint32_t dedicated_count_blocks = kNumLocalExperts;
    __shared__ uint32_t is_last_retiring_cta;
    uint32_t generation = 0;
    __shared__ uint32_t device_round_generation;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const bool dedicated_count_cta =
        blockIdx.x < dedicated_count_blocks;

    DispatchBufferLayout layout(num_total_experts,
                                max_tokens_per_expert, HIDDEN);
    void* true_sym_buf_base = sym_buf_base;
    // CTA leaders derive the next generation from the preceding GEMM's state.
    constexpr int kClaimWarp =
        SPECIALIZE_ATOMIC_WARP ? QUANT_WARPS : 0;
    const bool generation_leader = dedicated_count_cta
        ? threadIdx.x == 0
        : warp_id == kClaimWarp && lane_id == 0;
    uint32_t current = 0u;
    if (generation_leader) {
        // The preceding GEMM is stream-ordered before this kernel, so its state is
        // visible at entry; this one-shot handoff needs only a cached volatile load.
        current = *preprocess_ready_flags &
                  ~kPreprocessMetadataReadyBit;
        // Device-owned generation wraps to 2, preserving parity during Graph replay.
        current = current >= kFusedRoundGenerationLimit
            ? 2u : current + 1u;
        device_round_generation = current;
    }
    if (!dedicated_count_cta && warp_id == kClaimWarp)
        generation = __shfl_sync(0xffffffffu, current, 0);

    const bool claim_lane = SPECIALIZE_ATOMIC_WARP
        ? (warp_id == QUANT_WARPS && lane_id < TOPK)
        : (threadIdx.x < TOPK);

    // Shared memory: quantized FP4 data + packed scales.
    __shared__ __align__(16) int s_fp4_data[FP4_INTS];
    __shared__ __align__(16) uint16_t s_packed_scale[K_SCALE_BLOCKS];  // 16B-aligned for int4 push
    __shared__ uint32_t s_slot[TOPK];
    __shared__ int s_expert[TOPK];
    // Each dedicated count CTA accumulates one local expert across all ranks.
    constexpr uint32_t kCountOwners = NUM_RANKS_T;
    __shared__ uint32_t s_pushed_counts[kCountOwners];

    // Initialize this CTA-local flag before its first handoff at the Phase-3 barrier.
    if (threadIdx.x == 0)
        is_last_retiring_cta = 0;

    const int token_block_idx = static_cast<int>(blockIdx.x) -
        static_cast<int>(dedicated_count_blocks);
    const int token_block_count = static_cast<int>(gridDim.x) -
        static_cast<int>(dedicated_count_blocks);

    // One CTA per local expert publishes route counts, released with payloads in Phase 3.
    if (dedicated_count_cta) {
        if (threadIdx.x < num_ranks)
            s_pushed_counts[threadIdx.x] = 0u;
        __syncthreads();
        generation = device_round_generation;

        const uint32_t local_expert = blockIdx.x;
        const uint32_t num_routes = num_tokens * TOPK;
        for (uint32_t i = threadIdx.x; i < num_routes;
             i += blockDim.x) {
            const auto expert_idx = (HasActiveMask && !active_token_mask[i / TOPK])
                ? InputIndex(-1) : __ldg(topk_ids + i);
            if (expert_idx >= 0 &&
                static_cast<uint64_t>(expert_idx) < num_total_experts) {
                const uint32_t owner =
                    static_cast<uint32_t>(expert_idx) / kNumLocalExperts;
                const uint32_t owner_local =
                    static_cast<uint32_t>(expert_idx) -
                    owner * kNumLocalExperts;
                if (owner_local == local_expert)
                    atomicAdd(s_pushed_counts + owner, 1u);
            }
        }
        __syncthreads();

        if (threadIdx.x < num_ranks) {
            const uint32_t owner = threadIdx.x;
            uint8_t* peer_staging = sym_buffer.map(local_staging, owner);
            uint32_t* dst = reinterpret_cast<uint32_t*>(
                peer_staging + layout.staging_count_offset(
                    generation & 1u)) +
                rank_idx * kNumLocalExperts + local_expert;
            // Arrival publishes this overwritten plain count to the consumer.
            __stbl(dst, s_pushed_counts[owner]);
        }
    }
    // Grid-stride token CTAs amortize one Phase-3 fence over all tokens handled by each block.
    for (int token_idx = token_block_idx;
         !dedicated_count_cta && token_idx < num_tokens;
         token_idx += token_block_count) {
    const __ppu_bfloat16* token_data = input + static_cast<int64_t>(token_idx) * HIDDEN;

    // Phase 0: claim unique valid top-k slots while other warps quantize.
    // Use warp 7 for specialized shapes, otherwise the first TOPK threads.
    if (claim_lane) {
        const int t = SPECIALIZE_ATOMIC_WARP ? lane_id : threadIdx.x;
        const int route_index = token_idx * TOPK + t;
        const auto expert_idx = (HasActiveMask && !active_token_mask[token_idx])
            ? InputIndex(-1) : topk_ids[route_index];
        const bool accepted = expert_idx >= 0 &&
            static_cast<uint64_t>(expert_idx) < num_total_experts;
        routing_topk_ids[route_index] = accepted ? expert_idx : -1;
        routing_topk_weights[route_index] = accepted
            ? __ldg(topk_weights + route_index)
            : 0.0f;
        s_expert[t] = accepted ? expert_idx : -1;
        if (accepted) {
            void* count_base = layout.parity_base(
                true_sym_buf_base, generation & 1u);
            uint32_t* counts = layout.expert_token_counts_ptr(count_base);
            // The last retiring CTA clears this parity after all slot claims.
            s_slot[t] = atomicAdd(&counts[expert_idx], 1u);
        } else {
            s_slot[t] = max_tokens_per_expert;
        }
    }

    // Phase 1: hoist int4 loads while quantizing each token once in shared memory.
    constexpr int THREADS = SPECIALIZE_ATOMIC_WARP ? QUANT_THREADS : 256;
    constexpr int P1_ITERS = (QUANT_CHUNKS + THREADS - 1) / THREADS;
    const int4* token_v = reinterpret_cast<const int4*>(token_data);
    int4 raw[P1_ITERS][2];
    #pragma unroll
    for (int it = 0; it < P1_ITERS; ++it) {
        const int i = it * THREADS + threadIdx.x;
        if (threadIdx.x < THREADS && i < QUANT_CHUNKS) {
            raw[it][0] = token_v[2 * i];
            raw[it][1] = token_v[2 * i + 1];
        }
    }
    #pragma unroll
    for (int it = 0; it < P1_ITERS; ++it) {
        const int i = it * THREADS + threadIdx.x;
        if (threadIdx.x >= THREADS || i >= QUANT_CHUNKS) continue;
        __ppu_bfloat162 local_v2[ELEMS_PER_THREAD / 2];

        __ppu_bfloat162 amax2 = __float2bfloat162_rn(0.0f);
        for (int j = 0; j < ELEMS_PER_THREAD / 2; ++j) {
            local_v2[j] = reinterpret_cast<const __ppu_bfloat162*>(&raw[it][j / 4])[j % 4];
            amax2 = __hmax2(amax2, __habs2(local_v2[j]));
        }
        __ppu_bfloat16 amax = __hmax(amax2.x, amax2.y);

        amax = __hmax(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 2));

        // Both lanes recompute the shared scale; only the leader stores scale_inv.
        __ppu_bfloat16 scale;
        uint8_t scale_inv;
        calculate_mxfp4_scales_bf16(amax, scale, scale_inv);
        if ((lane_id & (LANES_PER_GROUP - 1)) == 0) {
            const int scale_group = i / LANES_PER_GROUP;
            uint8_t* packed_scale_bytes =
                reinterpret_cast<uint8_t*>(s_packed_scale);
            packed_scale_bytes[scale_group] = scale_inv;
        }
        __ppu_bfloat162 scale2 = __halves2bfloat162(scale, scale);
        int2 int_value;
        uint8_t* fp4x2_values = reinterpret_cast<uint8_t*>(&int_value);
        for (int j = 0; j < ELEMS_PER_THREAD / 2; ++j) {
            __ppu_bfloat162 scaled = __hmul2(local_v2[j], scale2);
            float2 f2 = __bfloat1622float2(scaled);
            fp4x2_values[j] = cvt_f32x2_to_fp4x2(f2.y, f2.x);
        }

        reinterpret_cast<int2*>(s_fp4_data)[i] = int_value;
    }
    // Publish slot claims, quantized data, and directly packed scales together.
    __syncthreads();
    generation = device_round_generation;
    sym_buf_base = layout.parity_base(
        true_sym_buf_base, generation & 1u);

    // Issue remote SFA stores before local FP4 scatter; Phase 3 releases both payloads.
    const uint32_t nlr = num_total_experts / num_ranks;
    const uint64_t scale_parity_bytes = layout.staging_parity_bytes();
    // Count matrices and token bands follow the two scale parities.
    constexpr int kPayloadWarps = 8;
    constexpr int kDestinationRounds =
        (TOPK + kPayloadWarps - 1) / kPayloadWarps;
    // One destination row per warp; TOPK > 8 needs multiple rounds.
    #pragma unroll
    for (int round = 0; round < kDestinationRounds; ++round) {
      const int t = warp_id + round * kPayloadWarps;
      if (t < TOPK) {
        int e = s_expert[t];
        uint32_t slot = s_slot[t];
        if (e >= 0 && slot < max_tokens_per_expert) {
            uint32_t d  = static_cast<uint32_t>(e) / nlr;
            uint32_t le = static_cast<uint32_t>(e) - d * nlr;
            uint8_t* peer_base = sym_buffer.map(local_staging, d);
            uint16_t* staging = reinterpret_cast<uint16_t*>(
                peer_base + static_cast<uint64_t>(generation & 1u) *
                    scale_parity_bytes);
            uint64_t sidx =
                (static_cast<uint64_t>(le * num_ranks + rank_idx)
                 * max_tokens_per_expert + slot) * K_SCALE_BLOCKS;
            uint16_t* dst = staging + sidx;
            constexpr int KSB_I4 = K_SCALE_BLOCKS / 8;
            const int4* s4 = reinterpret_cast<const int4*>(s_packed_scale);
            int4* d4 = reinterpret_cast<int4*>(dst);
            for (int v = lane_id; v < KSB_I4; v += 32)
                __stbl(&d4[v], s4[v]);
            // Source-token entries share the scale payload's Phase-3 release and arrival.
            if (lane_id == 0) {
                uint32_t* token_dst = reinterpret_cast<uint32_t*>(
                    peer_base + layout.staging_token_offset(
                        generation & 1u));
                const uint64_t token_sidx =
                    static_cast<uint64_t>(le * num_ranks + rank_idx) *
                        max_tokens_per_expert +
                    slot;
                __stbl(token_dst + token_sidx,
                        static_cast<uint32_t>(token_idx * TOPK + t));
            }
        }
      }
    }

    // Phase 2: scatter quantized data to top-k expert slots.
    constexpr int FP4_INT4S = FP4_INTS / 4;
    const int4* s_fp4_v = reinterpret_cast<const int4*>(s_fp4_data);

    // One warp per destination per compile-time round preserves contiguous
    // 512-byte transactions without duplicating the payload body.
    #pragma unroll
    for (int round = 0; round < kDestinationRounds; ++round) {
      const int t = warp_id + round * kPayloadWarps;
      if (t < TOPK) {
        int expert_idx = s_expert[t];
        uint32_t slot = s_slot[t];

        if (expert_idx >= 0 && slot < max_tokens_per_expert) {
            uint8_t* fp4_row_base = layout.fp4_data_ptr(sym_buf_base, expert_idx);
            int4* fp4_out = reinterpret_cast<int4*>(
                fp4_row_base + static_cast<int64_t>(slot) * (HIDDEN >> 1));
            constexpr int FP4_WARP_ITERS = (FP4_INT4S + 31) / 32;
            #pragma unroll
            for (int it = 0; it < FP4_WARP_ITERS; it += 2) {
                const int i0 = lane_id + it * 32;
                const int i1 = i0 + 32;
                int4 fp4_0;
                int4 fp4_1;
                if (i0 < FP4_INT4S) fp4_0 = s_fp4_v[i0];
                if (i1 < FP4_INT4S) fp4_1 = s_fp4_v[i1];
                if (i0 < FP4_INT4S) fp4_out[i0] = fp4_0;
                if (i1 < FP4_INT4S) fp4_out[i1] = fp4_1;
            }
        }
      }
    }
    // No __syncthreads() needed: all shared inputs are read-only here.

    // Wait only when another grid-stride iteration will reuse shared memory.
    if (token_idx + token_block_count < num_tokens) {
        __syncthreads();
    }
    }  // end grid-stride token loop

    if (!dedicated_count_cta && num_tokens == 0) {
        // Empty token CTAs still need the device generation before joining retire.
        __syncthreads();
        generation = device_round_generation;
        sym_buf_base = layout.parity_base(
            true_sym_buf_base, generation & 1u);
    }


    // Phase 3: each CTA releases payloads; the last retire ticket signals consumers.
    // Consumers system-acquire the generation before reading payloads.
    uint32_t* round_retire_counter = layout.round_retire_counter_ptr(
        true_sym_buf_base, generation & 1u);
    // Each writer-warp leader releases its stores before grid retire.
    __syncwarp();
    // Count CTAs release all warps; token CTAs release every active payload warp.
    if ((threadIdx.x & 31) == 0 &&
        (dedicated_count_cta || warp_id < TOPK)) {
        // Peer stores use __stbl; release them at system scope without flushing L1.
        __ppu_threadfence_system_uncache();
    }
    // A scalar release drains one warp; this barrier joins all warp releases before retire.
    __syncthreads();

    if (threadIdx.x == 0) {
        // The last ticket resets the counter to zero for the next serialized round.
        const uint32_t ticket = atomicInc(
            round_retire_counter, gridDim.x - 1);
        if (ticket == gridDim.x - 1) {
            // The final ticket follows every CTA release, so it publishes without another fence.
            // Publish generation with ready cleared; fused GEMM restores it after metadata.
            __stbl(const_cast<uint32_t*>(preprocess_ready_flags),
                   generation);
            is_last_retiring_cta = 1;
        }
    }
    // This barrier hands the final-ticket flag to consumer-notification lanes.
    __syncthreads();
    if (is_last_retiring_cta) {
        // Counts are only slot allocators; retire clears them for parity reuse.
        void* parity_base = layout.parity_base(
            true_sym_buf_base, generation & 1u);
        uint32_t* counts =
            layout.expert_token_counts_ptr(parity_base);
        for (uint32_t expert = threadIdx.x;
             expert < num_total_experts; expert += blockDim.x)
            counts[expert] = 0u;
        if (threadIdx.x < num_ranks) {
            const uint32_t c = threadIdx.x;
            const uint64_t arrival_offset =
                layout.ready_flag_offset(generation & 1u, num_ranks);
            uint32_t* slot = reinterpret_cast<uint32_t*>(
                sym_buffer.map(static_cast<uint8_t*>(true_sym_buf_base), c) +
                arrival_offset) + rank_idx;
            // Match the owner's uncached polling load with a cache-bypassing store.
            __stbl(slot, generation);
        }
    }
}

}  // namespace deep_gemm
