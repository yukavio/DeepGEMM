#pragma once

#include <cstdint>
#include <deep_gemm/dispatch_layout.cuh>
#include <deep_gemm/common/utils_rtc.cuh>

namespace deep_gemm {

// ===========================================================================
// The sole overlap kernel composes two CTA-local metadata phases.
//
// Prepare is BLOCK_M-independent: wait for this generation, read the pushed
// count matrix once, transpose it to rank-inclusive prefixes, and produce the
// GroupedMasked per-expert counts. The prefix cache is expert-major, with one
// contiguous rank-prefix vector per local expert. Finalize is BLOCK_M-dependent:
// consume the cached prefixes to build the compact expert layout and publish
// that prefix table. Each copy CTA expands its current M-block into shared
// metadata. These are device helpers, not standalone prepare/finalize launches.
// ===========================================================================
template <uint32_t STATIC_NUM_RANKS_T,
          uint32_t STATIC_NUM_LOCAL_EXPERTS_T,
          uint32_t STATIC_NUM_THREADS_T>
__device__ __forceinline__ void dispatch_expert_prepare_core(
    const uint8_t* __restrict__ local_dispatch,
    const DispatchBufferLayout& buf_layout,
    uint32_t local_source_rank,
    uint32_t generation,
    uint32_t* __restrict__ pair_counts,
    uint32_t* __restrict__ masked_m,
    uint32_t* __restrict__ smem_masked_m,
    const uint32_t* __restrict__ local_pushed_counts) {

    // PPU platform assumption: observing the peer-visible volatile generation
    // is sufficient to order the producer data read below; no acquire fence.
    if (generation > 0 && threadIdx.x < STATIC_NUM_RANKS_T) {
        const volatile uint32_t* arrival_slots = reinterpret_cast<const volatile uint32_t*>(
            local_dispatch +
            buf_layout.ready_flag_offset(
                generation & 1u, STATIC_NUM_RANKS_T));
        while (arrival_slots[threadIdx.x] != generation) { }
    }
    // Publish completion of every source-rank arrival wait before any thread
    // starts reading remote counts.
    __syncthreads();

    constexpr uint32_t total_pairs =
        STATIC_NUM_LOCAL_EXPERTS_T * STATIC_NUM_RANKS_T;
    // The pushed matrix is rank-major at rest. Load each source/expert pair and
    // transpose it into expert-major shared memory so finalize and the NCB
    // direct-metadata path see one contiguous prefix vector per expert.
    // Compile-time rounds cover every topology with one implementation. The
    // current 1/2/4/8/16/32-rank matrices each fit in one 512-thread CTA; larger
    // future matrices naturally add rounds without a rank-name specialization.
    #pragma unroll
    for (uint32_t tid = threadIdx.x; tid < total_pairs;
         tid += STATIC_NUM_THREADS_T) {
        const uint32_t src_rank = tid / STATIC_NUM_LOCAL_EXPERTS_T;
        const uint32_t local_expert = tid % STATIC_NUM_LOCAL_EXPERTS_T;
        pair_counts[local_expert * STATIC_NUM_RANKS_T + src_rank] = __ldbl(
            local_pushed_counts +
            src_rank * STATIC_NUM_LOCAL_EXPERTS_T + local_expert);
    }
    __syncthreads();

    // Each expert's compile-time rank vector is contiguous. One/two ranks use the
    // scalar path because the vector contains less than one uint4. Existing
    // 4/8/16/32-rank specializations retain the vectorized implementation.
    for (uint32_t e = threadIdx.x; e < STATIC_NUM_LOCAL_EXPERTS_T;
         e += STATIC_NUM_THREADS_T) {
        uint32_t total = 0;
        if constexpr (STATIC_NUM_RANKS_T <= 2) {
            uint32_t* expert_counts =
                pair_counts + e * STATIC_NUM_RANKS_T;
            #pragma unroll
            for (uint32_t r = 0; r < STATIC_NUM_RANKS_T; ++r) {
                total += expert_counts[r];
                expert_counts[r] = total;
            }
        } else {
            uint4* expert_counts = reinterpret_cast<uint4*>(
                pair_counts + e * STATIC_NUM_RANKS_T);
            constexpr uint32_t kNumCountVecs = STATIC_NUM_RANKS_T / 4;
            constexpr uint32_t kBatchVecs = kNumCountVecs < 4 ? 1 : 4;
            #pragma unroll
            for (uint32_t base = 0; base < kNumCountVecs;
                 base += kBatchVecs) {
                uint4 counts[kBatchVecs];

                #pragma unroll
                for (uint32_t v = 0; v < kBatchVecs; ++v)
                    counts[v] = expert_counts[base + v];

                #pragma unroll
                for (uint32_t v = 0; v < kBatchVecs; ++v) {
                    total += counts[v].x; counts[v].x = total;
                    total += counts[v].y; counts[v].y = total;
                    total += counts[v].z; counts[v].z = total;
                    total += counts[v].w; counts[v].w = total;
                }

                #pragma unroll
                for (uint32_t v = 0; v < kBatchVecs; ++v)
                    expert_counts[base + v] = counts[v];
            }
        }
        masked_m[e] = total;
        smem_masked_m[e] = total;
    }
    __syncthreads();

}

template <uint32_t BLOCK_M, uint32_t NUM_RANKS_T,
          uint32_t NUM_LOCAL_EXPERTS_T, uint32_t NUM_THREADS_T>
__device__ __forceinline__ void dispatch_expert_finalize_core(
    const uint32_t* __restrict__ pair_counts,
    int* __restrict__ grouped_layout,
    uint32_t* __restrict__ expert_rank_prefix,
    uint32_t* __restrict__ expert_base,
    uint32_t* smem_workspace) {

    uint32_t* expert_m_blocks_arr = smem_workspace;
    uint32_t* expert_cumsum_blocks =
        expert_m_blocks_arr + NUM_LOCAL_EXPERTS_T;
    uint32_t* expert_cumsum_m =
        expert_cumsum_blocks + NUM_LOCAL_EXPERTS_T;
    const uint32_t* masked_m_values =
        expert_cumsum_m + NUM_LOCAL_EXPERTS_T;

    if (threadIdx.x < NUM_LOCAL_EXPERTS_T) {
        uint32_t e = threadIdx.x;
        uint32_t total = masked_m_values[e];
        expert_m_blocks_arr[e] = (total + BLOCK_M - 1) / BLOCK_M;
    }
    __syncthreads();

    if constexpr (NUM_LOCAL_EXPERTS_T <= 32) {
        // Preserve the production single-warp scan exactly for the 12-expert path.
        if (threadIdx.x < 32) {
            uint32_t lane = threadIdx.x;
            uint32_t own_b = (lane < NUM_LOCAL_EXPERTS_T)
                ? expert_m_blocks_arr[lane] : 0u;
            uint32_t own_m = (lane < NUM_LOCAL_EXPERTS_T)
                ? masked_m_values[lane] : 0u;
            uint32_t scan_b = own_b;
            uint32_t scan_m = own_m;
            #pragma unroll
            for (uint32_t offset = 1; offset < 32; offset <<= 1) {
                uint32_t up_b = __shfl_up_sync(0xffffffff, scan_b, offset);
                uint32_t up_m = __shfl_up_sync(0xffffffff, scan_m, offset);
                if (lane >= offset) {
                    scan_b += up_b;
                    scan_m += up_m;
                }
            }
            if (lane < NUM_LOCAL_EXPERTS_T) {
                expert_cumsum_blocks[lane] = scan_b - own_b;
                expert_cumsum_m[lane] = scan_m - own_m;
                if (lane + 1 == NUM_LOCAL_EXPERTS_T) {
                    grouped_layout[0] = scan_b;
                }
            }
        }
    } else if (threadIdx.x == 0) {
        // Large expert shards are metadata-light (one pass per round), so a
        // deterministic scalar scan avoids a second global synchronization
        // protocol while retaining the exact compact layout contract.
        uint32_t block_prefix = 0;
        uint32_t row_prefix = 0;
        for (uint32_t e = 0; e < NUM_LOCAL_EXPERTS_T; ++e) {
            expert_cumsum_blocks[e] = block_prefix;
            expert_cumsum_m[e] = row_prefix;
            block_prefix += expert_m_blocks_arr[e];
            row_prefix += masked_m_values[e];
        }
        grouped_layout[0] = block_prefix;
    }
    __syncthreads();

    // Device-fused copy CTAs only need one expert's rank prefixes at a time.
    // Publish the compact expert-major prefix table; each NCB expands its own
    // m-block in shared memory.
    constexpr uint32_t total_pairs =
        NUM_LOCAL_EXPERTS_T * NUM_RANKS_T;
    for (uint32_t pair = threadIdx.x; pair < total_pairs;
         pair += NUM_THREADS_T)
        expert_rank_prefix[pair] = pair_counts[pair];

    for (uint32_t e = threadIdx.x; e < NUM_LOCAL_EXPERTS_T;
         e += NUM_THREADS_T) {
        const uint32_t total_count = masked_m_values[e];
        const uint32_t num_mblocks = expert_m_blocks_arr[e];
        const uint32_t base_block = expert_cumsum_blocks[e];
        const uint32_t base_m = expert_cumsum_m[e];
        expert_base[e] = base_m;
        if (e + 1 == NUM_LOCAL_EXPERTS_T)
            expert_base[NUM_LOCAL_EXPERTS_T] = base_m + total_count;
        for (uint32_t mb = 0; mb < num_mblocks; ++mb) {
            uint4 entry;
            entry.x = e;
            entry.y = total_count;
            entry.z = base_block;
            entry.w = base_m;
            reinterpret_cast<uint4*>(grouped_layout + 4)[base_block + mb] =
                entry;
        }
    }

}

// ===========================================================================
// Pre-GEMM overlap path for pushed SFA.
//
// SFA reshape does not require the materialized per-M-block metadata.  For one
// expert, the source rank counts alone define the same rank-concatenated row
// layout: rank r starts at prefix_sum(count[0:r]).  Reshape CTAs therefore wait
// for the peer-arrival publication, read the owner-local pushed count matrix,
// derive its prefix locally, and transpose pushed row-major SFA into
// local_sfa_buf while CTA 0 builds the normal FP4/GEMM metadata.
//
// Keep the row transform as a device helper so a future larger pre-GEMM kernel
// can compose it with different CTA-role or synchronization policies.
// ===========================================================================
template <uint32_t NUM_RANKS_T,
          uint32_t NUM_TOPK_T,
          uint32_t NUM_LOCAL_EXPERTS_T,
          uint32_t MAX_TOKENS_PER_EXPERT_T,
          uint32_t HIDDEN_DIM_T,
          uint32_t NUM_THREADS_T>
__device__ __forceinline__ void dispatch_expert_reshape_pushed_sfa_device(
    const uint8_t* __restrict__ local_staging,
    uint32_t local_source_rank,
    uint32_t generation,
    uint32_t expert_local,
    uint32_t partition_idx,
    uint16_t* __restrict__ local_sfa_buf,
    uint32_t* smem_rank_counts,
    const uint32_t* __restrict__ expert_base,
    volatile uint32_t* preprocess_ready_flags,
    uint32_t metadata_ready_value,
    void* __restrict__ src_info)
{
    constexpr uint32_t kScaleBlocks =
        DispatchBufferLayout::k_scale_blocks_for_hidden(HIDDEN_DIM_T);
    constexpr uint32_t kBalancedCountPerSource = 32 / NUM_RANKS_T;
    static_assert(FusedSfaPartitionLayout::kNumPartitions == 3,
                  "preprocess readiness is three K partitions per expert");
    DispatchBufferLayout layout(NUM_RANKS_T * NUM_LOCAL_EXPERTS_T,
                                MAX_TOKENS_PER_EXPERT_T, HIDDEN_DIM_T);
    uint32_t loaded_count_for_balance = 2;
    if (threadIdx.x < NUM_RANKS_T) {
        const uint32_t source_rank = threadIdx.x;
        const uint32_t* pushed_counts = reinterpret_cast<const uint32_t*>(
            local_staging + layout.staging_count_offset(generation & 1u));
        const uint32_t* count_entry = pushed_counts +
            source_rank * NUM_LOCAL_EXPERTS_T + expert_local;
        loaded_count_for_balance = __ldbl(count_entry);
        smem_rank_counts[source_rank] = loaded_count_for_balance;
    }
    if (threadIdx.x < 32) {
        const uint32_t count_lane = threadIdx.x;
        const bool lane_is_balanced =
            count_lane >= NUM_RANKS_T ||
            loaded_count_for_balance == kBalancedCountPerSource;
        const uint32_t balanced_mask =
            __ballot_sync(0xffffffffu, lane_is_balanced);
        if (count_lane == 0)
            smem_rank_counts[NUM_RANKS_T] =
                balanced_mask == 0xffffffffu;
    }
    __syncthreads();

    const uint16_t* staging = reinterpret_cast<const uint16_t*>(
        local_staging + static_cast<uint64_t>(generation & 1u) *
        layout.staging_parity_bytes());
    uint32_t compact_base_m = 0;
    if (threadIdx.x == 0) {
        while (*preprocess_ready_flags != metadata_ready_value) { }
        compact_base_m = __ldg(expert_base + expert_local);
        smem_rank_counts[NUM_RANKS_T + 1] = compact_base_m;
    }
    __syncthreads();
    compact_base_m = smem_rank_counts[NUM_RANKS_T + 1];
    // The three ready flags consumed by fused GEMM1 describe disjoint
    // K partitions; these CTAs must not be interpreted as source-rank cohorts.
    constexpr uint32_t kWarpThreads = 32;
    constexpr uint32_t kWarps = NUM_THREADS_T / kWarpThreads;
    const uint32_t partition_begin =
        FusedSfaPartitionLayout::partition_begin_vector(
            kScaleBlocks, partition_idx);
    const uint32_t partition_end =
        FusedSfaPartitionLayout::partition_begin_vector(
            kScaleBlocks, partition_idx + 1);
    uint32_t total_count = 0;
    #pragma unroll
    for (uint32_t r = 0; r < NUM_RANKS_T; ++r)
        total_count += smem_rank_counts[r];

    // base_m is an exclusive row prefix; each expert's scale matrix is
    // contiguous in the compact buffer.
    uint16_t* dst_expert = local_sfa_buf +
        static_cast<uint64_t>(compact_base_m) * kScaleBlocks;

    const uint32_t warp = threadIdx.x / kWarpThreads;
    const uint32_t lane = threadIdx.x % kWarpThreads;
    const bool exact_balanced = smem_rank_counts[NUM_RANKS_T] != 0;

    const uint32_t* token_base = reinterpret_cast<const uint32_t*>(
        local_staging + layout.staging_token_offset(generation & 1u));
    for (uint32_t row = lane; row < total_count; row += kWarpThreads) {
        uint32_t source_rank;
        uint32_t rank_begin;
        if (exact_balanced) {
            source_rank = row / kBalancedCountPerSource;
            rank_begin = source_rank * kBalancedCountPerSource;
        } else {
            source_rank = 0;
            rank_begin = 0;
            uint32_t prefix = 0;
            bool found = false;
            #pragma unroll
            for (uint32_t r = 0; r < NUM_RANKS_T; ++r) {
                const uint32_t rank_end = prefix + smem_rank_counts[r];
                if (!found && row < rank_end) {
                    source_rank = r;
                    rank_begin = prefix;
                    found = true;
                }
                prefix = rank_end;
            }
        }
        const uint32_t source_token = row - rank_begin;
        if (partition_idx == 0 && warp == 0) {
            const uint32_t* token_src = token_base +
                static_cast<uint64_t>(
                    expert_local * NUM_RANKS_T + source_rank) *
                    MAX_TOKENS_PER_EXPERT_T;
            const uint32_t token_topk = __ldbl(token_src + source_token);
            const uint32_t token = token_topk / NUM_TOPK_T;
            const uint32_t topk = token_topk % NUM_TOPK_T;
            auto* token_dst = static_cast<TokenSrcMetadata*>(src_info) +
                compact_base_m;
            token_dst[row] = {source_rank, token, topk};
        }
        const uint64_t src_index =
            (static_cast<uint64_t>(
                 expert_local * NUM_RANKS_T + source_rank) *
             MAX_TOKENS_PER_EXPERT_T) * kScaleBlocks;
        const uint16_t* src = staging + src_index +
            static_cast<uint64_t>(source_token) * kScaleBlocks;
        uint16_t* dst = dst_expert + row;
        for (uint32_t vec_k = partition_begin + warp;
             vec_k < partition_end; vec_k += kWarps) {
            int4 value = __ldbl(
                reinterpret_cast<const int4*>(src) + vec_k);
            const uint16_t* elems = reinterpret_cast<const uint16_t*>(&value);
            const uint32_t kb0 = vec_k * 8;
            #pragma unroll
            for (uint32_t e = 0; e < 8; ++e) {
                uint16_t* dst_elem = dst +
                    static_cast<uint64_t>(kb0 + e) * total_count;
                __stbl(dst_elem, elems[e]);
            }
        }
    }
}

template <uint32_t BLOCK_M,
          uint32_t NUM_RANKS_T,
          uint32_t NUM_LOCAL_EXPERTS_T,
          uint32_t NUM_TOTAL_EXPERTS_T,
          uint32_t MAX_TOKENS_PER_EXPERT_T,
          uint32_t HIDDEN_DIM_T,
          uint32_t NUM_THREADS_T>
__device__ __forceinline__ void
dispatch_expert_prepare_finalize_overlap_device(
    const uint8_t* __restrict__ local_dispatch,
    uint32_t local_source_rank,
    uint32_t generation,
    int* __restrict__ grouped_layout,
    uint32_t* __restrict__ expert_rank_prefix,
    uint32_t* __restrict__ expert_base,
    uint32_t* __restrict__ masked_m,
    uint32_t* smem_workspace,
    const uint32_t* __restrict__ local_pushed_counts) {

    constexpr uint32_t total_pairs =
        NUM_LOCAL_EXPERTS_T * NUM_RANKS_T;
    uint32_t* smem_pair_counts = smem_workspace;
    uint32_t* finalize_smem = smem_workspace + total_pairs;
    uint32_t* smem_masked_m = finalize_smem + 3 * NUM_LOCAL_EXPERTS_T;

    // prepare_core's final barrier hands pair counts to finalize_core.
    DispatchBufferLayout buf_layout(
        NUM_TOTAL_EXPERTS_T, MAX_TOKENS_PER_EXPERT_T, HIDDEN_DIM_T);

    dispatch_expert_prepare_core<
        NUM_RANKS_T,
        NUM_LOCAL_EXPERTS_T, NUM_THREADS_T>(
        local_dispatch, buf_layout, local_source_rank, generation,
        smem_pair_counts, masked_m,
        smem_masked_m, local_pushed_counts);
    dispatch_expert_finalize_core<
        BLOCK_M, NUM_RANKS_T, NUM_LOCAL_EXPERTS_T, NUM_THREADS_T>(
        smem_pair_counts, grouped_layout, expert_rank_prefix, expert_base,
        finalize_smem);
}

}  // namespace deep_gemm
