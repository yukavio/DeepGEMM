#pragma once

#include <hggc_runtime.h>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/expert_preprocess.cuh>
#include <deep_gemm/mma/fp4_mma.cuh>

namespace deep_gemm {

// Copy CTAs reuse GEMM shared storage for the current M-block's rank metadata.
template <uint32_t BlockM, uint32_t NumRanks>
struct alignas(16) CopyRankMetadataShared {
    uint32_t rank_prefix[NumRanks];
    // Expand source rows once per block so every K-stripe can reuse the addresses.
    // Destination rows are already contiguous.
    uint64_t row_addr[BlockM];
};

__device__ __forceinline__ uint32_t resolve_fused_round_generation(
    const MegaMoESchedulerArguments& sched) {
    // Quant is stream-ordered before GEMM, so a single cached state load is enough.
    return *sched.preprocess_ready_flags & ~kPreprocessMetadataReadyBit;
}

// Copy CTAs expand CTA0's expert-major rank prefixes into shared row metadata.
template <uint32_t BLOCK_M, uint32_t NumRanks,
          uint32_t NumLocalExperts, uint32_t SHAPE_K,
          uint32_t SourceCapacity>
__device__ __forceinline__ void stage_direct_rank_metadata(
    const MegaMoESchedulerArguments& sched, uint32_t expert_local,
    uint32_t m_block_in_expert,
    CopyRankMetadataShared<BLOCK_M, NumRanks>& metadata,
    uint32_t generation) {
    constexpr uint32_t kNumTotalExperts = NumRanks * NumLocalExperts;
    constexpr uint32_t kHidden = SHAPE_K * 2;
    const uint32_t kLocalExpertStart =
        sched.local_source_rank * NumLocalExperts;
    const uint32_t global_expert = kLocalExpertStart + expert_local;
    DispatchBufferLayout layout(
        kNumTotalExperts, SourceCapacity, kHidden);

    if (threadIdx.x < 32) {
        const uint32_t source_rank = threadIdx.x;
        const uint32_t rank_end = source_rank < NumRanks ? __ldg(
            sched.expert_rank_prefix +
            expert_local * NumRanks + source_rank) : 0u;
        const uint32_t predecessor =
            __shfl_up_sync(0xffffffffu, rank_end, 1);
        const uint32_t rank_begin = source_rank == 0 ? 0u : predecessor;
        uint32_t block_count = 0;
        uint64_t block_addr = 0;
        if (source_rank < NumRanks) {
            const uint32_t block_begin = m_block_in_expert * BLOCK_M;
            const uint32_t begin = max(rank_begin, block_begin);
            const uint32_t end = min(rank_end, block_begin + BLOCK_M);
            if (begin < end) {
                // Static parameter indices preserve scalar ready-flag loads on PPU.
                int64_t peer_offset = 0;
                #pragma unroll
                for (uint32_t peer = 0; peer < NumRanks; ++peer) {
                    if (source_rank == peer)
                        peer_offset = sched.sym_buffer.offsets[peer];
                }
                void* peer_base = reinterpret_cast<void*>(
                    reinterpret_cast<int64_t>(sched.local_dispatch) + peer_offset);
                void* peer_parity = layout.parity_base(
                    peer_base, generation & 1u);
                uint8_t* remote_fp4 =
                    layout.fp4_data_ptr(peer_parity, global_expert);
                block_addr = reinterpret_cast<uint64_t>(
                    remote_fp4 + static_cast<uint64_t>(begin - rank_begin) *
                        SHAPE_K);
                block_count = end - begin;
            }
        }
        // Prefix real rows, including empty ranks, to keep group intervals stable.
        uint32_t block_rank_end = block_count;
        for (uint32_t offset = 1; offset < 32; offset *= 2) {
            const uint32_t prior =
                __shfl_up_sync(0xffffffffu, block_rank_end, offset);
            if (source_rank >= offset) block_rank_end += prior;
        }
        if (source_rank < NumRanks)
            metadata.rank_prefix[source_rank] = block_rank_end;
        // The copy path always consumes the real-row table.
        const bool has_hot_rank =
            __ballot_sync(0xffffffffu, block_count > 16) != 0;
        if (!has_hot_rank) {
            // These values are already live, so fill the row table before the CTA barrier.
            const uint32_t block_row_begin = block_rank_end - block_count;
            for (uint32_t row = 0; row < block_count; ++row) {
                metadata.row_addr[block_row_begin + row] =
                    block_addr + static_cast<uint64_t>(row) * SHAPE_K;
            }
        } else {
            // A hot rank can own most rows, so spread rows across the warp and find
            // each owner from the prefix while keeping the same row table.
            const uint32_t rows_total = __shfl_sync(
                0xffffffffu, block_rank_end, NumRanks - 1);
            const uint32_t addr_lo = static_cast<uint32_t>(block_addr);
            const uint32_t addr_hi = static_cast<uint32_t>(block_addr >> 32);
            for (uint32_t row_base = 0; row_base < rows_total;
                 row_base += 32) {
                const uint32_t row = row_base + source_rank;
                const uint32_t query_row = min(row, rows_total - 1);
                uint32_t owner = 0;
                for (uint32_t step = NumRanks / 2; step > 0; step /= 2) {
                    const uint32_t candidate = min(owner + step, NumRanks);
                    const uint32_t candidate_end = __shfl_sync(
                        0xffffffffu, block_rank_end, candidate - 1);
                    if (query_row >= candidate_end) owner = candidate;
                }
                // All lanes execute the full-mask shuffle; owner0's predecessor is zero.
                const uint32_t owner_predecessor = __shfl_sync(
                    0xffffffffu, block_rank_end,
                    owner == 0 ? 0u : owner - 1);
                const uint32_t owner_begin =
                    owner == 0 ? 0u : owner_predecessor;
                const uint64_t owner_addr =
                    static_cast<uint64_t>(__shfl_sync(
                        0xffffffffu, addr_hi, owner)) << 32 |
                    __shfl_sync(0xffffffffu, addr_lo, owner);
                if (row < rows_total) {
                    metadata.row_addr[row] = owner_addr +
                        static_cast<uint64_t>(row - owner_begin) * SHAPE_K;
                }
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void st_bulk_global(int4* ptr, const int4& v) {
    __ppu_global_stbl_bulk_b32x4(ptr, *reinterpret_cast<const uint4*>(&v));
}

// Read the CTA row table and pick the largest window that fits 16 loads.
__device__ __host__ inline constexpr uint32_t pbulk_row_depth(uint32_t cpr) {
    constexpr uint32_t kMaxBulkLoadsPerWarp = 16;
    uint32_t d = 1;
    while (cpr != 0 && d * 2u * cpr <= kMaxBulkLoadsPerWarp) d *= 2u;
    return d;
}

template <bool UseRemote>
__device__ __forceinline__ int4 ld_bulk_class(const int4* ptr) {
    uint4 v;
    if constexpr (UseRemote)
        v = __ppu_remote_load_bulk_b32x4(ptr);
    else
        v = __ppu_global_ldcg_bulk_b32x4(ptr);
    return *reinterpret_cast<const int4*>(&v);
}

// All rank tiers share this body. group_base picks the group, while rank opcodes
// and window sizes stay compile-time.
template <bool UseRemote, uint32_t Cpr, uint32_t RowDepth>
__device__ __forceinline__ void copy_rows_window_bulk_staged_runtime_group(
    const uint64_t* row_addr, int4* block_dst,
    uint32_t group_row_begin, uint32_t local_row_offset,
    uint32_t local_row_count, uint32_t row0, uint32_t row_stride,
    uint32_t pitch, uint32_t col_base, uint32_t lane) {
    constexpr uint32_t kBulkLanes = 32;
    const int4* s[RowDepth];
    int4* d[RowDepth];
    cute::for_each(cute::make_int_sequence<RowDepth>{}, [&](auto r) {
        constexpr uint32_t kR = decltype(r)::value;
        const uint32_t compact_row = row0 + kR * row_stride;
        uint32_t global_row;
        if constexpr (UseRemote) {
            global_row = group_row_begin + compact_row;
            // A group without the local source rank passes local_row_count=0,
            // so the select collapses to the contiguous remote interval.
            global_row +=
                local_row_count != 0 && compact_row >= local_row_offset
                    ? local_row_count : 0u;
        } else {
            global_row = group_row_begin + local_row_offset + compact_row;
        }
        s[kR] = reinterpret_cast<const int4*>(row_addr[global_row]) +
            col_base + lane;
        d[kR] = block_dst + global_row * pitch + col_base + lane;
    });
    int4 v[RowDepth][Cpr];
    cute::for_each(cute::make_int_sequence<RowDepth>{}, [&](auto r) {
        constexpr uint32_t kR = decltype(r)::value;
        cute::for_each(cute::make_int_sequence<Cpr>{}, [&](auto c) {
            constexpr uint32_t kC = decltype(c)::value;
            v[kR][kC] = ld_bulk_class<UseRemote>(
                s[kR] + kC * kBulkLanes);
        });
    });
    cute::for_each(cute::make_int_sequence<RowDepth>{}, [&](auto r) {
        constexpr uint32_t kR = decltype(r)::value;
        cute::for_each(cute::make_int_sequence<Cpr>{}, [&](auto c) {
            constexpr uint32_t kC = decltype(c)::value;
            st_bulk_global(d[kR] + kC * kBulkLanes, v[kR][kC]);
        });
    });
}

template <bool UseRemote, uint32_t Cpr,
          uint32_t RowDepth = pbulk_row_depth(Cpr)>
__device__ __forceinline__ void copy_group_bulk_rows_staged_runtime_group(
    const uint64_t* row_addr, int4* block_dst,
    uint32_t group_row_begin, uint32_t local_row_offset,
    uint32_t local_row_count, uint32_t rows_total, uint32_t pitch,
    uint32_t col_base, uint32_t lane, uint32_t row_stride,
    uint32_t& row) {
    for (; row + (RowDepth - 1) * row_stride < rows_total;
         row += RowDepth * row_stride) {
        copy_rows_window_bulk_staged_runtime_group<
            UseRemote, Cpr, RowDepth>(
                row_addr, block_dst, group_row_begin, local_row_offset,
                local_row_count, row, row_stride, pitch, col_base, lane);
    }
    if constexpr (RowDepth > 1) {
        copy_group_bulk_rows_staged_runtime_group<
            UseRemote, Cpr, RowDepth / 2>(
                row_addr, block_dst, group_row_begin, local_row_offset,
                local_row_count, rows_total, pitch, col_base, lane,
                row_stride, row);
    }
}

template <uint32_t FullStripeCpr, uint32_t LastStripeCpr,
          uint32_t SourceRankSpan = 8,
          uint32_t BlockM, uint32_t NumRanks>
__device__ __forceinline__ void copy_group_bulk_rank_ragged_staged_runtime_group(
    const CopyRankMetadataShared<BlockM, NumRanks>& metadata,
    int4* block_dst, uint32_t local_source_rank, uint32_t group_base,
    uint32_t stripe_int4s, uint32_t pitch, uint32_t col_base,
    uint32_t capacity_int4s, uint32_t worker_thread,
    uint32_t worker_stride) {
    constexpr uint32_t kBulkLanes = 32;
    // EP32 can use one M-block row pool; EP8/16 use group8 spans.
    constexpr uint32_t kGroupSize = SourceRankSpan;

    const uint32_t lane = worker_thread & (kBulkLanes - 1);
    const uint32_t warp = worker_thread / kBulkLanes;
    const uint32_t num_warps = worker_stride / kBulkLanes;
    const uint32_t group_row_begin = group_base == 0
        ? 0u : metadata.rank_prefix[group_base - 1];
    const uint32_t group_row_end =
        metadata.rank_prefix[group_base + kGroupSize - 1];
    const uint32_t rows_all = group_row_end - group_row_begin;
    if (rows_all == 0) return;

    // Rank is uniform for the launch. Keep the local/remote instruction choice
    // at warp ownership, outside the per-row bulk-load helpers.
    const bool group_contains_local =
        local_source_rank >= group_base &&
        local_source_rank < group_base + kGroupSize;
    const uint32_t local_row_begin = local_source_rank == 0
        ? 0u : metadata.rank_prefix[local_source_rank - 1];
    const uint32_t local_row_end = metadata.rank_prefix[local_source_rank];
    const uint32_t rows_local = group_contains_local
        ? local_row_end - local_row_begin : 0u;
    const uint32_t local_row_offset = group_contains_local
        ? local_row_begin - group_row_begin : 0u;
    const uint32_t rows_remote = rows_all - rows_local;

    static_assert(FullStripeCpr > 0 && LastStripeCpr > 0,
                  "copy stripes must contain at least one bulk chunk");
    const auto run_cpr = [&](auto cpr_tag) {
        constexpr uint32_t kCpr = decltype(cpr_tag)::value;

        // Keep the tuned warp ownership: shallow work can overlap, while deeper work
        // is split by row count and stripe width.
        constexpr uint32_t kLocalLoadsPerWarp = 16;
        constexpr uint32_t kRemoteLoadsPerWarp = 8;
        constexpr bool kWideTwoStripeSchedule = FullStripeCpr >= 3;
        const bool overlap_ownership =
            kWideTwoStripeSchedule || rows_all <= 16;
        uint32_t warps_remote;
        uint32_t warps_local;
        if (overlap_ownership) {
            warps_remote = rows_remote == 0 ? 0u :
                (rows_local == 0 ? num_warps :
                 min(num_warps, max(1u,
                     (rows_remote * kCpr + kRemoteLoadsPerWarp - 1) /
                     kRemoteLoadsPerWarp)));
            warps_local = rows_local == 0 ? 0u :
                (rows_remote == 0 ? num_warps :
                 min(num_warps, max(1u,
                     (rows_local * kCpr + kLocalLoadsPerWarp - 1) /
                     kLocalLoadsPerWarp)));
        } else if (rows_local == 0) {
            warps_remote = num_warps;
            warps_local = 0;
        } else if (rows_remote == 0) {
            warps_remote = 0;
            warps_local = num_warps;
        } else {
            warps_remote =
                (rows_remote * num_warps + rows_all / 2) / rows_all;
            warps_remote = max(1u, min(num_warps - 1, warps_remote));
            warps_local = num_warps - warps_remote;
        }
        if (warp < warps_remote) {
            uint32_t row = warp;
            copy_group_bulk_rows_staged_runtime_group<
                true, kCpr>(
                    metadata.row_addr, block_dst, group_row_begin,
                    local_row_offset, rows_local, rows_remote, pitch,
                    col_base, lane, warps_remote, row);
        }
        // Give local copy an idle warp pool when both pools fit.
        const uint32_t local_warp_begin =
            warps_remote + warps_local <= num_warps ? warps_remote : 0u;
        if (warp >= local_warp_begin &&
            warp < local_warp_begin + warps_local) {
            uint32_t row = warp - local_warp_begin;
            copy_group_bulk_rows_staged_runtime_group<
                false, kCpr>(
                    metadata.row_addr, block_dst, group_row_begin,
                    local_row_offset, rows_local, rows_local, pitch,
                    col_base, lane, warps_local, row);
        }
    };

    if (stripe_int4s == capacity_int4s) {
        run_cpr(cute::Int<FullStripeCpr>{});
    } else {
        run_cpr(cute::Int<LastStripeCpr>{});
    }
}

template <uint32_t BLOCK_M, uint32_t NumRanks,
          uint32_t NumLocalExperts, uint32_t BLOCK_K,
          uint32_t K_TILES_PER_FLAG, uint32_t SHAPE_K, uint32_t NumThreads,
          uint32_t SourceCapacity>
__device__ void run_copy_block(
    const MegaMoESchedulerArguments& sched, uint32_t total_m_blocks,
    char* smem_buf, uint32_t generation)
{
    constexpr uint32_t configured_stripe_capacity_int4s =
        K_TILES_PER_FLAG * BLOCK_K / 16;

    static_assert(SHAPE_K % 512 == 0,
                  "the packed K row must contain whole 32-int4 bulk waves");
    static_assert(BLOCK_K % 16 == 0,
                  "stripe capacity must be an exact int4 count: BLOCK_K must "
                  "be a multiple of 16");
    static_assert(configured_stripe_capacity_int4s % 32 == 0,
                  "each K stripe must contain a whole 32-int4 bulk wave");

    // Keep the scheduler load; constexpr adds extra scalar waits on 16 ranks.
    const uint32_t num_copy = sched.num_copy_blocks;
    constexpr uint32_t k_half = SHAPE_K;
    const uint32_t k_half_int4s = k_half / 16;
    constexpr uint32_t stripe_capacity_int4s =
        configured_stripe_capacity_int4s;
    // Ceil stripe counts so the last partial stripe publishes readiness.
    constexpr uint32_t num_stripes =
        ceil_div(k_half_int4s, stripe_capacity_int4s);
    constexpr uint32_t nr = NumRanks;
    auto& shared_rank_metadata =
        *reinterpret_cast<CopyRankMetadataShared<BLOCK_M, NumRanks>*>(smem_buf);
    constexpr uint32_t kPBulkLanes = 32;
    constexpr uint32_t kStaticFinalStripeInt4s =
        k_half_int4s - (num_stripes - 1) * stripe_capacity_int4s;
    constexpr uint32_t kFullStripeCpr =
        stripe_capacity_int4s / kPBulkLanes;
    constexpr uint32_t kFinalStripeCpr =
        kStaticFinalStripeInt4s / kPBulkLanes;

    // Copy-cohort geometry is fixed for this specialization.
    constexpr uint32_t kCopyRankSpan =
        nr < 8 || nr == 32 ? nr : 8;
    constexpr uint32_t kNumCopyGroups = nr / kCopyRankSpan;
    constexpr uint32_t kCtaWarps = NumThreads / 32;
    constexpr bool kCanSplitCopyGroups =
        kNumCopyGroups > 1 && NumThreads % 32 == 0 &&
        kCtaWarps % kNumCopyGroups == 0 &&
        kCtaWarps / kNumCopyGroups >= 2;
    constexpr uint32_t kNumCopyCohorts =
        kCanSplitCopyGroups ? kNumCopyGroups : 1;
    constexpr uint32_t kWarpsPerCohort =
        kCtaWarps / kNumCopyCohorts;
    constexpr uint32_t kThreadsPerCohort =
        kWarpsPerCohort * 32;

    for (uint32_t mb = blockIdx.x; mb < total_m_blocks; mb += num_copy) {
        uint4 gl = (reinterpret_cast<const uint4*>(sched.copy_grouped_layout) + 1)[mb];
        uint32_t expert_local = gl.x;
        uint32_t m_block_in_expert = mb - gl.z;

        stage_direct_rank_metadata<
            BLOCK_M, NumRanks, NumLocalExperts, SHAPE_K,
            SourceCapacity>(
                sched, expert_local, m_block_in_expert,
                shared_rank_metadata, generation);

        const uint64_t row_base = static_cast<uint64_t>(gl.w) +
            static_cast<uint64_t>(m_block_in_expert) * BLOCK_M;
        uint8_t* local_a_base =
            sched.local_fp4_buf + row_base * k_half;
        int4* local_block_dst = reinterpret_cast<int4*>(local_a_base);

        for (uint32_t ks = 0; ks < num_stripes; ++ks) {
            const uint32_t stripe_col_start =
                ks * stripe_capacity_int4s;
            // Clamp the last stripe; copying full capacity would overrun each row.
            const uint32_t stripe_int4s = min(
                stripe_capacity_int4s,
                k_half_int4s - stripe_col_start);
            uint32_t cohort = 0;
            if constexpr (kNumCopyCohorts > 1) {
                const uint32_t physical_warp =
                    __builtin_ppu_to_uniform_b32(__ppu_read_warpid());
                cohort = physical_warp / kWarpsPerCohort;
            }
            const uint32_t worker_thread =
                threadIdx.x - cohort * kThreadsPerCohort;
            for (uint32_t group = cohort; group < kNumCopyGroups;
                 group += kNumCopyCohorts) {
                copy_group_bulk_rank_ragged_staged_runtime_group<
                    kFullStripeCpr, kFinalStripeCpr, kCopyRankSpan>(
                        shared_rank_metadata,
                        local_block_dst, sched.local_source_rank,
                        group * kCopyRankSpan, stripe_int4s,
                        k_half_int4s, stripe_col_start,
                        stripe_capacity_int4s, worker_thread,
                        kThreadsPerCohort);
            }
            // Join all stripe stores before the CTA leader publishes readiness.
            __ppu_threadfence_uncache();
            __syncthreads();
            if (threadIdx.x == 0) {
                const uint32_t ready_value =
                    (generation << 3) | (ks + 1u);
                __stbl(const_cast<uint32_t*>(
                    sched.copy_ready_flags + mb), ready_value);
            }
        }
    }
}

template <class TileScheduler, uint32_t BlockM, uint32_t NumThreads>
__device__ __forceinline__ void run_rank_generic_device_preprocess_role(
    const MegaMoESchedulerArguments& sched, char* smem_buf,
    uint32_t generation) {
    constexpr uint32_t kNumRanks = TileScheduler::kNumRanks;
    constexpr uint32_t kNumLocalExperts = TileScheduler::kNumGroups;
    constexpr uint32_t kNumTotalExperts =
        kNumRanks * kNumLocalExperts;
    constexpr uint32_t kSourceCapacity =
        TileScheduler::kSourceCapacity;
    constexpr uint32_t kOwnerCapacity =
        kNumRanks * kSourceCapacity;
    constexpr uint32_t kHidden = TileScheduler::SHAPE_K * 2;
    constexpr uint32_t kRankSubgroups =
        FusedSfaPartitionLayout::kNumPartitions;
    constexpr uint32_t kReshapeWork =
        kNumLocalExperts * kRankSubgroups;

    volatile uint32_t* ready = sched.preprocess_ready_flags;
    auto* smem = reinterpret_cast<uint32_t*>(smem_buf);
    DispatchBufferLayout layout(
        kNumTotalExperts, kSourceCapacity, kHidden);
    const uint8_t* local_staging = sched.local_staging;
    const uint32_t* local_pushed_counts =
        reinterpret_cast<const uint32_t*>(
            local_staging + layout.staging_count_offset(generation & 1u));

    if (blockIdx.x == 0) {
        // Publish metadata after its stores are visible. Generation tags avoid
        // clearing readiness every round.
        dispatch_expert_prepare_finalize_overlap_device<
            BlockM, kNumRanks,
            kNumLocalExperts, kNumTotalExperts, kSourceCapacity,
            kHidden, NumThreads>(
                sched.local_dispatch, sched.local_source_rank, generation,
                sched.copy_grouped_layout,
                const_cast<uint32_t*>(sched.expert_rank_prefix),
                sched.expert_base,
                reinterpret_cast<uint32_t*>(sched.grouped_layout),
                smem, local_pushed_counts);

        if (generation == 2u) {
            // Tags are reused after wrap. Include dormant blocks in the shared
            // flag allocation, sized for the smallest workspace tile (BM16).
            constexpr uint32_t kNumCopyReadyFlags =
                kNumLocalExperts * ceil_div(kOwnerCapacity, 16u);
            for (uint32_t i = threadIdx.x; i < kNumCopyReadyFlags;
                 i += NumThreads)
                __stbl(const_cast<uint32_t*>(sched.copy_ready_flags + i), 0u);
            __ppu_threadfence_uncache();
        }

        // Fusion has no kernel boundary; flush cached metadata before publication.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
            __stbl(const_cast<uint32_t*>(ready),
                   kPreprocessMetadataReadyBit | generation);
    } else {
        volatile uint32_t* arrival_slots =
            reinterpret_cast<volatile uint32_t*>(
                sched.local_dispatch +
                layout.ready_flag_offset(generation & 1u, kNumRanks));

        // Compute CTAs reshape; copy CTAs only copy FP4.
        if (blockIdx.x >= TileScheduler::kNumCopyBlocks &&
            blockIdx.x - TileScheduler::kNumCopyBlocks < kReshapeWork) {
            if (threadIdx.x < kNumRanks) {
                const uint32_t source_rank = threadIdx.x;
                while (arrival_slots[source_rank] != generation) { }
            }
            const uint32_t work_stride =
                gridDim.x - TileScheduler::kNumCopyBlocks;
            for (uint32_t work = blockIdx.x - TileScheduler::kNumCopyBlocks;
                 work < kReshapeWork; work += work_stride) {
                const uint32_t expert_local = work / kRankSubgroups;
                const uint32_t partition_idx =
                    work - expert_local * kRankSubgroups;

                dispatch_expert_reshape_pushed_sfa_device<
                    kNumRanks,
                    TileScheduler::kNumTopK,
                    kNumLocalExperts, kSourceCapacity,
                    kHidden, NumThreads>(
                        sched.local_staging, sched.local_source_rank, generation,
                        expert_local, partition_idx,
                        sched.local_sfa_buf, smem,
                        sched.expert_base, ready,
                        kPreprocessMetadataReadyBit | generation,
                        sched.src_info);

                __ppu_threadfence_uncache();
                __syncthreads();
                if (threadIdx.x == 0) {
                    __stbl(const_cast<uint32_t*>(
                        ready + 1 + work), generation);
                }
            }
        }
    }

    // Copy CTAs wait for copy metadata; GEMM CTAs wait for masked_m. Per-expert
    // readiness avoids a grid barrier.
    if (threadIdx.x == 0) {
        const uint32_t metadata_ready =
            kPreprocessMetadataReadyBit | generation;
        while (deep_gemm::ld_copy_ready_uncached(ready) != metadata_ready) { }
    }
    // Join metadata and reshape readiness stores before returning.
    __syncthreads();
    __threadfence();

}

}  // namespace deep_gemm
