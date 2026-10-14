#pragma once

#include <cstdint>

namespace deep_gemm {

// preprocess_ready_flags[0] is also the graph-safe round handoff. Quant writes
// the raw generation; fused GEMM1 sets this bit after metadata is complete.
static constexpr uint32_t kPreprocessMetadataReadyBit = 1u << 31;
// Copy flags encode (generation << 3) | K-stripe watermark.
// Wrap to generation 2 after clearing flags, preserving parity alternation.
static constexpr uint32_t kFusedRoundGenerationLimit = (1u << 27) - 1u;

// Metadata owner search requires a power-of-two rank count.
__host__ __device__
static constexpr bool is_supported_megamoe_rank_count(uint32_t num_ranks) {
    return num_ranks >= 1 && num_ranks <= 32 &&
        (num_ranks & (num_ranks - 1)) == 0;
}

// MegaMoE-compatible per-row metadata used by GEMM2 direct scatter.
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};
static_assert(sizeof(TokenSrcMetadata) == 12);

// Map packed-scale producer partitions to GEMM K tiles.
// SFA readiness uses this mapping independently of FP4 copy stripes.
struct FusedSfaPartitionLayout {
    static constexpr uint32_t kNumPartitions = 3;
    // One int4 load/store in the reshaper carries eight uint16_t scale blocks.
    static constexpr uint32_t kScaleBlocksPerVector = 8;

    __host__ __device__
    static constexpr uint32_t partition_begin_vector(
            uint32_t scale_blocks, uint32_t partition) {
        const uint32_t vectors = scale_blocks / kScaleBlocksPerVector;
        return (vectors * partition + kNumPartitions - 1) /
            kNumPartitions;
    }

    __host__ __device__
    static constexpr uint32_t partition_begin_scale_block(
            uint32_t scale_blocks, uint32_t partition) {
        return partition_begin_vector(scale_blocks, partition) *
            kScaleBlocksPerVector;
    }

    __host__ __device__
    static constexpr uint32_t first_k_tile(
            uint32_t scale_blocks, uint32_t block_k,
            uint32_t partition) {
        // Use floor, not ceil: when a K tile straddles a producer boundary it
        // must wait for both partitions before issuing the SFA load.
        return partition_begin_scale_block(scale_blocks, partition) /
            (block_k / 32);
    }
};

// Each dispatch parity stores uint32 counts[num_total_experts], then
// FP4 bytes[num_total_experts][max_tokens_per_expert][hidden/2].
// Scales are pushed to a separate owner-local staging allocation.
struct DispatchBufferLayout {
    uint32_t num_total_experts;
    uint32_t max_tokens_per_expert;
    uint32_t hidden_dim;

    __host__ __device__
    DispatchBufferLayout() = default;

    __host__ __device__
    DispatchBufferLayout(uint32_t num_total_experts_,
                         uint32_t max_tokens_per_expert_, uint32_t hidden_dim_)
        : num_total_experts(num_total_experts_)
        , max_tokens_per_expert(max_tokens_per_expert_)
        , hidden_dim(hidden_dim_) {}

    // Metadata region: expert_token_counts array.
    __host__ __device__
    uint64_t metadata_bytes() const {
        // Align to 16 bytes for AIU descriptor requirements
        uint64_t raw = num_total_experts * sizeof(uint32_t);
        return (raw + 15) & ~uint64_t(15);
    }

    // FP4 data region: packed FP4 values (2 values per byte)
    __host__ __device__
    uint64_t fp4_data_bytes_per_expert() const {
        return uint64_t(max_tokens_per_expert) * (hidden_dim / 2);
    }

    __host__ __device__
    uint64_t fp4_data_bytes() const {
        return num_total_experts * fp4_data_bytes_per_expert();
    }

    // Each pushed scale is uint16_t packing two consecutive uint8 UE8M0
    // values. K dimension: ceil_div(hidden / 32, 2).
    __host__ __device__
    static constexpr uint32_t k_scale_blocks_for_hidden(uint32_t hidden_dim) {
        return hidden_dim / 64 + (hidden_dim % 64 != 0);
    }

    __host__ __device__
    uint32_t k_scale_blocks() const {
        return k_scale_blocks_for_hidden(hidden_dim);
    }

    __host__ __device__
    uint64_t total_bytes() const {
        return metadata_bytes() + fp4_data_bytes();
    }

    // Arrival synchronization bounds rank skew to one round; two parities prevent overwrites.
    __host__ __device__
    uint64_t parity_offset(uint32_t parity) const {
        return static_cast<uint64_t>(parity & 1u) * total_bytes();
    }

    // Base pointer of the parity buffer's data region.
    __host__ __device__
    void* parity_base(void* base, uint32_t parity) const {
        return static_cast<uint8_t*>(base) + parity_offset(parity);
    }

    __host__ __device__
    uint64_t round_retire_counter_offset() const {
        // Two retire counters padded to 16 bytes.
        return (2 * total_bytes() + 15) & ~uint64_t(15);
    }

    __host__ __device__
    uint32_t* round_retire_counter_ptr(void* base, uint32_t parity) const {
        return reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(base) + round_retire_counter_offset()) +
            (parity & 1u);
    }

    __host__ __device__
    static constexpr uint64_t arrival_vector_bytes(uint32_t num_ranks) {
        const uint64_t raw = static_cast<uint64_t>(num_ranks) * sizeof(uint32_t);
        return raw <= 128u ? 128u : (raw + 127u) & ~uint64_t(127);
    }

    __host__ __device__
    uint64_t ready_flag_offset(uint32_t parity, uint32_t num_ranks) const {
        // Separate arrival vectors prevent the next round from overwriting unread flags.
        const uint64_t vector_bytes = arrival_vector_bytes(num_ranks);
        return round_retire_counter_offset() + 16u +
               static_cast<uint64_t>(parity & 1u) * vector_bytes;
    }

    // Owner-local SFA: [parity][local_expert][source_rank][token][scale_block], uint16.
    __host__ __device__
    uint64_t staging_parity_bytes() const {
        return uint64_t(num_total_experts) * k_scale_blocks() * max_tokens_per_expert * sizeof(uint16_t);
    }

    // Owner-local counts[parity][source_rank][local_expert].
    // Each round overwrites counts; its arrival vector carries the generation.
    __host__ __device__
    uint64_t staging_count_parity_bytes() const {
        return uint64_t(num_total_experts) * sizeof(uint32_t);
    }

    __host__ __device__
    uint64_t staging_count_offset(uint32_t parity) const {
        return 2 * staging_parity_bytes() +
            static_cast<uint64_t>(parity & 1u) *
                staging_count_parity_bytes();
    }

    __host__ __device__
    uint64_t staging_token_offset(uint32_t parity) const {
        const uint64_t token_bytes =
            uint64_t(num_total_experts) * max_tokens_per_expert *
            sizeof(uint32_t);
        return 2 * staging_parity_bytes() +
            2 * staging_count_parity_bytes() +
            static_cast<uint64_t>(parity & 1u) * token_bytes;
    }

    // Offset getters
    __host__ __device__
    uint32_t* expert_token_counts_ptr(void* base) const {
        return static_cast<uint32_t*>(base);
    }

    __host__ __device__
    uint8_t* fp4_data_ptr(void* base, uint32_t expert_idx = 0) const {
        uint8_t* p = static_cast<uint8_t*>(base) + metadata_bytes();
        return p + expert_idx * fp4_data_bytes_per_expert();
    }

};

}  // namespace deep_gemm
