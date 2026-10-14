#pragma once

#include <cstdint>
#include <deep_gemm/dispatch_layout.cuh>

namespace deep_gemm::layout {

// Byte ranges only. Tensor dtype/shape/stride belong to the host API.
struct Buffer {
    int64_t offset, bytes;
};

struct MegaMoEBuffer {
    Buffer barrier_storage, copy_ready, dispatch, expert_base,
           expert_rank_prefix, grouped_layout, local_fp4, local_sfa, preprocess_ready,
           scatter_storage, src_info, stage1_fp4, stage1_scale_storage,
           staging, topk_idx, topk_weights;
    int64_t bytes = 0, groups, compact, blocks;

    Buffer allocate(int64_t size) {
        bytes = ((bytes + 255) / 256) * 256;
        const Buffer result{bytes, size};
        bytes += size;
        return result;
    }

    MegaMoEBuffer(int ranks, int experts, int capacity, int topk, int hidden, int intermediate) {
        const DispatchBufferLayout layout(experts, capacity, hidden);
        groups = experts / ranks;
        const int64_t owner = int64_t(ranks) * capacity;
        compact = owner * topk;
        blocks = groups * ((owner + 15) / 16);
        // Peer bases/offsets are kernel parameters, not regions in this buffer.
        barrier_storage = allocate(4 * sizeof(int32_t));
        copy_ready = allocate(blocks * sizeof(int32_t));
        dispatch = allocate(layout.ready_flag_offset(1, ranks) + layout.arrival_vector_bytes(ranks));
        expert_base = allocate((groups + 1) * sizeof(int32_t));
        expert_rank_prefix = allocate(groups * ranks * sizeof(int32_t));
        grouped_layout = allocate(((1 + blocks) * 4 + groups) * sizeof(int32_t));
        local_fp4 = allocate(compact * (hidden / 2));
        local_sfa = allocate(compact * layout.k_scale_blocks() * sizeof(uint16_t));
        preprocess_ready = allocate((1 + groups * FusedSfaPartitionLayout::kNumPartitions) * sizeof(int32_t));
        scatter_storage = allocate(int64_t(topk) * capacity * hidden * sizeof(uint16_t));
        src_info = allocate(compact * 3 * sizeof(int32_t));
        stage1_fp4 = allocate(compact * (intermediate / 2));
        stage1_scale_storage = allocate((intermediate / 64) * compact * sizeof(uint16_t));
        staging = allocate(layout.staging_token_offset(1) + int64_t(experts) * capacity * sizeof(uint32_t));
        topk_idx = allocate(int64_t(capacity) * topk * sizeof(int32_t));
        topk_weights = allocate(int64_t(capacity) * topk * sizeof(float));
        bytes = ((bytes + 255) / 256) * 256;
    }

    int64_t get_num_bytes() const { return bytes; }
};

}  // namespace deep_gemm::layout
