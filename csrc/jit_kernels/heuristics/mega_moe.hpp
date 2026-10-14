#pragma once

#include <numeric>
#include <optional>
#include "common_fp4.hpp"

namespace deep_gemm::mega {

struct MegaMoEStageConfig {
    int num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages;
    int smem_bytes;
};

struct MegaMoEStage1Config : MegaMoEStageConfig {
    int copy_blocks, k_tiles_per_flag;
};

struct MegaMoELaunchConfig {
    MegaMoEStage1Config l1;
    MegaMoEStageConfig l2;
};

struct MegaMoEHeuristicInput {
    int num_sms, num_ranks, num_experts_per_rank, num_max_tokens_per_rank;
    int hidden, intermediate_hidden, expected_m;
};

static MegaMoEStageConfig make_stage_config(const ConfigTuple& config) {
    const auto [num_sms, bm, bn, bk, wm, wn, stages, smem] = config;
    return {num_sms, bm, bn, bk, wm, wn, stages, std::get<0>(smem)};
}

struct CopyTile {
    int bm, bn, bk, wm, wn, stages, copy_blocks, ktpf;
};

static bool fused_tile_fits(const MegaMoEStageConfig& config, int ranks, int groups) {
    const int bm = config.block_m, bn = config.block_n;
    const int threads = (bm / config.warp_m) * (bn / config.warp_n) * 32;
    return (bm == 16 || bm == 32 || bm == 64 || bm == 128 || bm == 256) &&
        threads >= std::max({128, ranks, groups}) && bn >= 64 && bn % 64 == 0 &&
        groups * (ranks + 4) * 4 <= config.smem_bytes;
}

static std::optional<CopyTile> select_copy_tile(
    int ranks, int n, int k, int groups, int em, const MegaMoEStageConfig& base) {
    constexpr CopyTile wide{128, 512, 128, 64, 64, 3, 3, 16};
    constexpr CopyTile large{256, 256, 128, 64, 64, 3, 3, 16};
    if (n == 6144 && k == 3584) {
        if (ranks * groups == 384) {
            if (ranks == 8) {
                if (em >= 4 && em <= 5) return CopyTile{64, 128, 128, 32, 64, 3, 2, 16};
                if ((em >= 6 && em <= 7) || em == 15)
                    return CopyTile{128, 256, 64, 64, 64, 3, 3, 16};
                // Two full average BM256 tiles amortize the wider copy stripe.
                if (em >= 2 * large.bm + 1)
                    return CopyTile{large.bm, large.bn, large.bk, large.wm, large.wn, large.stages, 2, 24};
            }
            if (em >= 8 && em <= 16) return wide;
        }
        if ((ranks == 8 || ranks == 16) && em == 8) return wide;
        if (em >= 8 && em <= 13) return CopyTile{32, 512, 128, 32, 64, 3, 3, 16};
        if (em == 1) return CopyTile{16, 128, 128, 16, 32, 3, 3, 8};
        if (em == 16) return wide;
        const int bm = base.block_m, bn = base.block_n, bk = base.block_k;
        const bool family = (bm == 128 && bn == 256 && bk == 64) ||
            (bm == 256 && bn == 256 && bk == 128) ||
            (bm == 128 && bn == 512 && bk == 128);
        if (ranks >= 2 && ranks <= 32 && (ranks & (ranks - 1)) == 0 && em > 16 && family)
            return em <= wide.bm ? wide : large;
    }
    if (ranks == 8 && n == 4096 && k == 4096 && groups == 64) {
        if (em == 10) return CopyTile{32, 512, 128, 32, 64, 3, 2, 16};
        if (em == 20) return CopyTile{64, 512, 128, 32, 64, 3, 2, 16};
    }
    return std::nullopt;
}

static MegaMoEStage1Config get_stage1_config(
    int shape_m, int em, int n, int k, int groups, int sms, int ranks) {
    MegaMoEStageConfig config{};
    for (int min_m : {16, 64, 128}) {
        config = make_stage_config(deep_gemm_fp4_common::get_best_configs(
            shape_m, em, n, k, groups, sms, false, false,
            GemmType::MegaMoE, 256, min_m == 16 ? 32 : 128, min_m));
        if (fused_tile_fits(config, ranks, groups)) break;
    }
    DG_HOST_ASSERT(fused_tile_fits(config, ranks, groups));
    const auto record = select_copy_tile(ranks, n, k, groups, em, config);
    if (record) {
        const auto& t = *record;
        const MegaMoEStageConfig candidate{config.num_sms, t.bm, t.bn, t.bk,
            t.wm, t.wn, t.stages, std::get<0>(deep_gemm_fp4_common::get_smem_config_fp4(
                t.stages, t.bm, t.bn, t.wm, t.wn, t.bk, n, false, false))};
        if (fused_tile_fits(candidate, ranks, groups)) config = candidate;
    }
    const int bk = config.block_k;
    DG_HOST_ASSERT(bk > 0 && k > 0 && k % bk == 0 && k % 512 == 0);
    const int preferred = record ? ceil_div(record->bk * record->ktpf, bk) : 8;
    const int alignment = 512 / std::gcd(bk, 512);
    const int minimum = std::max(preferred, ceil_div(k / bk, 7));
    return {config, record ? record->copy_blocks : 3,
            ceil_div(minimum, alignment) * alignment};
}

static MegaMoEStageConfig get_stage2_config(
    int shape_m, int em, int n, int k, int groups, int sms) {
    auto config = make_stage_config(deep_gemm_fp4_common::get_best_configs(
        shape_m, em, n, k, groups, sms, false, false,
        GemmType::MegaMoE, 256, 128));
    // Remote bulk scatter uses a full warp of eight BF16 values per lane.
    const int old_bn = config.block_n, old_wn = config.warp_n;
    if (config.block_n == 128) config.block_n = 256;
    config.warp_n = std::max(config.warp_n, 32);
    if (config.block_n != old_bn || config.warp_n != old_wn)
        config.smem_bytes = std::get<0>(deep_gemm_fp4_common::get_smem_config_fp4(
            config.num_stages, config.block_m, config.block_n, config.warp_m, config.warp_n,
            config.block_k, n, false, false));
    DG_HOST_ASSERT(config.block_n == 256);
    // Scatter needs two accumulator tiles in addition to the mainloop resource contract.
    config.smem_bytes = std::max(config.smem_bytes, 2 * (config.block_m / config.warp_m) *
                                16 * config.block_n * static_cast<int>(sizeof(float)));
    return config;
}

static MegaMoELaunchConfig select_mega_moe(const MegaMoEHeuristicInput& input) {
    return {
        get_stage1_config(input.expected_m * input.num_experts_per_rank, input.expected_m,
            input.intermediate_hidden * 2, input.hidden / 2, input.num_experts_per_rank,
            input.num_sms, input.num_ranks),
        get_stage2_config(input.num_ranks * input.num_max_tokens_per_rank, input.expected_m,
            input.hidden, input.intermediate_hidden / 2, input.num_experts_per_rank,
            input.num_sms)
    };
}

} // namespace deep_gemm::mega
