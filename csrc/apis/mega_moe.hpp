#pragma once

#include <functional>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <deep_gemm/layout/mega_moe.cuh>

#include "../jit_kernels/impls/sm89_fp4_fp4_mega_moe.hpp"
#include "../jit_kernels/heuristics/mega_moe.hpp"

namespace deep_gemm::mega {

static layout::MegaMoEBuffer get_mega_moe_buffer_layout(
    int ranks, int experts, int capacity, int topk, int hidden, int intermediate) {
    DG_HOST_ASSERT(is_supported_megamoe_rank_count(ranks));
    DG_HOST_ASSERT(experts > 0 && experts % ranks == 0);
    DG_HOST_ASSERT(capacity > 0 && capacity % 32 == 0 && topk >= 1 && topk <= 16);
    DG_HOST_ASSERT(hidden >= 2048 && hidden % 1024 == 0 && intermediate > 0 && intermediate % 64 == 0);
    return layout::MegaMoEBuffer(ranks, experts, capacity, topk, hidden, intermediate);
}

static void check_mega_moe_buffer(const torch::Tensor& buffer, const layout::MegaMoEBuffer& layout) {
    TORCH_CHECK(buffer.scalar_type() == torch::kUInt8 && buffer.dim() == 1 &&
                buffer.is_contiguous() && buffer.storage_offset() == 0 &&
                buffer.numel() >= layout.get_num_bytes(), "invalid MegaMoE symmetric buffer size/layout");
}

static auto get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0) {
    TORCH_CHECK(mma_type == "fp4xfp4", "MegaMoE only supports mma_type=fp4xfp4");
    TORCH_CHECK(activation == "swiglu", "MegaMoE only supports activation=swiglu");
    TORCH_CHECK(num_shared_experts == 0, "MegaMoE shared experts are not supported yet");
    const auto layout = get_mega_moe_buffer_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden);
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        check_mega_moe_buffer(buffer, layout);
        auto topk_idx = make_buffer_tensor(buffer, torch::kInt32, layout.topk_idx.offset, {num_max_tokens_per_rank, num_topk});
        auto topk_weights = make_buffer_tensor(buffer, torch::kFloat32, layout.topk_weights.offset, {num_max_tokens_per_rank, num_topk});
        auto l1_acts = make_buffer_tensor(buffer, torch::kUInt8, layout.local_fp4.offset, {layout.compact * (hidden / 2)});
        auto l1_acts_sf = make_buffer_tensor(buffer, torch::kUInt16, layout.local_sfa.offset, {layout.compact * (hidden / 64)});
        auto l2_acts = make_buffer_tensor(buffer, torch::kUInt8, layout.stage1_fp4.offset, {layout.compact, intermediate_hidden / 2});
        auto l2_acts_sf = make_buffer_tensor(buffer, torch::kUInt16, layout.stage1_scale_storage.offset,
            {layout.compact, intermediate_hidden / 64}, {1, layout.compact});
        const DispatchBufferLayout dispatch_layout(num_experts, num_max_tokens_per_rank, hidden);
        auto x = make_buffer_tensor(buffer, torch::kUInt8, layout.dispatch.offset + dispatch_layout.metadata_bytes(),
            {2, num_experts, num_max_tokens_per_rank, hidden / 2},
            {static_cast<int64_t>(dispatch_layout.total_bytes()), int64_t(num_max_tokens_per_rank) * (hidden / 2), hidden / 2, 1});
        auto x_sf = make_buffer_tensor(buffer, torch::kUInt16, layout.staging.offset,
            {2, layout.groups, num_ranks, num_max_tokens_per_rank, hidden / 64});
        return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                               l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    using Tensors = decltype(slice_input_buffers(std::declval<const torch::Tensor&>()));
    return std::make_tuple(layout.get_num_bytes(), std::function<Tensors(const torch::Tensor&)>(slice_input_buffers));
}


static void quant(const torch::Tensor& x, const torch::Tensor& ids, const torch::Tensor& weights,
                  const torch::Tensor& sym_buffer, const std::vector<int64_t>& buffer_ptrs,
                  int ranks, int experts, int capacity, int rank,
                  int intermediate) {
    DG_HOST_ASSERT(sym_buffer.is_cuda());
    DG_HOST_ASSERT(x.device() == sym_buffer.device() && ids.device() == sym_buffer.device() &&
                   weights.device() == sym_buffer.device());
    const int hidden = x.size(1), topk = ids.size(1);
    const auto layout = get_mega_moe_buffer_layout(
        ranks, experts, capacity, topk, hidden, intermediate);
    DG_HOST_ASSERT(rank >= 0 && rank < ranks && x.size(0) <= capacity);
    DG_HOST_ASSERT(buffer_ptrs.size() == ranks);
    check_mega_moe_buffer(sym_buffer, layout);
    auto* base = sym_buffer.data_ptr<uint8_t>();
    QuantRuntime::Args a{hidden, topk, capacity, ranks, experts,
        {dim3(1), dim3(256), 0}, x.data_ptr(), ids.data_ptr<int32_t>(), weights.data_ptr<float>(),
        reinterpret_cast<int32_t*>(base + layout.topk_idx.offset),
        reinterpret_cast<float*>(base + layout.topk_weights.offset), base + layout.dispatch.offset,
        static_cast<uint32_t>(x.size(0)), layout::SymBuffer<>(buffer_ptrs, rank), base + layout.staging.offset,
        reinterpret_cast<uint32_t*>(base + layout.preprocess_ready.offset)};
    const auto runtime = compiler->build("megamoe_quant", QuantRuntime::generate(a));
    const int count_blocks = experts / ranks;
    const int resident = resident_blocks(runtime, 256, 0) * device_runtime->get_prop()->multiProcessorCount;
    const int token_blocks = std::max(1, std::min(static_cast<int>(a.tokens), std::max(1, resident - count_blocks)));
    a.launch_args.grid_dim.x = count_blocks + token_blocks;
    QuantRuntime::launch(runtime, a);
}

// Python owns one symmetric allocation; C++ slices it and submits computation.
static void fp4_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe, const std::string& activation,
    const std::optional<float>& activation_clamp_opt, const bool& fast_math,
    const std::optional<int>& expected_m, const std::function<void()>& stage1_done) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    DG_HOST_ASSERT(y.dim() == 2);
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(fast_math);

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(sym_buffer.is_cuda());
    for (const auto& tensor : {y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf})
        DG_HOST_ASSERT(tensor.device() == sym_buffer.device());
    DG_HOST_ASSERT(l1_weights.dim() == 3 && l2_weights.dim() == 3);
    DG_HOST_ASSERT(l1_weights.is_contiguous() && l2_weights.is_contiguous());
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kUInt8 && l2_weights.scalar_type() == torch::kUInt8);
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden_packed] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden, intermediate_hidden_packed] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(y.size(1) == hidden);
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(y.is_contiguous());
    const auto intermediate_hidden = intermediate_hidden_packed * 2;
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    // MXFP4 packs two K elements per byte; GEMM1 has two SwiGLU branches.
    DG_HOST_ASSERT(hidden == 2 * hidden_packed);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);

    // Check scale extents against the weights before any conversion or launch.
    // Two FP4 elements share one weight byte; two UE8M0 scales share one uint16.
    auto normalize_sf = [](torch::Tensor sf, const torch::Tensor& weight) {
        DG_HOST_ASSERT(sf.dim() == 3);
        DG_HOST_ASSERT(sf.scalar_type() == torch::kUInt8 || sf.scalar_type() == torch::kUInt16);
        DG_HOST_ASSERT(sf.size(0) == weight.size(0) && sf.size(1) == weight.size(1));
        const int64_t packed_k_per_scale = sf.scalar_type() == torch::kUInt8 ? 16 : 32;
        DG_HOST_ASSERT(sf.size(2) == ceil_div<int64_t>(weight.size(2), packed_k_per_scale));
        if (!deep_gemm_fp4_common::check_mxfp4_scales_layout(sf)) {
            sf = deep_gemm_fp4_common::post_preprocess_mxfp4_scales(sf);
            if (!deep_gemm_fp4_common::check_mxfp4_scales_layout(sf))
                sf = deep_gemm_fp4_common::preprocess_mxfp4_scales(sf);
        }
        DG_HOST_ASSERT(deep_gemm_fp4_common::check_mxfp4_scales_layout(sf));
        return sf;
    };
    const auto l1_weights_sf_t = normalize_sf(l1_weights_sf, l1_weights);
    const auto l2_weights_sf_t = normalize_sf(l2_weights_sf, l2_weights);

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->device() == sym_buffer.device());
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    DG_HOST_ASSERT(rank_idx >= 0 && rank_idx < num_ranks);
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "fp4xfp4", activation, 0
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);
    MegaMoEComputeViews views{
        .l1_acts = l1_acts, .l1_acts_sf = l1_acts_sf,
        .l2_acts = l2_acts, .l2_acts_sf = l2_acts_sf,
        .topk_idx = topk_idx, .topk_weights = topk_weights};
    prepare_mega_moe_compute_views(
        views, sym_buffer, num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden);
    const int64_t tokens = num_tokens, groups = num_experts_per_rank;
    const int64_t owner_capacity = static_cast<int64_t>(num_ranks) * num_max_tokens_per_rank;
    int em;
    if (!expected_m.has_value()) {
        em = std::max<int64_t>(1, std::min(owner_capacity,
            tokens * num_topk / groups + 1));
    } else {
        DG_HOST_ASSERT(expected_m.value() >= 1 && expected_m.value() <= owner_capacity);
        em = expected_m.value();
    }

    const auto [arch_major, arch_minor] = device_runtime->get_arch_pair();
    if (arch_major != 8 || arch_minor != 9)
        DG_HOST_UNREACHABLE("FP4 MegaMoE requires SM89 (M890)");

    fp4_fp4_mega_moe_impl(
        y, std::make_tuple(l1_weights, l1_weights_sf_t),
        std::make_tuple(l2_weights, l2_weights_sf_t),
        cumulative_local_expert_recv_stats, sym_buffer_ptrs,
        rank_idx, num_max_tokens_per_rank, num_tokens, activation_clamp,
        views, em, stage1_done);
}

static void register_apis(pybind11::module_& m) {
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
    m.def("fp4_fp4_mega_moe", &fp4_fp4_mega_moe);
    m.def("_mega_moe_quant", &quant);
}

}  // namespace deep_gemm::mega
