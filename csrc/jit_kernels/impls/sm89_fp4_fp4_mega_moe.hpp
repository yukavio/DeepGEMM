#pragma once

#include <functional>
#include <deep_gemm/layout/mega_moe.cuh>
#include "fp4_gemm.hpp"
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm::mega {

static torch::Tensor make_buffer_tensor(const torch::Tensor& buffer, torch::ScalarType dtype,
                            int64_t offset, torch::IntArrayRef shape,
                            torch::IntArrayRef strides = {}) {
    // Non-owning views, as upstream: keep the symmetric buffer alive while in use.
    auto* pointer = buffer.data_ptr<uint8_t>() + offset;
    const auto options = buffer.options().dtype(dtype);
    return strides.empty() ? torch::from_blob(pointer, shape, options)
                           : torch::from_blob(pointer, shape, strides, options);
}

static int resident_blocks(const std::shared_ptr<KernelRuntime>& runtime, int threads, int smem) {
    construct_launch_config(runtime->kernel, current_stream(), smem, dim3(1), dim3(threads));
    int blocks = 0;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
    DG_HGGC_RUNTIME_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, runtime->kernel, threads, smem));
#else
    DG_HGGC_DRIVER_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, runtime->kernel, threads, smem));
#endif
    DG_HOST_ASSERT(blocks > 0);
    return blocks;
}

class QuantRuntime final : public LaunchRuntime<QuantRuntime> {
public:
    struct Args {
        int hidden, topk, capacity, ranks, experts;
        LaunchArgs launch_args;
        const void* input;
        const void* ids;
        const float* weights;
        int32_t* routing_ids;
        float* routing_weights;
        void* buffer;
        uint32_t tokens;
        layout::SymBuffer<> sym_buffer;
        void* staging;
        uint32_t* ready;
        bool ids64;
        const bool* active_mask;
    };

    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/mxfp4_quant.cuh>
extern "C" __global__ void megamoe_quant(
    const __ppu_bfloat16* input, const {5}* ids, const float* weights,
    int32_t* routing_ids, float* routing_weights, void* buffer, uint32_t tokens,
    deep_gemm::layout::SymBuffer<> sym_buffer, uint8_t* staging, uint32_t* ready, const bool* active_mask) {{
    deep_gemm::mxfp4_quantize_impl<{0}, {1}, {2}, {3}, {4}, {5}, {6}>(
        input, ids, weights, routing_ids, routing_weights, buffer, tokens,
        sym_buffer, staging, ready, active_mask);
}}
)", a.hidden, a.topk, a.capacity, a.ranks, a.experts, a.ids64 ? "int64_t" : "int32_t", a.active_mask != nullptr);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, a.input, a.ids, a.weights,
            a.routing_ids, a.routing_weights, a.buffer, a.tokens, a.sym_buffer,
            a.staging, a.ready, a.active_mask));
    }
};

class ReduceRuntime final : public LaunchRuntime<ReduceRuntime> {
public:
    struct Args {
        int topk, hidden;
        LaunchArgs launch_args;
        const void* input;
        const int32_t* ids;
        const float* weights;
        void* output;
        int capacity, local_experts, rank;
        bool bulk_scatter;
    };

    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/moe_reduce.cuh>
extern "C" __global__ void megamoe_reduce(const __ppu_bfloat16* input, const int32_t* ids,
    const float* weights, __ppu_bfloat16* output, int capacity, int local_experts, int rank) {{
    moe_reduce_impl<{}, {}, kMoeReduceThreads, {}>(input, ids, weights, output, capacity, local_experts, rank);
}}
)", a.topk, a.hidden, a.bulk_scatter);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, a.input, a.ids, a.weights,
            a.output, a.capacity, a.local_experts, a.rank));
    }
};

struct ScatterEpilogueParams {
    decltype(FP4GemmRuntime::EpilogueArgsNoTsm{}.thread) thread{};
    const void* src_info = nullptr;
    const int32_t* masked_m = nullptr;
    uint32_t token_capacity = 0;
};

struct FusedDispatchMainloopArguments : FP4GemmRuntime::MainLoopArguments {
    volatile uint32_t* copy_ready_flags = nullptr;
    uint32_t copy_ready_generation = 0;
    uint32_t copy_k_tiles_per_flag = 0;
    volatile uint32_t* sfa_ready_flags = nullptr;
    uint32_t sfa_ready_generation = 0;
};

template <int Ranks>
struct ScatterParams : FP4GemmRuntime::GemmKernelParamsT<ScatterEpilogueParams> {
    layout::SymBuffer<Ranks> buffer{};
    int32_t* barrier = nullptr;
    const void* reduce_input = nullptr;
    const int32_t* reduce_ids = nullptr;
    const float* reduce_weights = nullptr;
    void* reduce_output = nullptr;
    int reduce_tokens = 0, reduce_capacity = 0, reduce_rank = 0;
};

template <MegaMoEStage Stage, int Ranks>
class GemmRuntime final : public LaunchRuntime<GemmRuntime<Stage, Ranks>> {
public:
    static_assert(Stage == MegaMoEStage::Linear1 || Stage == MegaMoEStage::Linear2,
                  "unsupported MegaMoE stage");
    static constexpr bool kIsLinear1 = Stage == MegaMoEStage::Linear1;
    static constexpr const char* kName = kIsLinear1 ? "megamoe_fused_dispatch_gemm1_swiglu" : "megamoe_scatter";
    using Params = std::conditional_t<kIsLinear1,
        FP4GemmRuntime::GemmKernelParamsT<FP4GemmRuntime::EpilogueArgsSiluAndMulPostQuant,
                                        FusedDispatchMainloopArguments, MegaMoESchedulerArguments>,
        ScatterParams<Ranks>>;
    using MainloopArguments = std::conditional_t<kIsLinear1,
        FusedDispatchMainloopArguments, FP4GemmRuntime::MainLoopArguments>;
    using EpilogueArguments = std::conditional_t<kIsLinear1,
        FP4GemmRuntime::EpilogueArgsSiluAndMulPostQuant, ScatterEpilogueParams>;
    using SchedulerArguments = std::conditional_t<kIsLinear1,
        MegaMoESchedulerArguments, TileSchedulerArguments>;

    struct LaunchInfo {
        int n, k, block_m, block_n, block_k, warp_m, warp_n, num_groups, num_stages;
        int copy_blocks = 0, k_tiles_per_flag = 0, source_capacity = 1, topk = 1;
        bool apply_clamp = false, wide_offsets = false, preserve_bf16 = false, fuse_reduce = false;
    };

    struct GemmArguments {
        cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
        FP4GemmRuntime::GemmProblemSize problem_shape;
        MainloopArguments mainloopargs{};
        EpilogueArguments epilogueargs{};
        cutlass::KernelHardwareInfo hw_info{};
        SchedulerArguments scheduler{};
    };

    struct ScatterGemmArguments : GemmArguments {
        layout::SymBuffer<Ranks> scatter_buffer{};
        int32_t* scatter_barrier = nullptr;
        const void* reduce_input = nullptr;
        const int32_t* reduce_ids = nullptr;
        const float* reduce_weights = nullptr;
        void* reduce_output = nullptr;
        int reduce_tokens = 0, reduce_capacity = 0, reduce_rank = 0;
    };
    using Arguments = std::conditional_t<kIsLinear1, GemmArguments, ScatterGemmArguments>;

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        Params kernel_params{};
    };

    static Params to_underlying_arguments_rtc(const Arguments& args) {
        Params params{};
        params.mode = args.mode;
        params.problem_shape = args.problem_shape;
        params.collective_mainloop_params = args.mainloopargs;
        params.collective_epilogue_params = args.epilogueargs;
        params.hw_info = args.hw_info;
        params.scheduler = args.scheduler;
        if constexpr (!kIsLinear1) {
            params.buffer = args.scatter_buffer;
            params.barrier = args.scatter_barrier;
            params.reduce_input = args.reduce_input;
            params.reduce_ids = args.reduce_ids;
            params.reduce_weights = args.reduce_weights;
            params.reduce_output = args.reduce_output;
            params.reduce_tokens = args.reduce_tokens;
            params.reduce_capacity = args.reduce_capacity;
            params.reduce_rank = args.reduce_rank;
        }
        return params;
    }

    static std::string generate_impl(const Args& a) {
        const auto& info = a.launch_info;
        return fmt::format(R"(
#include <deep_gemm/impls/mega_moe.cuh>
using Builder = deep_gemm::MegaMoEFp4Kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;
using Kernel = Builder::Kernel;
static_assert(Kernel::SharedStorageSize == {}, "MegaMoE shared storage size differs from host");
static_assert(sizeof(Kernel::Params) == {}, "MegaMoE parameter size differs from host");
static_assert(offsetof(Kernel::Params, mainloop) == {}, "MegaMoE mainloop offset differs from host");
static_assert(offsetof(Kernel::Params, epilogue) == {}, "MegaMoE epilogue offset differs from host");
static_assert(offsetof(Kernel::Params, scheduler) == {}, "MegaMoE scheduler offset differs from host");
extern "C" __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
__global__ void {}(Kernel::Params params) {{
    extern __shared__ char smem[];
    Kernel op;
    op(params, smem);
}}
)", info.n, info.k, info.block_m, info.block_n, info.block_k, info.warp_m, info.warp_n, info.num_groups, info.num_stages,
            Ranks, kIsLinear1 ? "MegaMoEStage::Linear1" : "MegaMoEStage::Linear2",
            info.copy_blocks, info.k_tiles_per_flag, info.source_capacity, info.topk,
            info.apply_clamp, info.wide_offsets, info.preserve_bf16, info.fuse_reduce, a.launch_args.smem_size, sizeof(Params),
            offsetof(Params, collective_mainloop_params), offsetof(Params, collective_epilogue_params),
            offsetof(Params, scheduler), kName);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, a.kernel_params));
    }
};

template <typename F>
static void for_ranks(int ranks, F&& f) {
    switch (ranks) {
        case 1: return f(std::integral_constant<int, 1>{});
        case 2: return f(std::integral_constant<int, 2>{});
        case 4: return f(std::integral_constant<int, 4>{});
        case 8: return f(std::integral_constant<int, 8>{});
        case 16: return f(std::integral_constant<int, 16>{});
        case 32: return f(std::integral_constant<int, 32>{});
        default: DG_HOST_UNREACHABLE("unsupported MegaMoE rank count");
    }
}

static FP4GemmRuntime::MainLoopArguments make_mainloop_arguments(int m, int n, int k, int sfa_m,
                        const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                        const torch::Tensor& rhs, const torch::Tensor& rhs_scales) {
    using S = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using SF = cute::Stride<cute::Int<1>, int64_t, int64_t>;
    return {
        cute::make_shape(m, n, k), lhs.data_ptr<uint8_t>(),
        cutlass::make_cute_packed_stride(S{}, cute::make_shape(m, k, 1)),
        rhs.data_ptr<uint8_t>(), cutlass::make_cute_packed_stride(S{}, cute::make_shape(n, k, 1)),
        lhs_scales.data_ptr<uint16_t>(),
        cutlass::make_cute_packed_stride(SF{}, cute::make_shape(sfa_m, ceil_div(k, 32), 1)),
        rhs_scales.data_ptr<uint16_t>(),
        cutlass::make_cute_packed_stride(SF{}, cute::make_shape(n, ceil_div(k, 32), 1))};
}

template <typename Runtime>
static void launch_gemm(typename Runtime::Args& args, int num_sms) {
    constexpr auto name = Runtime::kName;
    const auto runtime = compiler->build(name, Runtime::generate(args));
    const int blocks = resident_blocks(runtime, args.launch_args.block_dim.x, args.launch_args.smem_size);
    args.kernel_params.hw_info.cu_count = num_sms * blocks;
    args.launch_args.grid_dim.x = args.kernel_params.hw_info.cu_count;
    const auto& info = args.launch_info;
    DgProfParam profile;
    auto& profiling = ProfilingInterface::Instance();
    if (profiling.get_op_info()) {
        profile.set_params(GemmType::MegaMoE, false, "fp4", info.num_groups,
            cute::get<0>(args.kernel_params.problem_shape), info.n, info.k, 0, nullptr, current_stream());
        profile.add_params("stage", name);
    }
    profiling.instrument(true, profile);
    Runtime::launch(runtime, args);
    profiling.instrument(false, profile);
    if (get_env<int>("show_log", 0)) {
        int registers = 0, stack_bytes = 0;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
        hggcFuncAttributes attributes{};
        DG_HGGC_RUNTIME_CHECK(hggcFuncGetAttributes(&attributes, runtime->kernel));
        registers = attributes.numRegs;
        stack_bytes = attributes.localSizeBytes;
#else
        DG_HGGC_DRIVER_CHECK(hgFuncGetAttribute(&registers, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel));
        DG_HGGC_DRIVER_CHECK(hgFuncGetAttribute(&stack_bytes, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel));
#endif
        printf("[%s] groups:%d problem:[%d,%d,%d] num_sms:%d blocks_per_sm:%d grid:%u\n"
               "tile:[%d,%d,%d] warp:[%d,%d] stages:%d smem:%d vreg:%d stack:%d\n",
            name, info.num_groups, cute::get<0>(args.kernel_params.problem_shape), info.n, info.k,
            num_sms, blocks, args.launch_args.grid_dim.x, info.block_m, info.block_n, info.block_k,
            info.warp_m, info.warp_n, info.num_stages, args.launch_args.smem_size, registers, stack_bytes);
    }
}

template <int Ranks>
static auto make_linear1_args(const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                              const torch::Tensor& out, const torch::Tensor& out_scale,
                              const torch::Tensor& local_fp4, const torch::Tensor& local_sfa,
                              const MegaMoESchedulerArguments& scheduler,
                              float swiglu_limit, const MegaMoEStage1Config& config, bool preserve_bf16) {
    using Runtime = GemmRuntime<MegaMoEStage::Linear1, Ranks>;
    DG_HOST_ASSERT(config.num_sms > config.copy_blocks &&
                   config.num_sms <= device_runtime->get_prop()->multiProcessorCount);
    const int n = rhs.size(1), k = rhs.size(2), groups = rhs.size(0);
    const int m = scheduler.shape_m, compact_m = out.size(0);
    typename Runtime::MainloopArguments mainloop{};
    static_cast<FP4GemmRuntime::MainLoopArguments&>(mainloop) =
        make_mainloop_arguments(m, n, k, compact_m, local_fp4, local_sfa, rhs, rhs_scales);
    using S = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    const typename Runtime::EpilogueArguments epilogue{
        .stride_C = cutlass::make_cute_packed_stride(S{}, cute::make_shape(m, 0, 1)),
        .ptr_D = out.data_ptr<uint8_t>(),
        .stride_D = cutlass::make_cute_packed_stride(S{}, cute::make_shape(m, n, 1)),
        .ptr_SFD = out_scale.data_ptr<uint16_t>(),
        .shape_m = static_cast<uint32_t>(compact_m), .swiglu_limit = swiglu_limit};
    const typename Runtime::GemmArguments gemm_args{
        .problem_shape = {m, n, k, 1}, .mainloopargs = mainloop,
        .epilogueargs = epilogue, .hw_info = {rhs.get_device(), config.num_sms}, .scheduler = scheduler};
    return typename Runtime::Args{
        .launch_info = {.n = n, .k = k, .block_m = config.block_m, .block_n = config.block_n, .block_k = config.block_k,
                        .warp_m = config.warp_m, .warp_n = config.warp_n, .num_groups = groups, .num_stages = config.num_stages,
                        .copy_blocks = config.copy_blocks, .k_tiles_per_flag = config.k_tiles_per_flag,
                        .source_capacity = m / Ranks, .topk = compact_m / m,
                        .apply_clamp = std::isfinite(swiglu_limit), .preserve_bf16 = preserve_bf16},
        .launch_args = {dim3(1), dim3((config.block_m / config.warp_m) * (config.block_n / config.warp_n) * 32), config.smem_bytes},
        .kernel_params = Runtime::to_underlying_arguments_rtc(gemm_args)};
}

template <int Ranks>
static auto make_linear2_args(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                             const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                             int32_t* masked_m, const void* src_info,
                             const layout::SymBuffer<Ranks>& scatter_buffer, int32_t* barrier,
                             int token_capacity, int topk,
                             const MegaMoEStageConfig& config, bool fuse_reduce,
                             const torch::Tensor& reduce_input, const torch::Tensor& ids,
                             const torch::Tensor& weights, const torch::Tensor& y, int num_tokens, int rank) {
    using Runtime = GemmRuntime<MegaMoEStage::Linear2, Ranks>;
    const int m = lhs.size(0), n = rhs.size(1), k = rhs.size(2), groups = rhs.size(0);
    typename Runtime::Arguments gemm_args{};
    static_cast<typename Runtime::GemmArguments&>(gemm_args) = {
        .problem_shape = {m, n, k, 1},
        .mainloopargs = make_mainloop_arguments(m, n, k, m, lhs, lhs_scales, rhs, rhs_scales),
        .epilogueargs = {.src_info = src_info, .masked_m = masked_m,
                         .token_capacity = static_cast<uint32_t>(token_capacity)},
        .hw_info = {rhs.get_device(), config.num_sms},
        .scheduler = TileSchedulerArguments(m, masked_m)};
    gemm_args.scatter_buffer = scatter_buffer;
    gemm_args.scatter_barrier = barrier;
    gemm_args.reduce_input = reduce_input.data_ptr();
    gemm_args.reduce_ids = ids.data_ptr<int32_t>();
    gemm_args.reduce_weights = weights.data_ptr<float>();
    gemm_args.reduce_output = y.data_ptr();
    gemm_args.reduce_tokens = num_tokens;
    gemm_args.reduce_capacity = token_capacity;
    gemm_args.reduce_rank = rank;
    return typename Runtime::Args{
        .launch_info = {.n = n, .k = k, .block_m = config.block_m, .block_n = config.block_n, .block_k = config.block_k,
                        .warp_m = config.warp_m, .warp_n = config.warp_n, .num_groups = groups, .num_stages = config.num_stages,
                        .source_capacity = token_capacity, .topk = topk,
                        .wide_offsets = static_cast<int64_t>(topk) * token_capacity * n > UINT32_MAX,
                        .fuse_reduce = fuse_reduce},
        .launch_args = {dim3(1), dim3((config.block_m / config.warp_m) * (config.block_n / config.warp_n) * 32), config.smem_bytes},
        .kernel_params = Runtime::to_underlying_arguments_rtc(gemm_args)};
}

static void reduce(const torch::Tensor& input, const torch::Tensor& ids, const torch::Tensor& weights,
                   const torch::Tensor& output, int num_tokens, int capacity,
                   int local_experts, int rank, bool bulk_scatter) {
    if (num_tokens == 0) return;
    ReduceRuntime::Args a{static_cast<int>(ids.size(1)), static_cast<int>(output.size(1)),
        {dim3(num_tokens), dim3(512), 0}, input.data_ptr(), ids.data_ptr<int32_t>(), weights.data_ptr<float>(),
        output.data_ptr(), capacity, local_experts, rank, bulk_scatter};
    const auto runtime = compiler->build("megamoe_reduce", ReduceRuntime::generate(a));
    ReduceRuntime::launch(runtime, a);
}

// Internal compute entry; the public API owns input validation and preprocessing.
struct MegaMoEComputeViews {
    torch::Tensor l1_acts, l1_acts_sf, l2_acts, l2_acts_sf;
    torch::Tensor topk_idx, topk_weights, scatter_buffer;
    MegaMoESchedulerArguments scheduler;
    int32_t* barrier = nullptr;
    int64_t scatter_offset = 0;
};

// Reuse the public slicer's activations/routing; prepare only compute-specific resources.
static void prepare_mega_moe_compute_views(
    MegaMoEComputeViews& views, const torch::Tensor& buffer,
    int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk,
    int hidden, int intermediate_hidden) {
    const layout::MegaMoEBuffer layout(num_ranks, num_experts, num_max_tokens_per_rank,
                                      num_topk, hidden, intermediate_hidden);
    views.scatter_buffer = make_buffer_tensor(buffer, torch::kBFloat16, layout.scatter_storage.offset,
        {int64_t(num_topk) * num_max_tokens_per_rank, hidden});
    auto* base = buffer.data_ptr<uint8_t>();
    auto* grouped_layout = reinterpret_cast<int32_t*>(base + layout.grouped_layout.offset);
    auto* masked_m = grouped_layout + (1 + layout.blocks) * 4;
    views.scheduler = MegaMoESchedulerArguments(num_ranks * num_max_tokens_per_rank, masked_m, grouped_layout,
        reinterpret_cast<uint32_t*>(base + layout.expert_rank_prefix.offset),
        views.l1_acts.data_ptr<uint8_t>(), views.l1_acts_sf.data_ptr<uint16_t>(),
        reinterpret_cast<uint32_t*>(base + layout.copy_ready.offset), 0,
        layout::SymBuffer<>{}, base + layout.dispatch.offset,
        reinterpret_cast<uint32_t*>(base + layout.preprocess_ready.offset),
        reinterpret_cast<uint32_t*>(base + layout.expert_base.offset), base + layout.src_info.offset,
        base + layout.staging.offset, 0);
    views.barrier = reinterpret_cast<int32_t*>(base + layout.barrier_storage.offset);
    views.scatter_offset = layout.scatter_storage.offset;
}

static void fp4_fp4_mega_moe_impl(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    int rank_idx, int num_max_tokens_per_rank, int num_tokens, float activation_clamp,
    const MegaMoEComputeViews& views, int expected_m, bool preserve_bf16, bool fuse_reduce,
    const std::function<void()>& stage1_done) {
    // Reserved upstream-compatible argument; receive-count accumulation is not implemented yet.
    (void)cumulative_local_expert_recv_stats;
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto& [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto& [l2_weights, l2_weights_sf] = l2_weights_tuple;
    const int num_experts_per_rank = l1_weights.size(0), hidden = l2_weights.size(1);
    const int num_topk = views.topk_idx.size(1);
    // Inputs and weight scales have been validated/preprocessed by the API.
    const MegaMoEHeuristicInput heuristic_input{
        .num_sms = device_runtime->get_num_sms(), .num_ranks = num_ranks,
        .num_experts_per_rank = num_experts_per_rank, .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = static_cast<int>(l2_weights.size(2)) * 2,
        .expected_m = expected_m};
    const auto launch_config = select_mega_moe(heuristic_input);
    const auto& l1_config = launch_config.l1;
    const auto& l2_config = launch_config.l2;
    auto scheduler = views.scheduler;
    scheduler.sym_buffer = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx);
    scheduler.num_copy_blocks = l1_config.copy_blocks;
    scheduler.local_source_rank = rank_idx;
    for_ranks(num_ranks, [&](auto r) {
        using Linear1Runtime = GemmRuntime<MegaMoEStage::Linear1, decltype(r)::value>;
        using Linear2Runtime = GemmRuntime<MegaMoEStage::Linear2, decltype(r)::value>;
        auto scatter_buffer = layout::SymBuffer<decltype(r)::value>(sym_buffer_ptrs, rank_idx);
        scatter_buffer.base += views.scatter_offset;
        auto l1_args = make_linear1_args<decltype(r)::value>(
            l1_weights, l1_weights_sf, views.l2_acts, views.l2_acts_sf, views.l1_acts, views.l1_acts_sf,
            scheduler, activation_clamp, l1_config, preserve_bf16);
        auto l2_args = make_linear2_args<decltype(r)::value>(
            views.l2_acts, views.l2_acts_sf, l2_weights, l2_weights_sf, scheduler.grouped_layout, scheduler.src_info,
            scatter_buffer, views.barrier, num_max_tokens_per_rank, num_topk, l2_config, fuse_reduce,
            views.scatter_buffer, views.topk_idx, views.topk_weights, y, num_tokens, rank_idx);
        launch_gemm<Linear1Runtime>(l1_args, l1_config.num_sms);
        if (stage1_done) stage1_done();  // Optional test-only timing marker.
        launch_gemm<Linear2Runtime>(l2_args, l2_config.num_sms);
    });
    if (!fuse_reduce)
        reduce(views.scatter_buffer, views.topk_idx, views.topk_weights, y, num_tokens, num_max_tokens_per_rank,
               num_experts_per_rank, rank_idx,
               !(num_ranks == 8 && hidden == 4096 && num_experts_per_rank == 32 && l2_weights.size(2) == 1024));
}

}  // namespace deep_gemm::mega
