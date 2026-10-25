#pragma once

#include <type_traits>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <cute/tensor.hpp>
#include <deep_gemm/dispatch_layout.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <hggc_bf16.h>

namespace deep_gemm::epilogue {

// The selected epilogue is the single source of scatter mode and rank count.
template <class Epilogue, class = void>
struct Fp4ScatterTraits {
    static constexpr bool enabled = false;
    static constexpr int world_size = 1;
    static constexpr bool fuse_reduce = false;
    static constexpr bool bulk_scatter = true;
};

template <class Epilogue>
struct Fp4ScatterTraits<Epilogue, std::void_t<decltype(Epilogue::kWorldSize_)>> {
    static constexpr bool enabled = true;
    static constexpr int world_size = Epilogue::kWorldSize_;
    static constexpr bool fuse_reduce = Epilogue::kFuseReduce_;
    static constexpr bool bulk_scatter = !Epilogue::kLinearScatter_;
};

template <uint32_t kWorldSize>
using Fp4ScatterBuffer = layout::SymBuffer<kWorldSize>;

// FP4 GEMM2 epilogue used by the two-stage MegaMoE path. It converts the
// accumulator tile to BF16 and scatters each valid row directly into the
// source token rank's symmetric scratch buffer.
template <
    class StrideC_, class StrideD_, class SmemLayout_,
    class CopyAtomR2S_, class ElementAccumulator_, class ElementD_,
    class ThreadEpilogueOp_,
    int kWorldSize = 2,
    bool kWideScatterOffset = false, bool kFuseReduce = false,
    bool kLinearScatter = false>
class Fp4ScatterEpilogue {
public:
    using StrideC = StrideC_;
    using StrideD = StrideD_;
    using SmemLayout = SmemLayout_;
    using CopyAtomR2S = CopyAtomR2S_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementC = ElementD_;
    using ElementD = ElementD_;
    using ThreadEpilogueOp = ThreadEpilogueOp_;
    static constexpr int kWorldSize_ = kWorldSize;
    static constexpr bool kFuseReduce_ = kFuseReduce;
    static constexpr bool kLinearScatter_ = kLinearScatter;

    struct SharedStorage {
        cute::array_aligned<ElementAccumulator,
                            2 * cute::cosize_v<SmemLayout>> accumulator;
    };

    struct Arguments {
        typename ThreadEpilogueOp::Params thread{};
        TokenSrcMetadata const* src_info = nullptr;
        int32_t const* masked_m = nullptr;
        uint32_t token_capacity = 0;
    };

    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(
            ProblemShape const&, Arguments const& args, void*) {
        return args;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    CUTLASS_HOST_DEVICE explicit Fp4ScatterEpilogue(Params const& params)
        : params_(params), epilogue_op_(params.thread) {}

#if defined(__HGGCCC__)
    CUTLASS_HOST_DEVICE Fp4ScatterEpilogue(Params const& params, SharedStorage&)
        : params_(params), epilogue_op_(params.thread) {}
#endif

    CUTLASS_DEVICE bool is_source_needed() {
        return epilogue_op_.is_source_needed();
    }

    template <
        class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL,
        class FrgEngine, class FrgLayout, class TiledMma, class ResidueMNK>
    CUTLASS_DEVICE void operator()(
            ProblemShapeMNKL problem_shape_mnkl,
            BlockShapeMNK block_shape_mnk,
            BlockCoordMNKL block_coord_mnkl,
            cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
            TiledMma tiled_mma, ResidueMNK, int thread_idx, char* smem,
            uint32_t expert, uint32_t offset_m,
            Fp4ScatterBuffer<kWorldSize> const& scatter_buffer) {
        using namespace cute;

        const auto N = get<1>(problem_shape_mnkl);
        auto [m_coord, n_coord, k_coord, l_coord] = block_coord_mnkl;
        (void)k_coord;
        (void)l_coord;

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
        constexpr int kSmemStageElements = cute::cosize_v<SmemLayout>;
        Tensor shared_accumulator = make_tensor(
            make_smem_ptr(storage.accumulator.data()), SmemLayout{});

        auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
        auto thread_r2s = tiled_r2s.get_thread_slice(thread_idx);
        Tensor register_accumulator = thread_r2s.retile_S(accumulators);

        auto subtile_shape = make_shape(
            size<0>(shared_accumulator), size<1>(shared_accumulator));
        Tensor identity = make_identity_tensor(
            make_shape(get<0>(block_shape_mnk), get<1>(block_shape_mnk)));
        Tensor divided_identity = flat_divide(identity, subtile_shape);

        constexpr int kSmemM = size<0>(shared_accumulator);
        constexpr int kSmemN = size<1>(shared_accumulator);
        // Flash uses linear stores for every tile width, because ranks may
        // select different tiles for heterogeneous local token counts.
        // Bulk scatter requires a complete warp row and a different layout.
        static_assert(kSmemN == 128 || kSmemN == 256, "unsupported scatter tile width");
        static_assert(kLinearScatter || kSmemN == 256, "bulk scatter needs full warp rows");
        constexpr int kVector = 8;
        constexpr int kThreadsPerRow = kSmemN / kVector;
        constexpr int kNumThreads = CUTE_STATIC_V(size(TiledMma{}));
        constexpr int kRowsPerIteration = kNumThreads / kThreadsPerRow;
        static_assert(kNumThreads % kThreadsPerRow == 0,
                      "thread count must cover complete rows");
        static_assert(kSmemM % kRowsPerIteration == 0,
                      "scatter epilogue must cover every row");
        static_assert(kVector == 8,
                      "remote bulk scatter requires 8 BF16 values per lane");

        const int row_lane = thread_idx / kThreadsPerRow;
        const int column = (thread_idx % kThreadsPerRow) * kVector;
        constexpr int kRowsPerSubtile = kSmemM / kRowsPerIteration;
        static_assert(kRowsPerSubtile % 2 == 0,
                      "two-row scatter loop requires an even row count");
        const uint32_t tile_row = m_coord * get<0>(block_shape_mnk);
        const uint32_t block_row = offset_m + tile_row;
        const uint32_t valid_rows = static_cast<uint32_t>(
            __ldg(params_.masked_m + expert));
        auto* local_base = reinterpret_cast<__ppu_bfloat16*>(
            scatter_buffer.base);

        CUTLASS_PRAGMA_UNROLL
        for (int step_m = 0; step_m < size<2>(divided_identity); ++step_m) {
            const uint32_t subtile_m_begin = tile_row + step_m * kSmemM;
            if (subtile_m_begin >= valid_rows) {
                break;
            }

            CUTLASS_PRAGMA_UNROLL
            for (int step_n = 0; step_n < size<3>(divided_identity); ++step_n) {
                const int smem_stage =
                    (step_m * int(size<3>(divided_identity)) + step_n) & 1;
                Tensor shared_accumulator_stage = make_tensor(
                    make_smem_ptr(storage.accumulator.data() +
                                  smem_stage * kSmemStageElements),
                    SmemLayout{});
                Tensor shared_r2s =
                    thread_r2s.partition_D(shared_accumulator_stage);
                const uint32_t n_base =
                    n_coord * get<1>(block_shape_mnk) + step_n * kSmemN;
                auto* local_tile_base = local_base + n_base;

                CUTLASS_PRAGMA_UNROLL
                for (int pipe_m = 0; pipe_m < size<1>(shared_r2s); ++pipe_m) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int pipe_n = 0; pipe_n < size<2>(shared_r2s); ++pipe_n) {
                        const int mma_m =
                            step_m * size<1>(shared_r2s) + pipe_m;
                        const int mma_n =
                            step_n * size<2>(shared_r2s) + pipe_n;
                        copy(tiled_r2s,
                             register_accumulator(_, mma_m, mma_n),
                             shared_r2s(_, pipe_m, pipe_n));
                    }
                }
                __syncthreads();

                TokenSrcMetadata metadata[kRowsPerSubtile];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kRowsPerSubtile; ++i) {
                    const uint32_t row = row_lane + i * kRowsPerIteration;
                    // Compact metadata has no padding after the last expert.
                    // Mask the load itself, not only the later scatter store.
                    metadata[i] = {};
                    if (subtile_m_begin + row < valid_rows) {
                        metadata[i] = params_.src_info[
                            block_row + step_m * kSmemM + row];
                    }
                }

                CUTLASS_PRAGMA_UNROLL
                for (int row_base = 0; row_base < kSmemM;
                     row_base += 2 * kRowsPerIteration) {
                    const int rows[2] = {
                        row_lane + row_base,
                        row_lane + row_base + kRowsPerIteration};
                    const TokenSrcMetadata infos[2] = {
                        metadata[row_base / kRowsPerIteration],
                        metadata[(row_base + kRowsPerIteration) /
                                 kRowsPerIteration]};

                    CUTLASS_PRAGMA_UNROLL
                    for (int pair = 0; pair < 2; ++pair) {
                        const int row = rows[pair];
                        const TokenSrcMetadata token_info = infos[pair];
                        const uint32_t expert_row = subtile_m_begin + row;
                        const uint32_t destination_rank = token_info.rank_idx;
                        if (row < kSmemM && expert_row < valid_rows) {
                            __ppu_bfloat16* destination = scatter_buffer.map(
                                local_tile_base, destination_rank);
                            if constexpr (kWideScatterOffset) {
                                destination +=
                                    (static_cast<uint64_t>(token_info.topk_idx) *
                                         params_.token_capacity +
                                     token_info.token_idx) * N + column;
                            } else {
                                destination +=
                                    (token_info.topk_idx *
                                         params_.token_capacity +
                                     token_info.token_idx) * N + column;
                            }

                            float4* source = reinterpret_cast<float4*>(
                                &shared_accumulator_stage(row, column));
                            const float4 low = source[0];
                            const float4 high = source[1];
                            union {
                                __ppu_bfloat162 bf16[4];
                                int4 packed;
                            } value;
                            value.bf16[0] = __floats2bfloat162_rn(low.x, low.y);
                            value.bf16[1] = __floats2bfloat162_rn(low.z, low.w);
                            value.bf16[2] = __floats2bfloat162_rn(high.x, high.y);
                            value.bf16[3] = __floats2bfloat162_rn(high.z, high.w);
                            if constexpr (kLinearScatter) {
                                __stbl(reinterpret_cast<int4*>(destination), value.packed);
                            } else if (destination_rank == scatter_buffer.rank_idx) {
                                asm volatile(
                                    "ppu.st.global.cg.v4.b32 [%0], {%1, %2, %3, %4};"
                                    :: "l"(destination), "r"(value.packed.x),
                                       "r"(value.packed.y), "r"(value.packed.z),
                                       "r"(value.packed.w) : "memory");
                            } else {
                                __ppu_remote_store_bulk_volatile_b32x4(
                                    reinterpret_cast<volatile void*>(destination),
                                    *reinterpret_cast<uint4*>(&value.packed));
                            }
                        }
                    }
                }

            }
        }
        // Alternating buffers remove subtile read/write hazards, but the next
        // persistent GEMM tile still reuses this storage for its mainloop.
        __syncthreads();
    }

private:
    Params params_;
    ThreadEpilogueOp epilogue_op_;
};

}  // namespace deep_gemm::epilogue
