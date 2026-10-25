#pragma once

#include <deep_gemm/impls/fp4_gemm_cutlass3.cuh>

namespace deep_gemm {

template <int N, int K, int BM, int BN, int BK, int WM, int WN,
          int Groups, int Stages, int Ranks, MegaMoEStage Stage,
          int CopyBlocks = 0, int KTilesPerFlag = 0,
          int SourceCapacity = 1, int TopK = 1,
          bool ApplyClamp = false, bool WideScatterOffset = false,
          bool PreserveBf16 = false, bool FuseReduce = false>
struct MegaMoEFp4Kernel {
    static_assert(Stage == MegaMoEStage::Linear1 || Stage == MegaMoEStage::Linear2,
                  "unsupported MegaMoE stage");
    static constexpr bool kIsLinear1 = Stage == MegaMoEStage::Linear1;
    using WarpOnM = Int<BM / WM>;
    using WarpOnN = Int<BN / WN>;
    using Arch = cutlass::arch::PPU0015;
    using StrideCD = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
    using ElementD = std::conditional_t<kIsLinear1, uint8_t, cutlass::bfloat16_t>;
    static constexpr auto kGemmType = GemmType::MegaMoE;
    static constexpr auto kEpilogue = kIsLinear1 ? EpilogueType::SiluAndMulPostQuantFp4 : EpilogueType::Default;

    using OperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        Arch, cutlass::float4_t, false, Int<BM>, Int<BK>, false>;
    using OperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        Arch, cutlass::float4_t, false, Int<BN>, Int<BK>, true>;
    using OperandSFA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        Arch, uint16_t, true, Int<(BM < 16 ? 16 : BM)>, Int<BK / 32>, true, 0, false>;
    using OperandSFB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        Arch, uint16_t, true, Int<(BN < 16 ? 16 : BN)>, Int<BK / 32>, true, 0, false>;
    using Mma = TiledMMA<MMA_Atom<PPU0015_16x16x64_F32F4F4F32_TN>,
                         Layout<Shape<WarpOnM, WarpOnN, _1>>>;
    using Mainloop = cutlass::gemm::collective::CollectiveMmaScaleFp4<
        cutlass::gemm::MainloopWithScalePPU0015Aiu<Stages>, Shape<Int<BM>, Int<BN>, Int<BK>>,
        cutlass::float4_t, cutlass::detail::TagToStrideA_t<cutlass::layout::RowMajor>,
        cutlass::float4_t, cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>, Mma,
        typename OperandA::GmemTiledCopy, typename OperandA::SmemLayoutAtom,
        typename OperandA::SmemCopyAtom, identity,
        typename OperandB::GmemTiledCopy, typename OperandB::SmemLayoutAtom,
        typename OperandB::SmemCopyAtom, identity,
        uint16_t, cutlass::detail::TagToStrideA_t<cutlass::layout::ColumnMajor>,
        typename OperandSFA::GmemTiledCopy, typename OperandSFA::SmemLayoutAtom,
        uint16_t, cutlass::detail::TagToStrideB_t<cutlass::layout::RowMajor>,
        typename OperandSFB::GmemTiledCopy, typename OperandSFB::SmemLayoutAtom,
        FP4DynamicTileId::Disabled,
        kIsLinear1 ? DispatchBufferLayout::k_scale_blocks_for_hidden(K * 2) : 0,
        kIsLinear1 && Ranks == 8 && N == 4096 && K == 2048 && Groups == 32>;
    using ActivationEpilogue = cutlass::epilogue::collective::EpilogueSiluAndMulPostQuant<
        StrideCD, StrideCD,
        cutlass::epilogue::thread::LinearCombination<
            ElementD, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing,
            cutlass::FloatRoundStyle::round_to_nearest, float>,
        cutlass::gemm::EpilogueDefault, N % BN == 0, ApplyClamp, PreserveBf16>;
    using ScatterEpilogue = epilogue::Fp4ScatterEpilogue<
        StrideCD, StrideCD, Layout<Shape<Int<(BM / WM) * 16>, Int<BN>>, cute::Stride<Int<BN>, _1>>,
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>, float, ElementD,
        cutlass::epilogue::thread::LinearCombination<ElementD, 16, float, float>,
        Ranks, WideScatterOffset, FuseReduce,
        Ranks == 8 && N == 4096 && K == 1024 && Groups == 32>;
    using Epilogue = std::conditional_t<kIsLinear1, ActivationEpilogue, ScatterEpilogue>;
    using Scheduler = DeepGemmScheduler<
        kGemmType, N, K, BM, BN, Groups, ceil_div(N, BN),
        !kIsLinear1 && Ranks == 8 ? 16 : 2, kEpilogue, true,
        Ranks, CopyBlocks, KTilesPerFlag, SourceCapacity, TopK>;
    using Kernel = DeepGemmUniversal<Shape<int, int, int, int>,
                                    Mainloop, Epilogue, Scheduler, false, 1, false, kEpilogue>;

    static_assert(is_supported_megamoe_rank_count(Ranks), "unsupported MegaMoE rank count");
    static_assert(BM == 16 || BM == 32 || BM == 64 || BM == 128 || BM == 256,
                  "unsupported MegaMoE block M");
    static_assert(!kIsLinear1 ||
                  (K > 0 && N > 0 && Groups > 0 && N % 64 == 0 &&
                   DispatchBufferLayout::k_scale_blocks_for_hidden(K * 2) % 8 == 0 &&
                   Kernel::MaxThreadsPerBlock >= Groups && Kernel::MaxThreadsPerBlock >= 128 &&
                   CopyBlocks > 0 && KTilesPerFlag > 0 && SourceCapacity > 0 && TopK > 0 &&
                   ceil_div(K / BK, KTilesPerFlag) <= 7 &&
                   ActivationEpilogue::get_shared_storage_size(BM, BN) <= Kernel::SharedStorageSize),
                  "invalid fused dispatch resources");
};

}  // namespace deep_gemm
