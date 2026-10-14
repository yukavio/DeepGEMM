import argparse
import json
import tempfile
from bisect import bisect_right
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple
import triton
import triton.language as tl

import deep_gemm
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, do_bench, is_legacy_loaded = None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        from tilelang.profiler.bench import do_bench
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
        dist_print(once_in_node=True)
    return deep_ep, do_bench, is_legacy_loaded


def _kineto_stage_times(events, kernel_names, split_after, num_tests, exclude_kernel_names=()):
    """Sum kernels per eager forward, splitting at ordered launch checkpoints.

    CPU ranges identify launches only; durations come from device kernel events.
    Correlation IDs include communication streams without counting GPU range
    annotations, memcpy, or the benchmark's cache flush and barrier kernels.
    """
    forwards = sorted((e for e in events if e.get('cat') == 'user_annotation'
                       and e['name'] == 'bench_kineto.forward'), key=lambda e: e['ts'])
    if len(forwards) != num_tests:
        raise RuntimeError(f'Expected {num_tests} profiled forwards, found {len(forwards)}')
    starts = [e['ts'] for e in forwards]
    launches = {e['args']['correlation']: e for e in events
                if e.get('cat') in ('cuda_runtime', 'cuda_driver')
                and 'correlation' in e.get('args', {})}
    iterations = [[] for _ in forwards]
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        launch = launches.get(event.get('args', {}).get('correlation'))
        if launch is None:
            raise RuntimeError(f'Missing launch correlation for {event["name"]}')
        index = bisect_right(starts, launch['ts']) - 1
        if index < 0 or launch['ts'] >= forwards[index]['ts'] + forwards[index]['dur']:
            continue
        if (launch['pid'], launch['tid']) != (forwards[index]['pid'], forwards[index]['tid']):
            raise RuntimeError('Stage profiling requires launches on the profiling thread')
        if 'GraphLaunch' in launch['name']:
            raise RuntimeError('Stage profiling requires eager forwards, not graph replay')
        if not any(name in event['name'] for name in exclude_kernel_names):
            iterations[index].append((launch['ts'], event))

    totals = [0.0, 0.0]
    for index, iteration in enumerate(iterations):
        kernels = [event for _, event in sorted(iteration, key=lambda item: item[0])]
        checkpoints = [i for i, event in enumerate(kernels)
                       if any(name in event['name'] for name in kernel_names)]
        if len(checkpoints) != len(kernel_names) or any(
                name not in kernels[i]['name'] for name, i in zip(kernel_names, checkpoints)):
            names = [e['name'] for e in kernels]
            raise RuntimeError(f'Unexpected kernel sequence in forward {index}: {names}')
        boundary = checkpoints[split_after - 1] + 1
        totals[0] += sum(e['dur'] for e in kernels[:boundary])
        totals[1] += sum(e['dur'] for e in kernels[boundary:])
    return tuple(t / num_tests / 1e6 for t in totals)


def bench_baseline_stages(fn, kernel_names, *, split_after, barrier=None,
                          trace_path=None, exclude_kernel_names=(), num_tests=30):
    """Profile full eager forwards; split kernel sums using ordered checkpoints.

    Match bench_kineto's warmup, active iterations, 8 GB cache flush and barrier.
    CPU annotations only associate launches; timings come from device kernels.
    """
    if not 0 < split_after < len(kernel_names) or num_tests <= 0:
        raise ValueError('Stage profiling requires checkpoints on both sides of the split and num_tests > 0')
    fn()
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1), acc_events=True)
    with profiler:
        for _ in range(2):
            for _ in range(num_tests):
                torch.empty(int(8e9 // 4), dtype=torch.int, device='cuda').zero_()
                if barrier is not None:
                    torch.cuda._sleep(int(2e7))
                    barrier()
                with torch.profiler.record_function('bench_kineto.forward'):
                    fn()
            torch.cuda.synchronize()
            profiler.step()
    with tempfile.TemporaryDirectory() as temp_dir:
        path = trace_path or os.path.join(temp_dir, 'trace.json')
        profiler.export_chrome_trace(path)
        with open(path) as trace:
            events = json.load(trace)['traceEvents']
    return _kineto_stage_times(events, kernel_names, split_after, num_tests, exclude_kernel_names)


@triton.jit
def _expanded_metadata(psum, counts, indices, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    end = tl.load(psum + expert)
    start = tl.load(psum + expert - 1, mask=expert > 0, other=0)
    tl.store(counts + expert, end - start)
    for first in range(start, end, BLOCK):
        rows = first + tl.arange(0, BLOCK)
        tl.store(indices + rows, expert, mask=rows < end)


@triton.jit
def _quantize_received(
    x, packed, scales, num_tokens, STRIDE_X: tl.constexpr,
    STRIDE_SF: tl.constexpr, H: tl.constexpr, BLOCK_K: tl.constexpr,
    SWIGLU: tl.constexpr, CLAMP: tl.constexpr,
):
    """Only visit valid expanded rows; unused receive capacity is uninitialized."""
    offsets = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    packed_offsets = tl.program_id(0) * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    sf_offsets = tl.program_id(0) * (BLOCK_K // 64) + tl.arange(0, BLOCK_K // 64)
    valid_tokens = tl.load(num_tokens)
    for row in tl.range(tl.program_id(1), valid_tokens, tl.num_programs(1), num_stages=4):
        value = tl.load(x + row * STRIDE_X + offsets, mask=offsets < H, other=0).to(tl.float32)
        if SWIGLU:
            up = tl.load(x + row * STRIDE_X + H + offsets, mask=offsets < H, other=0).to(tl.float32)
            if CLAMP > 0:
                value = tl.minimum(value, CLAMP)
                up = tl.maximum(tl.minimum(up, CLAMP), -CLAMP)
            value = value * tl.sigmoid(value) * up
        groups = tl.reshape(value, [BLOCK_K // 32, 32])
        amax = tl.max(tl.abs(groups), axis=1)
        if SWIGLU:
            scale = tl.maximum(amax, 1.0e-10) / 6.0
            bits = (scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
            exponent = bits >> 23
        else:
            # Match mega_moe_pre_dispatch's BF16 scale calculation.
            reciprocal = tl.full((), 1.0 / 6.0, tl.float32).to(tl.bfloat16).to(tl.float32)
            scale = (amax * reciprocal).to(tl.bfloat16)
            bits = scale.to(tl.uint16, bitcast=True).to(tl.uint32)
            exponent = ((bits & 0x7FFF) + 0x7F) >> 7
        multiplier = tl.exp2(127.0 - exponent.to(tl.float32))
        values = tl.reshape(groups * multiplier[:, None], [BLOCK_K])
        lo, hi = tl.split(tl.reshape(values, [BLOCK_K // 2, 2]))
        quantized = tl.inline_asm_elementwise(
            "{ .reg .b8 r; cvt.rn.satfinite.e2m1x2.f32 r, $1, $2; mov.b32 $0, {r, r, r, r}; }",
            constraints="=r,f,f", args=[hi, lo], dtype=tl.uint8, is_pure=True, pack=1,
        )
        sf_lo, sf_hi = tl.split(tl.reshape(exponent, [BLOCK_K // 64, 2]))
        sf = (sf_lo | (sf_hi << 8)).to(tl.uint16)
        tl.store(packed + row * (H // 2) + packed_offsets, quantized, mask=packed_offsets < H // 2)
        tl.store(scales + row + sf_offsets * STRIDE_SF, sf, mask=sf_offsets < H // 64)


@triton.jit
def _apply_router_weights(x, weights, out, num_tokens, H: tl.constexpr, BLOCK: tl.constexpr):
    columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid_tokens = tl.load(num_tokens)
    for row in range(tl.program_id(1), valid_tokens, tl.num_programs(1)):
        values = tl.load(x + row * H + columns, mask=columns < H, other=0).to(tl.float32)
        weight = tl.load(weights + row)
        tl.store(out + row * H + columns, values * weight, mask=columns < H)


def quantize_received(x, psum, hidden, *, swiglu=False, clamp=0.0):
    rows = x.shape[0]
    packed = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty_strided(
        (rows, hidden // 64), (1, rows), dtype=torch.uint16, device=x.device)
    _quantize_received[(triton.cdiv(hidden, 256), 128)](
        x, packed, scales, psum[-1:], STRIDE_X=x.stride(0), STRIDE_SF=rows,
        H=hidden, BLOCK_K=256, SWIGLU=swiglu, CLAMP=clamp, num_warps=4,
    )
    return packed, scales


@triton.jit
def _swiglu_masked_post_quant_to_mxfp4_kernel(
    x_ptr,
    stride_x_e,
    stride_x_m,
    stride_x_n,
    y_ptr,
    stride_y_e,
    stride_y_m,
    stride_y_n,
    y_sf_ptr,
    stride_sf_e,
    stride_sf_h,
    stride_sf_m,
    masked_m_ptr,
    SWIGLU_LIMIT: tl.constexpr,
    H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Masked SwiGLU followed by per-32 E2M1/UE8M0 quantization."""
    GROUP_SIZE: tl.constexpr = 32
    QUANT_MAX: tl.constexpr = 6.0
    NUM_GROUPS: tl.constexpr = BLOCK_K // GROUP_SIZE
    NUM_SCALE_PAIRS: tl.constexpr = NUM_GROUPS // 2

    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)
    pid_e_i64 = pid_e.to(tl.int64)

    num_token_stripes = tl.num_programs(1)
    num_valid_tokens = tl.load(masked_m_ptr + pid_e).to(tl.int64)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_out_k = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    offs_scale_pairs = pid_k * NUM_SCALE_PAIRS + tl.arange(0, NUM_SCALE_PAIRS)
    mask_k = offs_k < H
    mask_out_k = offs_out_k < (H // 2)
    mask_scale = offs_scale_pairs < (H // GROUP_SIZE // 2)

    x_base = x_ptr + pid_e_i64 * stride_x_e + offs_k * stride_x_n
    y_base = y_ptr + pid_e_i64 * stride_y_e + offs_out_k * stride_y_n
    sf_base = (
        y_sf_ptr + pid_e_i64 * stride_sf_e
        + offs_scale_pairs * stride_sf_h)

    for token in tl.range(
            pid_m, num_valid_tokens, num_token_stripes,
            num_stages=NUM_STAGES):
        gate = tl.load(
            x_base + token * stride_x_m,
            mask=mask_k, other=0.0).to(tl.float32)
        up = tl.load(
            x_base + token * stride_x_m + H * stride_x_n,
            mask=mask_k, other=0.0).to(tl.float32)
        if SWIGLU_LIMIT != float('inf'):
            gate = tl.minimum(gate, SWIGLU_LIMIT)
            up = tl.maximum(tl.minimum(up, SWIGLU_LIMIT), -SWIGLU_LIMIT)
        value = gate * tl.sigmoid(gate) * up

        grouped = tl.reshape(value, [NUM_GROUPS, GROUP_SIZE])
        amax = tl.maximum(tl.max(tl.abs(grouped), axis=1), 1.0e-10)
        dequant_scale = amax / QUANT_MAX
        scale_bits = dequant_scale.to(tl.uint32, bitcast=True)
        scale_bits = (scale_bits + 0x007FFFFF) & 0x7F800000
        rounded_scale = scale_bits.to(tl.float32, bitcast=True)
        quant_scale = 1.0 / rounded_scale

        quant_values = tl.reshape(
            grouped * tl.reshape(quant_scale, [NUM_GROUPS, 1]), [BLOCK_K])
        pairs = tl.reshape(quant_values, [BLOCK_K // 2, 2])
        lo_f, hi_f = tl.split(pairs)
        packed = tl.inline_asm_elementwise(
            """
            {
                .reg .b8 r;
                cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
                mov.b32 $0, {r, r, r, r};
            }
            """,
            constraints="=r,f,f",
            args=[hi_f.to(tl.float32), lo_f.to(tl.float32)],
            dtype=tl.uint8,
            is_pure=True,
            pack=1,
        )

        scale_pairs = tl.reshape(scale_bits, [NUM_SCALE_PAIRS, 2])
        lo_u32, hi_u32 = tl.split(scale_pairs)
        scale_u16 = ((lo_u32 >> 23) | ((hi_u32 >> 23) << 8)).to(tl.uint16)

        tl.store(y_base + token * stride_y_m, packed, mask=mask_out_k)
        tl.store(sf_base + token * stride_sf_m, scale_u16, mask=mask_scale)


def swiglu_masked_post_quant_to_mxfp4(
    x: torch.Tensor,
    masked_m: torch.Tensor,
    out_fp4: torch.Tensor,
    out_scale_physical: torch.Tensor,
    swiglu_limit: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the common test-only SwiGLU/MXFP4 suffix kernel in-place."""
    if not (
        x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
        and x.ndim == 3 and x.shape[-1] % 2 == 0
    ):
        raise AssertionError("invalid masked SwiGLU input")
    experts, capacity, two_hidden = x.shape
    hidden = two_hidden // 2
    if hidden % 64:
        raise AssertionError(f"intermediate hidden must be divisible by 64, got {hidden}")
    if masked_m.shape != (experts,) or masked_m.dtype != torch.int32:
        raise AssertionError("invalid masked_m")
    if out_fp4.shape != (experts, capacity, hidden // 2):
        raise AssertionError("invalid SwiGLU FP4 output")
    if out_scale_physical.shape != (experts, hidden // 64, capacity):
        raise AssertionError("invalid SwiGLU scale output")

    block_k = 256 if hidden > 256 else (128 if hidden > 128 else 64)
    token_stripes = 64 if experts < 4 else 32
    grid = (triton.cdiv(hidden, block_k), token_stripes, experts)
    _swiglu_masked_post_quant_to_mxfp4_kernel[grid](
        x,
        x.stride(0), x.stride(1), x.stride(2),
        out_fp4,
        out_fp4.stride(0), out_fp4.stride(1), out_fp4.stride(2),
        out_scale_physical,
        out_scale_physical.stride(0),
        out_scale_physical.stride(1),
        out_scale_physical.stride(2),
        masked_m,
        SWIGLU_LIMIT=float('inf') if swiglu_limit is None else swiglu_limit,
        H=hidden,
        BLOCK_K=block_k,
        NUM_STAGES=4,
        num_warps=4,
    )
    return out_fp4, out_scale_physical.permute(0, 2, 1)


# PPU port of DeepGEMM@057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7; MXFP4 x MXFP4 only.
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens)) \
        if args.num_tokens == 0 else args.num_tokens
    num_shared_experts = args.num_shared_experts
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    shared_intermediate_hidden = intermediate_hidden * num_shared_experts
    assert num_tokens <= num_max_tokens_per_rank
    assert num_shared_experts == 0, 'PPU MegaMoE does not support shared experts'
    assert args.fast_math == 1 and not args.use_localization, 'PPU MegaMoE requires fast_math without localization'

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden
    )

    # Cast routed weights into MXFP4
    def _cast_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.uint8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.uint8)
        for i in range(num_groups):
            w[i], w_sf[i] = downcast_to_mxfp(bf16_weights[i], torch.uint8, axis=1)
        return w, w_sf

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, topk_idx, topk_weights, l1_weights, l2_weights
        global transformed_l1_weights, transformed_l2_weights
        global cumulative_local_expert_recv_stats_fused, cumulative_local_expert_recv_stats_baseline
        global initial_cumulative_local_expert_recv_stats_fused, initial_cumulative_local_expert_recv_stats_baseline
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        initial_cumulative_local_expert_recv_stats_fused = cumulative_local_expert_recv_stats_fused.clone()
        initial_cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_baseline.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        l1_weights = _cast_weights_to_fp4(l1_weights)
        l2_weights = _cast_weights_to_fp4(l2_weights)
        transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(
            l1_weights, l2_weights)
        l1_weights = l1_weights[0], deep_gemm.preprocess_mxfp4_scales(scale=l1_weights[1])
        l2_weights = transformed_l2_weights

    # Run fused mega MoE
    def copy_inputs_to_buffer():
        deep_gemm.mega_moe_pre_dispatch(x, topk_idx.to(torch.int32), topk_weights, sym_buffer=buffer)

    def run_fused():
        cumulative_local_expert_recv_stats_fused.copy_(initial_cumulative_local_expert_recv_stats_fused)
        copy_inputs_to_buffer()

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        kernel_kwargs = dict(
            y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
            sym_buffer=buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation_clamp=args.activation_clamp or None,
            fast_math=bool(args.fast_math))
        deep_gemm.fp4_fp4_mega_moe(**kernel_kwargs)
        # The C++ backend reserves the statistics argument without updating it.
        return y, cumulative_local_expert_recv_stats_fused

    dist_print('Config:', once_in_node=True)
    dist_print(f' > MMA: {args.mma_type}', once_in_node=True)
    dist_print(f' > Localized weights: {args.use_localization}', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Shared experts: {num_shared_experts}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Baseline: {args.baseline}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.numel() / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)
        run_fused()
        dist_print(f' > Done, exiting', once_in_node=True)

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine
    deep_ep, tilelang_bench, is_legacy_loaded = import_baseline()
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    ep_buffer, run_baseline = None, None
    if is_legacy_loaded:
        if args.baseline == 'v2':
            alignment = 1  # PPU GroupedNoPad consumes unpadded expert ranges.
            ep_buffer_type = (getattr(deep_ep, 'EPBuffer', None) or deep_ep.ElasticBuffer)
            ep_buffer = ep_buffer_type(
                group,
                num_max_tokens_per_rank=buffer.num_max_tokens_per_rank, hidden=hidden,
                num_topk=num_topk, use_fp8_dispatch=False,
                explicitly_destroy=True,
                allow_multiple_reduction=False,
                num_gpu_timeout_secs=10, num_cpu_timeout_secs=30
            )

            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False, 'do_expand': True}
            gemm_fn = deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad
            get_num_tokens = lambda recv_x: recv_x.size(0)

            def run_baseline():
                cumulative_local_expert_recv_stats_baseline.copy_(initial_cumulative_local_expert_recv_stats_baseline)
                # Dispatch
                recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                    x, topk_idx=topk_idx, topk_weights=topk_weights,
                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
                    num_experts=num_experts, expert_alignment=alignment,
                    **dispatch_kwargs)
                num_recv_tokens = get_num_tokens(recv_x)
                psum = handle.psum_num_recv_tokens_per_expert
                counts = torch.empty_like(psum)
                indices = torch.empty(num_recv_tokens, dtype=torch.int32, device='cuda')
                _expanded_metadata[(num_experts_per_rank,)](psum, counts, indices, BLOCK=256)
                recv_x = quantize_received(recv_x, psum, hidden)

                # L1 GEMM
                l1_y = torch.empty((num_recv_tokens, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
                gemm_fn(recv_x, l1_weights, None, l1_y, indices, counts)

                # SwiGLU
                l1_y = quantize_received(l1_y, psum, intermediate_hidden, swiglu=True, clamp=args.activation_clamp)

                # L2 GEMM
                l2_y = torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                gemm_fn(l1_y, l2_weights, None, l2_y, indices, counts)

                # PPU MegaMoE applies router weights after GEMM2.
                weighted_y = torch.empty_like(l2_y)
                _apply_router_weights[(triton.cdiv(hidden, 256), 128)](
                    l2_y, recv_topk_weights, weighted_y, psum[-1:], H=hidden, BLOCK=256)

                # Combine
                return (
                    ep_buffer.combine(weighted_y, handle=handle)[0],
                    cumulative_local_expert_recv_stats_baseline
                )
        elif args.baseline == 'v1-normal':
            assert num_ranks == num_local_ranks or int(os.environ.get('DEEPEP_NUM_MAX_ICN_PEERS', '0')) == num_ranks, \
                'V1 normal requires intranode configs; set DEEPEP_NUM_MAX_ICN_PEERS to the EP size for multi-node ICN'
            gemm_fn = deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad
            dispatch_config = deep_ep.Buffer.get_intranode_dispatch_config(num_ranks, hidden // 2, "quant")
            combine_config = deep_ep.Buffer.get_intranode_combine_config(num_ranks, hidden)
            nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden * 2, num_ranks)
                            for config in (dispatch_config, combine_config))
            ep_buffer = deep_ep.Buffer(
                dist.group.WORLD, nvl_bytes, 0, low_latency_mode=False,
                num_qps_per_rank=1, explicitly_destroy=True)

            # Reuse the existing PPU scatter/gather and activation kernels.
            from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter_sail, ep_gather
            try:
                from sglang.kernels.ops.elementwise.silu_mul_quant import silu_and_mul_post_quant_mxfp4
            except ImportError as error:
                module = 'sglang.kernels.ops.elementwise.silu_mul_quant'
                if not error.name or not (module == error.name or module.startswith(error.name + '.')):
                    raise
                from sglang.jit_kernel.silu_mul_quant import silu_and_mul_post_quant_mxfp4

            def run_baseline():
                cumulative_local_expert_recv_stats_baseline.copy_(initial_cumulative_local_expert_recv_stats_baseline)
                # Dispatch
                input_x = downcast_to_mxfp(x, torch.uint8, axis=1)
                per_rank, per_rdma_rank, per_expert, in_rank, _ = ep_buffer.get_dispatch_layout(
                    topk_idx, num_experts, async_finish=False)
                recv_x, recv_topk_idx, recv_topk_weights, counts, handle, _ = ep_buffer.dispatch(
                    input_x, topk_idx=topk_idx, topk_weights=topk_weights,
                    num_tokens_per_rank=per_rank, num_tokens_per_rdma_rank=per_rdma_rank,
                    is_token_in_rank=in_rank, num_tokens_per_expert=per_expert,
                    async_finish=False, expert_alignment=1, config=dispatch_config)
                num_recv_tokens = sum(counts)
                counts = torch.tensor(counts, dtype=torch.int32, device='cpu', pin_memory=True).to('cuda', non_blocking=True)
                cumulative_local_expert_recv_stats_baseline.add_(counts)
                gathered_y = torch.empty((recv_x[0].shape[0], hidden), dtype=torch.bfloat16, device='cuda')
                if num_recv_tokens:
                    # Convert the V1 token-major receive layout to PPU GroupedNoPad.
                    recv_data, recv_scale = recv_x
                    grouped_x = torch.empty((num_recv_tokens, hidden // 2), dtype=torch.uint8, device='cuda')
                    grouped_sf = torch.empty((num_recv_tokens, hidden // 32), dtype=torch.uint8, device='cuda')
                    indices = torch.empty(num_recv_tokens, dtype=torch.int32, device='cuda')
                    output_index = torch.empty_like(recv_topk_idx)
                    expert_start = torch.empty_like(counts)
                    ep_scatter_sail(
                        recv_data, recv_topk_idx.to(torch.int32), counts, expert_start,
                        grouped_x, indices, output_index, recv_scale, grouped_sf,
                        BLOCK_E=1, BLOCK_D=16, is_block_wise=True)
                    recv_x = grouped_x, deep_gemm.preprocess_mxfp4_scales(scale=grouped_sf)

                    # L1 GEMM
                    l1_y = torch.empty((num_recv_tokens, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
                    gemm_fn(recv_x, l1_weights, None, l1_y, indices, counts)

                    # SwiGLU
                    l1_y = silu_and_mul_post_quant_mxfp4(l1_y, swiglu_limit=args.activation_clamp or None)

                    # L2 GEMM
                    l2_y = torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    gemm_fn(l1_y, l2_weights, None, l2_y, indices, counts)
                    ep_gather(l2_y, recv_topk_idx, recv_topk_weights, output_index, gathered_y)
                else:
                    gathered_y.zero_()

                # Combine
                return (
                    ep_buffer.combine(gathered_y, handle=handle, async_finish=False, config=combine_config)[0],
                    cumulative_local_expert_recv_stats_baseline
                )
        else:  # V1 low latency
            local_ranks = int(os.environ.get('LOCAL_WORLD_SIZE', num_ranks))
            os.environ['DEEPEP_DISABLE_NETWORK'] = '0' if num_ranks > local_ranks else '1'
            os.environ.pop('SAILSHMEM_DISABLE_MNNVL', None)
            ep_buffer = deep_ep.Buffer(
                dist.group.WORLD,
                num_rdma_bytes=deep_ep.Buffer.get_low_latency_rdma_size_hint(
                    buffer.num_max_tokens_per_rank, hidden, num_ranks, num_experts),
                low_latency_mode=True, num_qps_per_rank=num_experts_per_rank,
                allow_nvlink_for_low_latency_mode=True, allow_mnnvl=True, explicitly_destroy=True)

            gemm_fn = deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked
            expected_m = num_tokens * num_topk // num_experts_per_rank + 1
            expected_m = min(num_ranks * buffer.num_max_tokens_per_rank, max(1, expected_m))

            def run_baseline():
                cumulative_local_expert_recv_stats_baseline.copy_(initial_cumulative_local_expert_recv_stats_baseline)
                # Dispatch
                recv_x, counts, handle, _, _ = ep_buffer.low_latency_dispatch(
                    x, topk_idx, buffer.num_max_tokens_per_rank, num_experts,
                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
                    use_mxfp4=True, quant_size=32, mxfp4_scale_row_major=False)
                num_recv_tokens = num_ranks * buffer.num_max_tokens_per_rank

                # L1 GEMM
                l1_y = torch.empty((num_experts_per_rank, num_recv_tokens, intermediate_hidden * 2),
                                   dtype=torch.bfloat16, device='cuda')
                gemm_fn(recv_x, l1_weights, None, l1_y, counts, expected_m)

                # SwiGLU
                l1_q = torch.empty((num_experts_per_rank, num_recv_tokens, intermediate_hidden // 2), dtype=torch.uint8, device='cuda')
                l1_sf = torch.empty((num_experts_per_rank, intermediate_hidden // 64, num_recv_tokens), dtype=torch.uint16, device='cuda')
                l1_y = swiglu_masked_post_quant_to_mxfp4(
                    l1_y, counts, l1_q, l1_sf, swiglu_limit=args.activation_clamp or None)

                # L2 GEMM
                l2_y = torch.empty((num_experts_per_rank, num_recv_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                gemm_fn(l1_y, l2_weights, None, l2_y, counts, expected_m)

                # Combine
                return (
                    ep_buffer.low_latency_combine(
                        l2_y, topk_idx, topk_weights, handle,
                        zero_copy=False, async_finish=False, return_recv_hook=False)[0],
                    cumulative_local_expert_recv_stats_baseline
                )

    # Check correctness
    # noinspection PyBroadException
    if is_legacy_loaded and num_correctness_tests > 0:
        dist_print('Running correctness tests:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            fused_y, fused_stats = run_fused()
            baseline_y, baseline_stats = run_baseline()
            assert torch.equal(fused_stats, initial_cumulative_local_expert_recv_stats_fused)
            expected_counts = torch.bincount(topk_idx[topk_idx >= 0].reshape(-1), minlength=num_experts)
            dist.all_reduce(expected_counts, group=group)
            expected_counts = expected_counts[rank_idx * num_experts_per_rank:(rank_idx + 1) * num_experts_per_rank]
            assert torch.equal(baseline_stats, initial_cumulative_local_expert_recv_stats_baseline + expected_counts)
            # Fused GEMM1 activates FP32 accumulators; the PPU baseline rounds to BF16.
            assert calc_diff(fused_y, baseline_y) < 0.002, f'calc_diff={calc_diff(fused_y, baseline_y)}'
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
    else:
        create_inputs()

    # Count local received tokens
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark
    if args.baseline == 'v2':
        barrier_kwargs = {'wait_comm_stream': False} if hasattr(deep_ep, 'EPBuffer') else {'use_comm_stream': False}
        barrier_fn = lambda: ep_buffer.barrier(**barrier_kwargs) if ep_buffer else dist.all_reduce(torch.empty(1, device='cuda'))
    elif args.baseline == 'v1-normal':
        # The intranode-only V1 buffer does not initialize SAILSHMEM for Buffer.barrier().
        barrier_fn = lambda: dist.all_reduce(torch.empty(1, device='cuda'))
    else:
        barrier_fn = lambda: ep_buffer.barrier() if ep_buffer else dist.all_reduce(torch.empty(1, device='cuda'))
    trace_path = None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json'
    # PPU MegaMoE uses four serialized kernels instead of the SM100 single kernel.
    kernel_names = ('megamoe_quant', 'megamoe_fused_dispatch_gemm1_swiglu', 'megamoe_scatter', 'megamoe_reduce')
    t_fused_kernels = bench_kineto(run_fused, kernel_names, barrier=barrier_fn, trace_path=trace_path)
    t_fused_stages = sum(t_fused_kernels[:2]), sum(t_fused_kernels[2:])
    t_fused = sum(t_fused_stages)
    # V1 normal reads receive counts on the CPU and cannot be captured.
    baseline_backend = 'event' if args.baseline == 'v1-normal' else 'cudagraph'
    t_baseline = tilelang_bench(
        run_baseline, _n_warmup=5, _n_repeat=1,
        backend=baseline_backend, return_mode='median') / 1e3 if is_legacy_loaded else 0

    t_baseline_stages = None
    if is_legacy_loaded and not int(os.environ.get('DG_USE_PPU_TOOLS', 0)):
        # Ordered checkpoints distinguish GEMM1/GEMM2 even when their kernel names match.
        exclude_kernels = ()
        if args.baseline == 'v2':
            baseline_kernels = ('_quantize_received', 'fp4_grouped_deep_gemm_nopad',
                                '_quantize_received', 'fp4_grouped_deep_gemm_nopad')
            split_after = 3
        elif args.baseline == 'v1-normal':
            # Empty receives still combine; exclude the test-only statistics update.
            baseline_kernels = ('fp4_grouped_deep_gemm_nopad', 'silu_and_mul_post_quant_mxfp4_kernel',
                                'fp4_grouped_deep_gemm_nopad') if num_recv_tokens else (
                                    '::intranode::dispatch<', '::intranode::combine<')
            split_after = 2 if num_recv_tokens else 1
            exclude_kernels = ('CUDAFunctor_add<int>',)
        else:
            baseline_kernels = ('fp4_grouped_deep_gemm_masked', '_swiglu_masked_post_quant_to_mxfp4_kernel',
                                'fp4_grouped_deep_gemm_masked')
            split_after = 2
        baseline_trace = None if not args.dump_profile_traces else f'{args.dump_profile_traces}/baseline_rank{rank_idx}.json'
        t_baseline_stages = bench_baseline_stages(
            run_baseline, baseline_kernels, barrier=barrier_fn, trace_path=baseline_trace,
            split_after=split_after, exclude_kernel_names=exclude_kernels)

    # TFLOPS: routed + shared L1/L2, each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    num_routed_flops = 2 * num_recv_tokens * hidden * intermediate_hidden * 3
    num_shared_flops = 2 * num_tokens * hidden * shared_intermediate_hidden * 3
    num_total_flops = num_routed_flops + num_shared_flops

    # HBM bytes: weights + activations + output
    num_touched_experts = torch.unique(gathered_topk_idx[gathered_topk_idx >= 0]).numel()
    act_elem_size = 0.5
    routed_weight_elem_size = 0.5
    shared_weight_elem_size = 0.5
    num_routed_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden * routed_weight_elem_size      # L1 weights
        + num_touched_experts * hidden * intermediate_hidden * routed_weight_elem_size        # L2 weights
        + num_recv_tokens * hidden * act_elem_size                                            # L1 acts read
        + num_recv_tokens * intermediate_hidden * act_elem_size                               # L1 output write
        + num_recv_tokens * intermediate_hidden * act_elem_size                               # L2 acts read
        + num_recv_tokens * hidden * 2                                                        # L2 output write (always BF16)
    )
    num_shared_hbm_bytes = 0 if num_shared_experts == 0 else (
        shared_intermediate_hidden * 2 * hidden * shared_weight_elem_size      # Shared L1 weights
        + hidden * shared_intermediate_hidden * shared_weight_elem_size        # Shared L2 weights
        + num_tokens * hidden * act_elem_size                                  # Shared L1 acts read
        + num_tokens * shared_intermediate_hidden * act_elem_size              # Shared L1 output write
        + num_tokens * shared_intermediate_hidden * act_elem_size              # Shared L2 acts read
        + num_tokens * hidden * 2                                              # Shared L2 output write
    )
    num_hbm_bytes = num_routed_hbm_bytes + num_shared_hbm_bytes

    # Communication bytes: MXFP4 dispatch + BF16 combine (scale/metadata bytes excluded)
    num_nvlink_bytes = num_recv_tokens * hidden * 2.5

    # Combine reduction (serial) time approximation
    t_reduction = 0  # No SM100 bandwidth assumption on PPU.

    # Summary
    def print_perf(elapsed: float, ref_time: float, ref_label: str):
        tflops = safe_div(num_total_flops / 1e12, elapsed)
        hbm_gbs = safe_div(num_hbm_bytes / 1e9, elapsed)
        nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, elapsed)
        approx_factor = safe_div(elapsed, elapsed - t_reduction)
        dist_print(f' > EP {rank_idx:2}/{num_ranks} | '
                   f'{tflops:4.0f} TFLOPS | '
                   f'overlap: '
                   f'{tflops * approx_factor:4.0f} TFLOPS, '
                   f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
                   f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
                   f'{elapsed * 1e6:4.0f} us, '
                   f'reduction: {t_reduction * 1e6:4.1f} us | '
                   f'{safe_div(ref_time, elapsed):.2f}x {ref_label}')

    dist_print(f'Performance (w/{"" if num_shared_experts else "o"} shared):', once_in_node=True)
    print_perf(t_fused, t_baseline, f'legacy{"+shared" if num_shared_experts else ""}')

    if not int(os.environ.get('DG_USE_PPU_TOOLS', 0)):
        dist_print('Stage breakdown (profiler mean kernel time, per rank):', once_in_node=True)
        for i, label in enumerate(('Stage1', 'Stage2', 'Kernel sum')):
            elapsed = t_fused_stages[i] if i < 2 else t_fused
            comparison = ''
            if t_baseline_stages is not None:
                ref_time = t_baseline_stages[i] if i < 2 else sum(t_baseline_stages)
                comparison = f' | legacy {ref_time * 1e6:.2f} us | {safe_div(ref_time, elapsed):.2f}x legacy'
            dist_print(f' > EP {rank_idx:2}/{num_ranks} | {label}: MegaMoE {elapsed * 1e6:.2f} us{comparison}')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PPU MXFP4 MegaMoE against DeepEP baselines')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max minus removed if 0)')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--num-shared-experts', type=int, default=0, help='Number of shared experts (use 0 to disable)')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')
    parser.add_argument('--mma-type', type=str, default='fp4xfp4', choices=('fp4xfp4',))
    parser.add_argument('--use-localization', action='store_true', help='Localize the weights into the locality domains (requires MLOPart)')

    # Test settings
    parser.add_argument('--baseline', choices=('v2', 'v1-normal', 'v1-low-latency'), default='v2',
                        help='DeepEP comparison path (default: v2, as in upstream SM100)')
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: each process is launched separately (e.g. by NCU)
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # Launch tests
        num_processes = args.num_processes
        torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
