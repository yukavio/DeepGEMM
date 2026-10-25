"""Public MegaMoE API and symmetric-buffer management."""

from __future__ import annotations

import weakref
import warnings
import types
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import deep_gemm_cpp as _C
from ..jit_kernels.utils import ceil_div

def _get_expected_m(num_tokens, num_topk, num_local_experts, owner_capacity):
    """SGLang's floor(mean) + 1 hint, bounded by one owner expert."""
    expected_m = num_tokens * num_topk // num_local_experts + 1
    return max(1, min(owner_capacity, expected_m))


__all__ = [
    "SymmBuffer",
    "get_symm_buffer_for_mega_moe",
    "transform_weights_for_mega_moe",
    "fp4_fp4_mega_moe",
    "mega_moe_pre_dispatch",
]


@dataclass
class _BufferState:
    # Aliases share published input shapes and asynchronous lifetimes.
    input_keepalive: Optional[tuple] = None


class SymmBuffer:
    """Fixed resources for the PPU two-stage MXFP4 MegaMoE functions.

    Allocates on the current CUDA device, selected by the caller before construction.
    Owns symmetric storage, persistent routing, intermediate activations and
    Graph resources. Prepare with mega_moe_pre_dispatch and compute with
    fp4_fp4_mega_moe.

    x is the parity-buffered FP4 dispatch view. x_sf is the actual pushed-SFA
    staging view in [parity, local_expert, source_rank, token, scale_block]
    order; the unused duplicate SFA band in dispatch storage is not allocated.
    base reuses an identical layout without reinitializing protocol state.
    Aliases must be used serially, including captured graphs, on all ranks.
    Call pre-dispatch before compute on the same stream; submission order is
    the caller's responsibility and is not tracked as a published round.
    """

    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp4xfp4',
                 activation: str = 'swiglu',
                 base: Optional['SymmBuffer'] = None):
        # Align token count to the PPU layout's 32-token granularity.
        if (not isinstance(num_max_tokens_per_rank, int)
                or isinstance(num_max_tokens_per_rank, bool)
                or num_max_tokens_per_rank <= 0):
            raise ValueError("num_max_tokens_per_rank must be a positive integer")
        num_max_tokens_per_rank = (num_max_tokens_per_rank + 31) // 32 * 32

        # Init
        assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

        # Allocate or reuse a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts
        )
        if base is None:
            allocator = torch if group.size() == 1 else symm_mem
            self.buffer = allocator.empty(num_bytes, dtype=torch.uint8, device='cuda')
            self.handle = (
                types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                if group.size() == 1
                else symm_mem.rendezvous(self.buffer, group=group)
            )
            self._state = _BufferState()
            self.buffer.zero_()
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            if (not isinstance(base, SymmBuffer) or base.buffer is None
                    or base.handle is None or base.group is not group):
                raise ValueError("Cannot reuse an invalid symmetric buffer or a different process group")
            # Shared token metadata and captured graphs require identical layouts.
            if any(getattr(base, name) != getattr(self, name) for name in (
                    'num_experts', 'num_max_tokens_per_rank', 'num_topk', 'hidden',
                    'intermediate_hidden', 'num_shared_experts', 'mma_type', 'activation')):
                raise ValueError("base reuse currently requires an identical MegaMoE layout")
            if num_bytes > base.buffer.nbytes:
                raise ValueError("base symmetric buffer is too small")
            self.buffer, self.handle, self._state = base.buffer, base.handle, base._state

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)
        self.x._mega_moe_buffer = weakref.ref(self)

    def destroy(self) -> None:
        """Release this alias after callers synchronize work and retire its graphs.

        No synchronization is performed. Other SymmBuffer aliases retain storage;
        exported non-owning views alone do not keep the allocation alive.
        """
        if self.buffer is None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SymmBuffer cannot be destroyed during graph capture")
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.topk_idx = self.topk_weights = None
        self.l1_acts = self.l1_acts_sf = self.l2_acts = self.l2_acts_sf = None
        self._state = None

    def _validate_inputs(self, hidden_states, topk_ids, topk_weights):
        if (not isinstance(hidden_states, torch.Tensor)
                or hidden_states.ndim != 2
                or hidden_states.dtype != torch.bfloat16
                or not hidden_states.is_contiguous()
                or hidden_states.shape[1] != self.hidden):
            raise ValueError(
                "hidden_states must be contiguous BF16 with shape "
                f"[tokens, {self.hidden}]")
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.num_max_tokens_per_rank:
            raise ValueError(
                f"token count must be in [0, {self.num_max_tokens_per_rank}], "
                f"got {num_tokens}")
        if (not isinstance(topk_ids, torch.Tensor)
                or topk_ids.shape != (num_tokens, self.num_topk)
                or topk_ids.dtype not in (torch.int32, torch.int64)
                or not topk_ids.is_contiguous()):
            raise ValueError(
                "topk_ids must be contiguous int32/int64 with shape "
                f"({num_tokens}, {self.num_topk})")
        if (not isinstance(topk_weights, torch.Tensor)
                or topk_weights.shape != topk_ids.shape
                or topk_weights.dtype != torch.float32
                or not topk_weights.is_contiguous()):
            raise ValueError(
                "topk_weights must be contiguous float32 and match topk_ids")
        if any(t.device != self.buffer.device
               for t in (hidden_states, topk_ids, topk_weights)):
            raise ValueError("activations and routing must be on the buffer device")

def _publish_mega_moe(sym_buffer, hidden_states, topk_ids, topk_weights, active_token_mask=None):
    sym_buffer._validate_inputs(hidden_states, topk_ids, topk_weights)
    if active_token_mask is not None and (
            active_token_mask.shape != (hidden_states.shape[0],)
            or active_token_mask.dtype != torch.bool
            or active_token_mask.device != hidden_states.device
            or not active_token_mask.is_contiguous()):
        raise ValueError("active_token_mask must be one contiguous device bool per token")
    _C._mega_moe_quant(
        hidden_states, topk_ids, topk_weights, sym_buffer.buffer, sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.size(), sym_buffer.num_experts,
        sym_buffer.num_max_tokens_per_rank, sym_buffer.group.rank(),
        sym_buffer.intermediate_hidden, active_token_mask)
    sym_buffer._state.input_keepalive = (hidden_states, topk_ids, topk_weights, active_token_mask)
    # Zero-row ranks still publish their empty route counts to every peer.


def get_symm_buffer_for_mega_moe(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    use_fp8_dispatch: Union[bool, None] = None,
    mma_type: str = "fp4xfp4",
    activation: str = "swiglu",
) -> SymmBuffer:
    """Create the PPU workspace with native MegaMoE dimensions.

    The workspace is filled by mega_moe_pre_dispatch and consumed by
    fp4_fp4_mega_moe. Only MXFP4 dispatch is supported.
    """
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )
    return SymmBuffer(
        group, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, num_shared_experts,
        mma_type=mma_type, activation=activation)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Prepare raw MXFP4 weight pairs for the PPU MegaMoE kernels once at load time.

    Inputs are contiguous uint8 packed weights and raw UE8M0 scales, before any
    interleaving or scale preprocessing. L1 gate/up rows are interleaved one by
    one; L2 weights are unchanged. Both scales become packed uint16 N-major.
    Use the returned pairs; do not transform already-interleaved weights again.
    """
    def validate_pair(name, pair):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f'{name} must be a (MXFP4 weight, raw UE8M0 scale) pair')
        weight, sf = pair
        if not isinstance(weight, torch.Tensor) or not isinstance(sf, torch.Tensor):
            raise TypeError(f'{name} weight and scale must both be tensors')
        if weight.ndim != 3 or sf.ndim != 3:
            raise ValueError(f'{name} weight and scale must both be rank-3 tensors')
        if weight.dtype != torch.uint8 or sf.dtype != torch.uint8:
            raise TypeError(f'{name} requires uint8 packed weights and raw uint8 scales')
        if weight.device != sf.device:
            raise ValueError(f'{name} weight and scale must be on the same device')
        if not weight.is_contiguous() or not sf.is_contiguous():
            raise ValueError(f'{name} weight and scale must use contiguous raw layouts')
        experts, n, packed_k = weight.shape
        if min(experts, n, packed_k) <= 0:
            raise ValueError(f'{name} dimensions must be positive')
        expected_sf_shape = (experts, n, ceil_div(packed_k, 16))
        if tuple(sf.shape) != expected_sf_shape:
            raise ValueError(f'{name} scale shape must be {expected_sf_shape}, got {tuple(sf.shape)}')
        return weight, sf

    l1_fp4, l1_sf = validate_pair('l1_weights', l1_weights)
    l2_fp4, l2_sf = validate_pair('l2_weights', l2_weights)
    hidden, intermediate = l1_fp4.shape[2] * 2, l2_fp4.shape[2] * 2
    if (l1_fp4.shape[0] != l2_fp4.shape[0] or
            l1_fp4.shape[1] != 2 * intermediate or l2_fp4.shape[1] != hidden):
        raise ValueError('MXFP4 MegaMoE weights must have shapes (E, 2*I, H/2) and (E, H, I/2)')
    if hidden < 2048 or hidden % 1024 or intermediate % 64:
        raise ValueError('MXFP4 MegaMoE requires H >= 2048 divisible by 1024 and I divisible by 64')
    if l1_fp4.device != l2_fp4.device:
        raise ValueError('L1 and L2 MXFP4 MegaMoE weights must be on the same device')
    l1_transformed = _C.preprocess_mxfp4_weight_for_act_and_quant_fusing(l1_fp4, l1_sf)
    l2_transformed = (l2_fp4, _C.preprocess_mxfp4_scales(l2_sf))
    return l1_transformed, l2_transformed


def mega_moe_pre_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x: Optional[torch.Tensor] = None,
    x_sf: Optional[torch.Tensor] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights_out: Optional[torch.Tensor] = None,
    *,
    sym_buffer: Optional[SymmBuffer] = None,
    num_tokens: Optional[int] = None,
    group_size: int = 32,
    use_fp4_acts: bool = True,
    active_token_mask: Optional[torch.Tensor] = None,
) -> None:
    """Publish MXFP4 inputs to a SymmBuffer without an intermediate copy.

    Existing SGLang callers may pass its four exact output views instead of
    sym_buffer. Valid expert IDs in each token row must be unique; invalid
    IDs are masked by the device publication kernel. Int32 and int64 route
    inputs are accepted; active_token_mask can suppress rows without a separate
    packing kernel. The buffer keeps the mask alive through Graph replay.
    """
    if group_size != 32 or not use_fp4_acts:
        raise ValueError("PPU MegaMoE requires MXFP4 with group_size=32")
    views = (x, x_sf, topk_idx, topk_weights_out)
    if sym_buffer is not None:
        if any(view is not None for view in views):
            raise ValueError("pass sym_buffer or the four output views, not both")
        if not isinstance(sym_buffer, SymmBuffer):
            raise TypeError("a live MegaMoE SymmBuffer is required")
        buffer = sym_buffer
    else:
        owner = getattr(x, "_mega_moe_buffer", None)
        buffer = owner() if isinstance(owner, weakref.ReferenceType) else None
        if buffer is None or any(actual is not expected for actual, expected in zip(
                views, (buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights))):
            raise ValueError("output tensors must belong to the same live MegaMoE buffer")
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("hidden_states must be a 2D tensor")
    if num_tokens is None:
        num_tokens = hidden_states.shape[0]
    if (not isinstance(num_tokens, int) or isinstance(num_tokens, bool)
            or not 0 <= num_tokens <= hidden_states.shape[0]):
        raise ValueError("num_tokens is outside the input tensor")
    if not all(isinstance(t, torch.Tensor) and t.ndim == 2
               for t in (topk_ids, topk_weights)):
        raise ValueError("topk_ids and topk_weights must be 2D tensors")
    _publish_mega_moe(
        buffer, hidden_states[:num_tokens], topk_ids[:num_tokens],
        topk_weights[:num_tokens],
        active_token_mask[:num_tokens] if active_token_mask is not None else None)


# TODO: review API alignment with upstream fp8_fp4_mega_moe and SGLang callers.
# This backend is FP4xFP4, not upstream's FP8-activation path; identical argument
# names do not imply interchangeable weight/scales or SymmBuffer layouts.
# Shared-expert weight arguments are intentionally omitted for this backend.
def fp4_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    expected_m: Optional[int] = None,  # Extra scheduling-hint override; absent upstream.
    preserve_bf16: bool = False,
    fuse_reduce: bool = False,
    after_gemm1: Optional[Callable[[], None]] = None,
) -> None:
    """Consume published buffers through the complete C++ MegaMoE entry.

    y must be contiguous BF16 [C, hidden] on the buffer device, with C >= T.
    T comes from the most recent pre-dispatch input. Only y[:T] is written;
    any remaining rows are left untouched.
    None or +inf disables activation clamping.
    expected_m optionally overrides the shape-only scheduling hint for tests.
    preserve_bf16 preserves the standalone BF16 GEMM1/activation boundary and
    its FP32 scale arithmetic inside the fused epilogue, without a BF16 store.
    fuse_reduce performs the ordered weighted reduction after GEMM2's peer
    completion barrier in the same kernel, removing the separate reduce launch.
    after_gemm1, when provided, runs on the submitting host thread after G1
    is enqueued and before G2 is enqueued. It does not synchronize the device.
    It may enqueue independent work on another stream; that stream must wait
    on the current stream before consuming G1 results and be joined by its
    owner. During Graph replay only captured device work runs, not the Python
    callback. Warm and capture with the same callback and retain its storage.
    cumulative_local_expert_recv_stats is a reserved CUDA int32 vector with
    one entry per local expert. It is forwarded to the C++ JIT implementation
    for API alignment; this backend does not update the counters yet.
    Callers must retain weights and the optional statistics tensor until GPU
    execution completes, including every replay of a captured Graph.
    """
    if after_gemm1 is not None and not callable(after_gemm1):
        raise TypeError('after_gemm1 must be callable')
    if not isinstance(sym_buffer, SymmBuffer):
        raise ValueError('Main MegaMoE requires a SymmBuffer')
    published_inputs = sym_buffer._state.input_keepalive
    if published_inputs is None:
        raise ValueError('Call mega_moe_pre_dispatch before fp4_fp4_mega_moe')
    num_tokens = published_inputs[0].shape[0]
    if not isinstance(y, torch.Tensor) or y.ndim != 2 or y.shape[0] < num_tokens:
        raise ValueError('y must be a 2D tensor with enough rows for the published inputs')
    _C.fp4_fp4_mega_moe(
        y[:num_tokens], l1_weights, l2_weights, cumulative_local_expert_recv_stats, sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts, sym_buffer.num_topk,
        tuple(recipe), activation, activation_clamp, fast_math,
        expected_m, preserve_bf16, fuse_reduce,
        after_gemm1 if after_gemm1 is not None else getattr(sym_buffer, "_stage1_done", None))
