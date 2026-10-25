"""Flash EP8 streaming regression; run with torchrun --nproc_per_node=8.

Compares all BF16 bytes to the original two-stage path while graph inputs,
routes and active masks change. Includes an empty sender and empty owners.
"""

import datetime
import os

import torch
import torch.distributed as dist

import deep_gemm


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(seconds=180),
        device_id=torch.device("cuda", rank),
    )
    assert dist.get_world_size() == 8
    torch.manual_seed(6000 + rank)
    group = dist.group.WORLD
    capacity, hidden, intermediate = 32, 4096, 2048

    def weights(n, k):
        return (
            torch.randint(0, 256, (32, n, k // 2), device="cuda", dtype=torch.uint8),
            torch.full((32, n, k // 32), 121, device="cuda", dtype=torch.uint8),
        )

    w1, w2 = deep_gemm.transform_weights_for_mega_moe(
        weights(2 * intermediate, hidden), weights(hidden, intermediate)
    )
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, 256, capacity, 8, hidden, intermediate
    )
    checks = 0
    for rows in (0 if rank == 7 else 8, 32 if rank % 2 == 0 else 24):
        x = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
        ids = torch.full((rows, 8), -1, device="cuda", dtype=torch.int64)
        route_weights = torch.zeros((rows, 8), device="cuda", dtype=torch.float32)
        active = torch.ones(rows, device="cuda", dtype=torch.bool)
        actual, expected = torch.empty_like(x), torch.empty_like(x)
        entry = torch.zeros(1, device="cuda")

        def publish():
            deep_gemm.mega_moe_pre_dispatch(
                x, ids, route_weights, sym_buffer=buffer, active_token_mask=active
            )

        def baseline():
            publish()
            deep_gemm.fp4_fp4_mega_moe(
                expected, w1, w2, buffer, activation_clamp=10.0, preserve_bf16=True
            )

        def candidate(kblock):
            dist.all_reduce(entry)
            publish()
            deep_gemm.fp4_fp4_mega_moe_streamed(
                actual, w1, w2, buffer, activation_clamp=10.0, kblock=kblock
            )

        for kblock in (False, True):
            for _ in range(3):
                candidate(kblock)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                candidate(kblock)
            for repeat in range(24):
                x.normal_().mul_((0.0, 0.001, 1.0, 10.0)[repeat % 4])
                selected = torch.randn((rows, 256), device="cuda").topk(6).indices
                if repeat % 3 == 0:
                    selected[:] = torch.arange(6, device="cuda")
                ids[:, :6].copy_(selected)
                route_weights[:, :6].copy_(torch.rand((rows, 6), device="cuda"))
                active.copy_(torch.rand(rows, device="cuda") > 0.25)
                if repeat % 7 == 0:
                    active.zero_()
                baseline()
                graph.replay()
                assert torch.isfinite(actual).all()
                assert torch.equal(
                    actual.view(torch.uint8), expected.view(torch.uint8)
                ), (rank, rows, kblock, repeat)
                checks += 1
            del graph
        # The explicit callback is captured once; replay only runs device work.
        marker = torch.zeros(1, device="cuda")
        callback_calls = []

        def callback():
            callback_calls.append(1)
            marker.add_(1)

        def callback_candidate():
            publish()
            deep_gemm.fp4_fp4_mega_moe(
                actual,
                w1,
                w2,
                buffer,
                activation_clamp=10.0,
                preserve_bf16=True,
                after_gemm1=callback,
            )

        callback_candidate()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            callback_candidate()
        baseline()
        before = len(callback_calls)
        for _ in range(5):
            graph.replay()
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert len(callback_calls) == before
        assert marker.item() == 6
        del graph
    torch.cuda.synchronize()
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()
    print(f"PASS rank={rank} changing_graph_checks={checks}", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
