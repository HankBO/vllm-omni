# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU wall time of the PersonaPlex depformer, eager vs CUDA graph, via CUDA events.

Builds the depformer with the PersonaPlex-7B shapes and random weights, so no
checkpoint is needed. For each batch size (one row per live duplex session) it
times the same call the Stage 0 talker makes, once eager and once through
``CUDAGraphDepformerWrapper``, with ``torch.cuda.Event`` pairs recorded around
the call. The GPU is drained before every call, as at the start of a serving
step, so the interval includes the idle gaps a launch-bound eager call leaves.

    python benchmarks/personaplex/depformer_cuda_event_bench.py \\
        --batch-sizes 1 8 16 24 32 48 64 --output-file tmp/depformer_events.md
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from vllm_omni.model_executor.models.personaplex.configuration_personaplex import PersonaPlexDepformerConfig
from vllm_omni.model_executor.models.personaplex.personaplex_depformer import PersonaPlexDepformer
from vllm_omni.model_executor.models.personaplex.personaplex_depformer_cudagraph import CUDAGraphDepformerWrapper
from vllm_omni.platforms import current_omni_platform


def _inputs(batch: int, config: PersonaPlexDepformerConfig, temporal: int, text_card: int, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(batch)
    text = torch.randint(0, text_card, (batch,), generator=gen).to(device)
    hidden = torch.randn(batch, 1, temporal, generator=gen).to(device=device, dtype=torch.bfloat16)
    tokens = torch.randint(0, config.card, (batch, config.dep_q), generator=gen).to(device)
    provided = torch.zeros(batch, config.dep_q, dtype=torch.bool, device=device)
    provided[:, 0] = True
    return text, hidden, tokens, provided


def _time_gpu_ms(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    current_omni_platform.synchronize()
    pairs = []
    for _ in range(iters):
        current_omni_platform.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        pairs.append((start, end))
    current_omni_platform.synchronize()
    return [s.elapsed_time(e) for s, e in pairs]


def _p99(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.99 * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 24, 32, 48, 64])
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--num-steps", type=int, default=8, help="Active codebooks served (default 8).")
    parser.add_argument("--temporal-hidden-size", type=int, default=4096)
    parser.add_argument("--text-card", type=int, default=32000)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda:0")
    config = PersonaPlexDepformerConfig(num_active_codebooks=args.num_steps)
    torch.manual_seed(0)
    model = PersonaPlexDepformer(config, temporal_hidden_size=args.temporal_hidden_size, text_card=args.text_card)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    sizes = sorted(set(args.batch_sizes))
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=sizes, warmup_iters=3, num_steps=args.num_steps)
    with torch.inference_mode():
        wrapper.warmup(device)
    if not wrapper.is_ready or wrapper.stats.capture_failure:
        raise SystemExit(f"graph capture failed: {wrapper.stats_snapshot()}")

    rows = []
    with torch.inference_mode():
        for batch in sizes:
            text, hidden, tokens, provided = _inputs(batch, config, args.temporal_hidden_size, args.text_card, device)

            def eager():
                return model(text, hidden, audio_tokens=tokens, audio_provided=provided, num_steps=args.num_steps)

            def graphed():
                return wrapper(text, hidden, audio_tokens=tokens, audio_provided=provided)

            torch.testing.assert_close(graphed(), eager(), rtol=0, atol=0)
            off = _time_gpu_ms(eager, args.iters, args.warmup)
            replays_before = wrapper.stats.replays
            on = _time_gpu_ms(graphed, args.iters, args.warmup)
            if wrapper.stats.replays - replays_before != args.iters + args.warmup:
                raise SystemExit(f"batch {batch}: graph path fell back to eager: {wrapper.stats_snapshot()}")
            off_med, on_med = statistics.median(off), statistics.median(on)
            rows.append(
                [
                    str(batch),
                    f"{off_med:.3f}",
                    f"{on_med:.3f}",
                    f"{on_med - off_med:+.3f}",
                    f"{off_med / on_med:.2f}x",
                    f"{_p99(off):.3f}",
                    f"{_p99(on):.3f}",
                ]
            )

    headers = [
        "sessions (batch)",
        "GPU wall off med ms",
        "GPU wall on med ms",
        "Δ ms",
        "speedup",
        "off p99 ms",
        "on p99 ms",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    note = (
        f"\nDepformer only, {args.num_steps} steps, bf16, random weights, {torch.cuda.get_device_name(device)}, "
        f"{args.iters} timed calls per cell, CUDA events, GPU drained before each call. "
        "Graph output is bitwise equal to eager. Δ is on minus off.\n"
    )
    report = "\n".join(lines) + "\n" + note
    print(report)
    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
