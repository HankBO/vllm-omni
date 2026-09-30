# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Replay real depformer inputs through eager and CUDA-graph code paths

Capture 1 live session, then compare codes from the real bf16 checkpoint.
Exact-size B=1 is the pass/fail. Padded batches(3->4, 12->16) are reported.

Capture (set the env on the server process; Stage 0 worker writes the file)::

    PERSONAPLEX_DEPFORMER_DUMP=tmp/pplex-depformer.pt \\
    vllm serve /path/to/personaplex-7b-v1 --omni \\
  --deploy-config vllm_omni/deploy/personaplex.yaml

Capture refuses to overwrite an existing dump. Remove the old file or choose a
new ``PERSONAPLEX_DEPFORMER_DUMP`` path before capturing another checkpoint.

python tests/e2e/online_serving/personaplex_realtime_duplex.py \\
  --url 'ws://127.0.0.1:8000/v1/realtime?duplex=1' \\
  --model /path/to/personaplex-7b-v1 --input-wav speech.wav \\
  --output-dir tmp/pplex-capture --sessions 1

python personaplex_depformer_code_parity.py \\
  --dump tmp/pplex-depformer.pt \\
  --model /path/to/personaplex-7b-v1

Exist status is 1 only when an exact-size B=1 replay disagrees. A padded
mismatch is printed and does not fail the process.

"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
    PersonaPlexDepformer,
)
from vllm_omni.model_executor.models.personaplex.personaplex_depformer_cudagraph import (
    CUDAGraphDepformerWrapper,
)

from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    HeliumConfig,
    PersonaPlexConfig,
)

_PAD_NOTE = "padded GEMM M differs from eager; cuBLAS may select another kernel and flip codes"
_CAPTURE_SIZES = (1, 4, 16)
_PAD_CASES = ((3, 4), (12, 16))


def _load_depformer_weights(model: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("safetensors is required to load depformer weights") from exc

    files = sorted(model.glob("*.safetensors"))
    if not files:
        raise SystemExit("No safetensors files found in the model directory")
    weights: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(("depformer.", "depformer_in.", "depformer_emb.", "depformer_text_emb", "linears.")):
                    weights[key] = f.get_tensor(key)
    if not weights:
        raise SystemExit(f"no depformer tensors in {model}")
    return weights


def _make_depformer(weights: dict[str, torch.Tensor], device: torch.device) -> PersonaPlexDepformer:
    config = PersonaPlexConfig()
    dep_config = config.depformer_config

    module = PersonaPlexDepformer(
        dep_config,
        temporal_hidden_size=HeliumConfig().hidden_size,
        text_card=config.text_vocab_size,
    )
    module.to(device=device, dtype=torch.bfloat16)
    module.eval()
    loaded = module.load_weights(weights)
    if not loaded:
        raise SystemExit("depformer load_weights accepted no tensors")
    return module


def _b1_rows(frames: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in frames:
        text = frame["text_token"]
        hidden = frame["hidden"]
        if not isinstance(text, torch.Tensor) or not isinstance(hidden, torch.Tensor):
            raise SystemExit("dump frame is missing text_token or hidden")
        batch = int(text.reshape(-1).shape[0])
        hidden = hidden.reshape(batch, 1, -1)
        tokens = frame.get("audio_tokens")
        provided = frame.get("audio_provided")
        for index in range(batch):
            rows.append(
                {
                    "text_token": text.reshape(-1)[index],
                    "hidden": hidden[index],
                    "audio_tokens": None if not isinstance(tokens, torch.Tensor) else tokens.reshape(batch, -1)[index],
                    "audio_provided": None
                    if not isinstance(provided, torch.Tensor)
                    else provided.reshape(batch, -1)[index],
                    "num_steps": int(frame["num_steps"]),
                }
            )
    return rows


def _stack_rows(rows: Sequence[dict[str, object]], device: torch.device) -> dict[str, torch.Tensor | None]:
    text = torch.stack([row["text_token"] for row in rows]).to(device=device, dtype=torch.long)
    hidden = torch.stack([row["hidden"] for row in rows]).to(device=device, dtype=torch.bfloat16)
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(1)
    tokens = rows[0]["audio_tokens"]
    provided = rows[0]["audio_provided"]
    audio_tokens = None
    audio_provided = None
    if isinstance(tokens, torch.Tensor) and isinstance(provided, torch.Tensor):
        audio_tokens = torch.stack([row["audio_tokens"] for row in rows]).to(device=device, dtype=torch.long)
        audio_provided = torch.stack([row["audio_provided"] for row in rows]).to(device=device, dtype=torch.bool)
    return {
        "text_token": text,
        "hidden": hidden,
        "audio_tokens": audio_tokens,
        "audio_provided": audio_provided,
    }


def _codes(
    module: PersonaPlexDepformer | CUDAGraphDepformerWrapper,
    batch: dict[str, torch.Tensor | None],
    *,
    num_steps: int,
    eager: bool,
) -> torch.Tensor:
    kwargs = {
        "audio_tokens": batch["audio_tokens"],
        "audio_provided": batch["audio_provided"],
    }
    if eager:
        out = module(batch["text_token"], batch["hidden"], num_steps=num_steps, **kwargs)
    else:
        out = module(batch["text_token"], batch["hidden"], **kwargs)
    if isinstance(out, tuple):
        out = out[0]
    return out.to("cpu")


def _compare(
    eager: PersonaPlexDepformer,
    wrapper: CUDAGraphDepformerWrapper,
    rows: Sequence[dict[str, object]],
    *,
    actual_b: int,
    padded_b: int,
    num_steps: int,
    device: torch.device,
) -> dict[str, object]:
    batch = _stack_rows(rows[:actual_b], device)
    before = wrapper.stats.replays
    want = _codes(eager, batch, num_steps=num_steps, eager=True)
    got = _codes(wrapper, batch, num_steps=num_steps, eager=False)
    replayed = wrapper.stats.replays - before + 1
    if got.shape[0] != actual_b:
        raise SystemExit(f"graph replay returned {got.shape[0]} rows, expected {actual_b}")
    mismatches = int((want != got).any(dim=-1).sum().item()) if want.numel() else 0
    report: dict[str, object] = {
        "actual_b": actual_b,
        "padded_b": padded_b,
        "bitwise_match": mismatches == 0 and torch.equal(want, got),
        "mismatches": mismatches,
        "graph_replayed": replayed,
    }
    if not replayed:
        report["note"] = "padded graph was not replayed; this row fell back to eager"
    elif padded_b != actual_b and not report["bitwise_match"]:
        report["note"] = _PAD_NOTE
    return report


def replay(dump_path: Path, model: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list) or not frames:
        raise SystemExit(f"No valid frames found in dump at {dump_path}")
    rows = _b1_rows(frames)
    steps = {int(row["num_steps"]) for row in rows}
    if len(steps) != 1:
        raise SystemExit(f"dump mixes num_steps values: {sorted(steps)}")
    num_steps = steps.pop()
    device = torch.device("cuda:0")
    weights = _load_depformer_weights(model)
    eager = _make_depformer(weights=weights, device=device)
    for index, row in enumerate(rows):
        hidden = row["hidden"]
        if hidden.shape[-1] != eager.temporal_hidden_size:
            raise SystemExit(
                f"dump frame {index} hidden width {hidden.shape[-1]} does not match model width "
                f"{eager.temporal_hidden_size}; recapture with PERSONAPLEX_DEPFORMER_DUMP set on the server"
            )
    graphed = _make_depformer(weights=weights, device=device)
    wrapper = CUDAGraphDepformerWrapper(
        graphed,
        capture_sizes=_CAPTURE_SIZES,
        warmup_iters=1,
        num_steps=num_steps,
    )
    wrapper.warmup(device)
    if not wrapper.is_ready:
        raise SystemExit("CUDA graph wrapper is not ready")
    b1_mismatch = 0
    with torch.inference_mode():
        replays_before = wrapper.stats.replays
        for row in rows:
            batch = _stack_rows([row], device)
            want = _codes(eager, batch, num_steps=num_steps, eager=True)
            got = _codes(wrapper, batch, num_steps=num_steps, eager=False)
            if not torch.equal(want, got):
                b1_mismatch += 1
        b1_replays = wrapper.stats.replays - replays_before
        padded = []
        for actual_b, padded_b in _PAD_CASES:
            if len(rows) < actual_b:
                padded.append(
                    {
                        "actual_b": actual_b,
                        "padded_b": padded_b,
                        "skipped": f"need {actual_b} B=1 frames, got {len(rows)}",
                    }
                )
                continue
            padded.append(
                _compare(eager, wrapper, rows, actual_b=actual_b, padded_b=padded_b, num_steps=num_steps, device=device)
            )
    return {
        "dump": str(dump_path),
        "model": str(model),
        "frames": len(rows),
        "num_steps": num_steps,
        "dtype": "bfloat16",
        "b1": {
            "padded_b": 1,
            "bitwise_match": b1_mismatch == 0 and b1_replays == len(rows),
            "mismatches": b1_mismatch,
            "frames": len(rows),
            "graph_replays": b1_replays,
        },
        "padded": padded,
        "graph_ready": wrapper.is_ready,
        "capture_failures": wrapper.stats.capture_failure,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True, help="PERSONAPLEX_DEPFORMER_DUMP .pt from one session")
    parser.add_argument("--model", type=Path, required=True, help="Path to local PPlex checkpoint")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = replay(args.dump, args.model)
    print(json.dumps(report, indent=2))
    if not report["b1"]["bitwise_match"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
