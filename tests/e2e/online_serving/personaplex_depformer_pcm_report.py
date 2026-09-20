"""Manual PCM report: graphed vs eager PersonaPlex depformer.

Not a bit-identical gate.

Eager first: delete Stage 0 ``hf_overrides``; then stock ``personaplex.yaml``.
Then either pass ``--eager-url`` / ``--graphs-url`` or 
compare directories from earlier driver runs.

Example request::
    python tests/e2e/online_serving/personaplex_depformer_pcm_report.py \\
        --model nvidia/personaplex-7b-v1 --input-wav speech.wav \\
        --eager-url ws://127.0.0.1:8000/v1/realtime?duplex=1 \\
        --graphs-url ws://127.0.0.1:8001/v1/realtime?duplex=1 \\
        --output-dir /tmp/pplex-pcm-report

Example starting the eager server::
python -m vllm_omni.entrypoints.cli.main serve \
  /path/to/personaplex-7b-v1 \
  --omni \
  --deploy-config /tmp/personaplex-eager.yaml \
  --port 8000

Example starting the graphs server::
python -m vllm_omni.entrypoints.cli.main serve \
  /path/to/personaplex-7b-v1 \
  --omni \
  --deploy-config /tmp/personaplex-graphs.yaml \
  --port 8001
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

try:
    from tests.e2e.online_serving.personaplex_realtime_duplex import (
        SAMPLE_RATE_HZ,
        _read_wav_as_float32,
        parse_args as parse_driver_args,
        run as run_driver,
    )
except ImportError:
    from personaplex_realtime_duplex import (
        SAMPLE_RATE_HZ,
        _read_wav_as_float32,
        parse_args as parse_driver_args,
        run as run_driver,
    )

WAV_NAME = "primary-output.wav"

def _pcm_metrics(eager: np.ndarray, graphed: np.ndarray) -> dict[str, object]:
    eager_n = int(eager.size)
    graphed_n = int(graphed.size)
    n = min(eager_n, graphed_n)
    if n == 0:
        raise ValueError("one or both WAVs are empty")
    left = eager[:n].astype(np.float64, copy=False)
    right = graphed[:n].astype(np.float64, copy=False)
    diff = left - right
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        cosine = float("nan")
    else:
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "eager_samples": eager_n,
        "graphs_samples": graphed_n,
        "compared_samples": n,
        "length_delta_samples": abs(eager_n - graphed_n),
        "cosine": cosine,
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rms_diff": float(np.sqrt(np.mean(diff ** 2))),
        "byte_identical": eager_n == graphed_n and bool(np.array_equal(eager, graphed)),
        "eager_peak": float(np.max(np.abs(left))) if eager_n else 0.0,
        "graphs_peak": float(np.max(np.abs(right))) if graphed_n else 0.0,
    }

def _driver_argv(
    *,
    url: str,
    model: str,
    input_wav: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        "--url",
        url,
        "--model",
        model,
        "--input-wav",
        str(input_wav),
        "--output-dir",
        str(output_dir),
        "--single-session",
        "--voice",
        args.voice,
        "--persona",
        args.persona,
        "--tail-s",
        str(args.tail_s),
        "--drain-s",
        str(args.drain_s),
        "--timeout-s",
        str(args.timeout_s),
        "--min-voiced-frames",
        str(args.min_voiced_frames),
    ]

def compare_dirs(eager_dir: Path, graphs_dir: Path, *, wav_name: str = WAV_NAME) -> dict[str, dict[str, object]]:
    eager_wav = eager_dir / wav_name
    graphs_wav = graphs_dir / wav_name
    if not eager_wav.is_file():
        raise FileNotFoundError(f"missing eager WAV: {eager_wav}")
    if not graphs_wav.is_file():
        raise FileNotFoundError(f"missing graphs WAV: {graphs_wav}")
    metrics = _pcm_metrics(_read_wav_as_float32(eager_wav), _read_wav_as_float32(graphs_wav))
    return {
        "eager_wav": str(eager_wav.resolve()),
        "graphs_wav": str(graphs_wav.resolve()),
        **metrics,
    }

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-dir", type=Path, help="Directory with eager-depformer driver output.")
    parser.add_argument("--graphs-dir", type=Path, help="Directory with graphs-depformer driver output.")
    parser.add_argument("--eager-url", help="Realtime WS URL for the eager-depformer server.")
    parser.add_argument("--graphs-url", help="Realtime WS URL for the graphs-depformer server.")
    parser.add_argument("--model", help="Required when driving live servers.")
    parser.add_argument("--input-wav", type=Path, help="Input WAV file for the live server.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/personaplex-depformer-pcm-report"),
        help="Parent directory for live driver runs (eager/ and graphs/ subdirs)",
    )
    parser.add_argument("--wav-name", default=WAV_NAME)
    parser.add_argument("--report-json", type=Path, help="Write the report JSON here (default: stdout only).")
    parser.add_argument("--voice", default="NATF2.pt")
    parser.add_argument("--persona", default="You are a concise and helpful assistant.")
    parser.add_argument("--tail-s", type=float, default=0.4)
    parser.add_argument("--drain-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--min-voiced-frames", type=int, default=5)
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    eager_dir = args.eager_dir
    graphs_dir = args.graphs_dir

    live = args.eager_url is not None or args.graphs_url is not None
    if live:
        if not args.eager_url or not args.graphs_url:
            raise SystemExit("Both --eager-url and --graphs-url must be provided for live mode.")
        if not args.model or args.input_wav is None:
            raise SystemExit("--model and --input-wav are required when driving live servers")
        eager_dir = args.output_dir / "eager"
        graphs_dir = args.output_dir / "graphs"
        eager_result = asyncio.run(
            run_driver(
                parse_driver_args(
                    _driver_argv(
                        url=args.eager_url,
                        model=args.model,
                        input_wav=args.input_wav,
                        output_dir=eager_dir,
                        args=args,
                    )
                )
            )
        )
        graphs_result = asyncio.run(
            run_driver(
                parse_driver_args(
                    _driver_argv(
                        url=args.graphs_url,
                        model=args.model,
                        input_wav=args.input_wav,
                        output_dir=graphs_dir,
                        args=args,
                    )
                )
            )
        )
    else:
        if eager_dir is None or graphs_dir is None:
            raise SystemExit("Both --eager-dir and --graphs-dir must be provided for non-live mode.")
        eager_result = None
        graphs_result = None

    report = {
        "note": (
            "PCM similarity of one Realtime session, graphed depformer vs eager "
            "depformer. Not a byte-identical gate."
        ),
        "metrics": compare_dirs(eager_dir, graphs_dir, wav_name=args.wav_name),
        "eager_driver": eager_result,
        "graphs_driver": graphs_result,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text, encoding="utf-8")
    elif live:
        path = args.output_dir / "pcm-report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
    
