# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run the PersonaPlex paced load driver at several session counts and summarize.

Run against an already started server. One variant per invocation:

    python benchmarks/personaplex/realtime_sweep.py run \\
        --label graphs --depformer-graphs on --mimi-graphs off \\
        --url "ws://127.0.0.1:8001/v1/realtime?duplex=1" \\
        --input-wav tmp/input_assistant.wav --sessions 8 16 24 --repeats 3

Each point writes tmp/pplex_<label>_n<N>[_r<k>]/ with load-result.json and
meta.json. Existing load-result.json files are preserved and cause the sweep to
stop before starting any point. Then build the comparison table:

    python benchmarks/personaplex/realtime_sweep.py summarize --root tmp \\
        --output-file tmp/rtf_report.md

Directories without meta.json (for example created by hand) are read too: the
depformer flag is inferred from `eager`/`graphs` in the name, and the Mimi flag
comes from --default-mimi.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tests" / "e2e" / "online_serving" / "personaplex_realtime_duplex.py"
DIR_RE = re.compile(r"^pplex_(?P<label>.+?)_n(?P<n>\d+)(?:_r(?P<rep>\d+))?$")


def run_sweep(args: argparse.Namespace) -> int:
    output_dirs = []
    for sessions in args.sessions:
        for rep in range(1, args.repeats + 1):
            suffix = f"_r{rep}" if args.repeats > 1 else ""
            output_dirs.append(args.root / f"pplex_{args.label}_n{sessions}{suffix}")
    existing_results = []
    for out_dir in output_dirs:
        result_path = out_dir / "load-result.json"
        if result_path.exists():
            existing_results.append(result_path)
    if existing_results:
        paths = ", ".join(str(path) for path in existing_results)
        raise FileExistsError(f"Refusing to overwrite existing sweep results: {paths}")

    failures = 0
    for sessions in args.sessions:
        for rep in range(1, args.repeats + 1):
            suffix = f"_r{rep}" if args.repeats > 1 else ""
            out_dir = args.root / f"pplex_{args.label}_n{sessions}{suffix}"
            out_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "label": args.label,
                "sessions": sessions,
                "depformer_graphs": args.depformer_graphs,
                "mimi_graphs": args.mimi_graphs,
                "max_client_rtf": args.max_client_rtf,
                "revision": args.server_revision,
                "hardware": args.server_hardware,
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            cmd = [
                sys.executable,
                str(DRIVER),
                "--url",
                args.url,
                "--model",
                args.model,
                "--input-wav",
                str(args.input_wav),
                "--output-dir",
                str(out_dir),
                "--sessions",
                str(sessions),
                "--load-frames",
                str(args.load_frames),
                "--drain-s",
                str(args.drain_s),
            ]
            if args.max_client_rtf is not None:
                cmd += ["--max-client-rtf", str(args.max_client_rtf)]
            if args.server_revision:
                cmd += ["--server-revision", args.server_revision]
            if args.server_hardware:
                cmd += ["--server-hardware", args.server_hardware]
            print(f"\n=== {args.label} sessions={sessions} repeat={rep}/{args.repeats} -> {out_dir}", flush=True)
            if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
                failures += 1
                print(f"driver exited non-zero for {out_dir}", file=sys.stderr)
    return 1 if failures else 0


def _load_point(directory: Path, default_mimi: str) -> dict[str, object] | None:
    match = DIR_RE.match(directory.name)
    result_path = directory / "load-result.json"
    if match is None or not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    label = match["label"]
    depformer = meta.get("depformer_graphs") or ("off" if "eager" in label else "on" if "graph" in label else "?")
    sessions = result.get("sessions", [])

    def values(key: str, kind: type | tuple) -> list:
        return [s[key] for s in sessions if isinstance(s.get(key), kind)]

    rtfs = values("client_stream_rtf", (int, float))
    first = values("client_first_audio_after_stream_start_ms", (int, float))
    deficits = values("frame_deficit", int)
    return {
        "label": label,
        "n": int(match["n"]),
        "depformer": depformer,
        "mimi": meta.get("mimi_graphs") or default_mimi,
        "passed_sessions": int(result.get("passed_sessions", 0)),
        "requested_sessions": int(result.get("requested_sessions", len(sessions))),
        "ok": bool(result.get("ok")),
        "rtf_ceiling": meta.get("max_client_rtf", result.get("max_client_rtf")),
        "rtf_worst": max(rtfs) if rtfs else None,
        "rtf_median": statistics.median(rtfs) if rtfs else None,
        "first_audio_ms": statistics.median(first) if first else None,
        "deficit": max(deficits) if deficits else None,
        "pacing_warn": len(result.get("client_pacing_warning_sessions") or []),
    }


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def summarize(args: argparse.Namespace) -> int:
    points = [p for d in sorted(args.root.glob("pplex_*")) if d.is_dir() and (p := _load_point(d, args.default_mimi))]
    if not points:
        print(f"No load-result.json found under {args.root}", file=sys.stderr)
        return 1
    grouped: dict[tuple, list[dict[str, object]]] = {}
    for p in points:
        grouped.setdefault((p["n"], p["depformer"], p["mimi"], p["label"]), []).append(p)
    headers = [
        "sessions",
        "depformer graph",
        "mimi encoder graph",
        "label",
        "RTF ceiling",
        "result",
        "worst RTF",
        "median RTF",
        "first audio ms (med)",
        "max frame deficit",
        "pacing warns",
        "runs",
    ]
    rows = []
    for key in sorted(grouped, key=lambda k: (k[0], k[1], k[3])):
        runs = grouped[key]

        def agg(name: str, fn):
            vals = [r[name] for r in runs if r[name] is not None]
            return fn(vals) if vals else None

        status = "PASS" if all(r["ok"] for r in runs) else "FAIL"
        passed_total = sum(int(r["passed_sessions"]) for r in runs)
        requested_total = sum(int(r["requested_sessions"]) for r in runs)
        rows.append(
            [
                str(key[0]),
                key[1],
                key[2],
                key[3],
                _fmt(runs[0]["rtf_ceiling"]),
                f"{status} ({passed_total}/{requested_total})",
                _fmt(agg("rtf_worst", max)),
                _fmt(agg("rtf_median", statistics.median)),
                _fmt(agg("first_audio_ms", statistics.median), 0),
                _fmt(agg("deficit", max)),
                str(sum(r["pacing_warn"] for r in runs)),
                str(len(runs)),
            ]
        )
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    note = (
        "\nRTF is the client stream RTF per session (worst over sessions, then over repeats). "
        "A point passes when every session is admitted, voiced and within the frame deficit; "
        "RTF is an acceptance limit only when --max-client-rtf is set.\n"
    )
    report = "\n".join(lines) + "\n" + note
    print(report)
    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(report, encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the load driver for each session count against a live server.")
    run.add_argument("--label", required=True, help="Variant name used in output directories, e.g. eager or graphs.")
    run.add_argument("--depformer-graphs", choices=["on", "off"], required=True)
    run.add_argument("--mimi-graphs", choices=["on", "off"], required=True)
    run.add_argument("--url", default="ws://127.0.0.1:8001/v1/realtime?duplex=1")
    run.add_argument("--model", default="nvidia/personaplex-7b-v1")
    run.add_argument("--input-wav", type=Path, required=True)
    run.add_argument("--sessions", type=int, nargs="+", required=True)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--load-frames", type=int, default=375)
    run.add_argument("--drain-s", type=float, default=5.0)
    run.add_argument("--max-client-rtf", type=float, help="Fail sessions whose client stream RTF exceeds this value.")
    run.add_argument("--root", type=Path, default=Path("tmp"))
    run.add_argument("--server-revision")
    run.add_argument("--server-hardware")
    run.set_defaults(func=run_sweep)

    summ = sub.add_parser("summarize", help="Collect load-result.json files into a Markdown table.")
    summ.add_argument("--root", type=Path, default=Path("tmp"))
    summ.add_argument("--default-mimi", choices=["on", "off"], default="off")
    summ.add_argument("--output-file", type=Path)
    summ.set_defaults(func=summarize)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
