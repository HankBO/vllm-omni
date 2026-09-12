# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare PersonaPlex depformer CUDA graphs on vs off from Nsight Systems traces.

Capture one trace per variant and per session count (the server profiles ticks
20-120 via torch.cuda.profiler in gpu_ar_model_runner):

    nsys profile --trace-fork-before-exec=true -t cuda,nvtx,osrt \\
      --capture-range=cudaProfilerApi --cuda-graph-trace=node \\
      --force-overwrite=true -o tmp/pplex-nsys/graphs_n16 \\
      vllm serve nvidia/personaplex-7b-v1 --omni --deploy-config <yaml> --port 8099

Name traces so that removing `eager`/`off` or `graphs`/`on` leaves the same case
key (eager_n16 pairs with graphs_n16). `--cuda-graph-trace=node` is required, or
kernels replayed from a graph are not individually visible.

    python analyze_nsys.py tmp/pplex-nsys --output-file tmp/pplex-nsys/report.md
    python analyze_nsys.py tmp/pplex-nsys/graphs_n16.sqlite

Metrics are computed per `personaplex_depformer` NVTX range (one depformer call
per scheduler step) from the kernels launched inside it, matched by launch
correlation id and by thread:

* GPU wall: first correlated kernel start to last correlated kernel end. NVTX
  alone is host time and cannot be used for this, since launches are async.
* Kernels and launch API calls per call.
* GPU idle fraction: share of the GPU wall window not covered by any kernel.
"""

import argparse
import sqlite3
import subprocess
from pathlib import Path

import regex as re

DEPFORMER_MARKER = "personaplex_depformer"
LAUNCH_API_TOKENS = ("Launch",)


def _table_exists(cursor, table_name):
    cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None


def _columns(cursor, table_name) -> set[str]:
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _load_depformer_ranges(cursor) -> list[tuple[int, int, int | None]]:
    tid_col = "globalTid" if "globalTid" in _columns(cursor, "NVTX_EVENTS") else "NULL"
    cursor.execute(
        f"SELECT start, end, {tid_col} FROM NVTX_EVENTS WHERE text = ? AND end IS NOT NULL ORDER BY start",
        (DEPFORMER_MARKER,),
    )
    return [(int(s), int(e), None if t is None else int(t)) for s, e, t in cursor.fetchall()]


def _load_runtime(cursor) -> list[tuple[int, int, str, int | None]]:
    tid_col = "r.globalTid" if "globalTid" in _columns(cursor, "CUPTI_ACTIVITY_KIND_RUNTIME") else "NULL"
    cursor.execute(
        f"""
        SELECT r.start, r.correlationId, COALESCE(s.value, ''), {tid_col}
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        LEFT JOIN StringIds s ON s.id = r.nameId
        ORDER BY r.start
        """
    )
    return [(int(s), int(c), str(n), None if t is None else int(t)) for s, c, n, t in cursor.fetchall()]


def _load_kernels(cursor) -> dict[int, list[tuple[int, int]]]:
    cursor.execute("SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL")
    grouped: dict[int, list[tuple[int, int]]] = {}
    for start, end, corr in cursor.fetchall():
        grouped.setdefault(int(corr), []).append((int(start), int(end)))
    return grouped


def _busy_ns(intervals: list[tuple[int, int]]) -> int:
    busy = 0
    cur_start = cur_end = None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                busy += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        busy += cur_end - cur_start
    return busy


def _trace_metrics(db_path: Path) -> dict[str, object] | None:
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        print(f"{db_path.name}: cannot open ({exc})")
        return None
    cursor = conn.cursor()
    required = ("NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds")
    missing = [t for t in required if not _table_exists(cursor, t)]
    if missing:
        print(f"{db_path.name}: missing tables {missing}; capture with -t cuda,nvtx")
        conn.close()
        return None
    ranges = _load_depformer_ranges(cursor)
    runtime = _load_runtime(cursor)
    kernels = _load_kernels(cursor)
    conn.close()

    walls, kernel_counts, launch_counts, idles = [], [], [], []
    for start, end, tid in ranges:
        launches = 0
        intervals: list[tuple[int, int]] = []
        for r_start, corr, name, r_tid in runtime:
            if r_start < start:
                continue
            if r_start > end:
                break
            if tid is not None and r_tid is not None and r_tid != tid:
                continue
            if any(token in name for token in LAUNCH_API_TOKENS):
                launches += 1
            intervals.extend(kernels.get(corr, ()))
        if not intervals:
            continue
        span = max(e for _, e in intervals) - min(s for s, _ in intervals)
        walls.append(span / 1e6)
        kernel_counts.append(len(intervals))
        launch_counts.append(launches)
        idles.append(100.0 * (1.0 - _busy_ns(intervals) / span) if span else 0.0)

    return {
        "calls": len(walls),
        "wall_med_ms": _percentile(walls, 0.5),
        "wall_p99_ms": _percentile(walls, 0.99),
        "kernels": _percentile(kernel_counts, 0.5),
        "launches": _percentile(launch_counts, 0.5),
        "idle_pct": _percentile(idles, 0.5),
    }


def _export_sqlite(nsys_rep: Path) -> Path | None:
    sqlite_path = nsys_rep.with_suffix(".sqlite")
    if sqlite_path.is_file():
        return sqlite_path
    try:
        subprocess.run(
            ["nsys", "export", "--type", "sqlite", "--output", str(sqlite_path), str(nsys_rep)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"{nsys_rep.name}: sqlite export failed ({exc})")
        return None
    return sqlite_path if sqlite_path.is_file() else None


def _sqlite_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.rglob("*.sqlite"))
    known = {path.with_suffix("") for path in paths}
    for nsys_rep in sorted(directory.rglob("*.nsys-rep")):
        if nsys_rep.with_suffix("") in known:
            continue
        exported = _export_sqlite(nsys_rep)
        if exported is not None:
            paths.append(exported)
    return paths


_OFF_TOKENS = ("eager_depformer", "graphs_off", "graph_off", "eager", "off")
_ON_TOKENS = ("graphs_on", "graph_on", "graphs", "on")


def _mode_and_key(name: str) -> tuple[str | None, str]:
    """Split `eager_n16` / `graphs_n16` into a mode and the shared case key."""
    stem = Path(name).name.lower()
    for mode, tokens in (("off", _OFF_TOKENS), ("on", _ON_TOKENS)):
        for token in tokens:
            if token in stem:
                return mode, re.sub(r"[^a-z0-9]+", "", stem.replace(token, " "))
    return None, re.sub(r"[^a-z0-9]+", "", stem)


def _case_sort_key(case_key: str) -> tuple[int, str]:
    numbers = re.findall(r"\d+", case_key)
    return (int(numbers[-1]) if numbers else 10**9, case_key)


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _delta(off: object, on: object) -> str:
    if isinstance(off, float) and isinstance(on, float):
        return f"{on - off:+.2f}"
    return "-"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(target: Path) -> str:
    db_paths = _sqlite_paths(target) if target.is_dir() else [target]
    grouped: dict[str, dict[str | None, dict[str, object]]] = {}
    for db_path in db_paths:
        metrics = _trace_metrics(db_path)
        if metrics is None:
            continue
        mode, key = _mode_and_key(db_path.stem)
        case = key if mode else db_path.stem
        grouped.setdefault(case, {})[mode] = metrics
    if not grouped:
        return f"No usable nsys traces in {target}\n"

    headers = [
        "case",
        "GPU wall med ms (off / on / Δ)",
        "GPU wall p99 ms (off / on)",
        "kernels per call (off / on)",
        "launch calls per call (off / on)",
        "GPU idle % (off / on / Δ)",
        "calls (off / on)",
    ]
    rows = []
    for case in sorted(grouped, key=_case_sort_key):
        pair = grouped[case]
        off, on = pair.get("off") or pair.get(None) or {}, pair.get("on") or {}
        rows.append(
            [
                case,
                f"{_fmt(off.get('wall_med_ms'))} / {_fmt(on.get('wall_med_ms'))} / "
                f"{_delta(off.get('wall_med_ms'), on.get('wall_med_ms'))}",
                f"{_fmt(off.get('wall_p99_ms'))} / {_fmt(on.get('wall_p99_ms'))}",
                f"{_fmt(off.get('kernels'), 0)} / {_fmt(on.get('kernels'), 0)}",
                f"{_fmt(off.get('launches'), 0)} / {_fmt(on.get('launches'), 0)}",
                f"{_fmt(off.get('idle_pct'), 1)} / {_fmt(on.get('idle_pct'), 1)} / "
                f"{_delta(off.get('idle_pct'), on.get('idle_pct'))}",
                f"{_fmt(off.get('calls'))} / {_fmt(on.get('calls'))}",
            ]
        )
    note = (
        f"Per `{DEPFORMER_MARKER}` NVTX range, medians over calls. GPU wall is the first to last correlated "
        "kernel; idle is the uncovered share of that window. Δ is on minus off. Profiler timings are "
        "diagnostic only."
    )
    return _md_table(headers, rows) + "\n\n" + note + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="One .sqlite file, or a directory of .sqlite / .nsys-rep traces.")
    parser.add_argument("--output-file", type=Path, help="Also save the Markdown report to this file.")
    args = parser.parse_args()

    report = build_report(args.path)
    print(report, end="")
    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(report, encoding="utf-8")
