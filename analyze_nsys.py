# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Analyze Nsight Systems SQLite export for PersonaPlex Baseline profiling.

Capture (use the lifecycle e2e driver on main):

  # Terminal 1: request (max_sessions: 2)
  python tests/e2e/online_serving/personaplex_realtime_duplex.py   \\
    --url ws://127.0.0.1:8099/v1/realtime?duplex=1   --model nvidia/personaplex-7b-v1 \\
    --input-wav <input.wav>  --output-dir tmp/personaplex-realtime-duplex

  # Terminal 2: nsys capture
    nsys profile \\
    --trace-fork-before-exec=true \\
    -t cuda,nvtx,osrt \\
    --capture-range=cudaProfilerApi \\
    -o personaplex_main_b2_baseline \\
    --force-overwrite=true \\
    python -m vllm_omni.entrypoints.cli.main serve   "$MODEL" \\
    --omni   --deploy-config vllm_omni/deploy/personaplex.yaml   --host 0.0.0.0   --port 8099

    nsys export --type sqlite --output personaplex_main_b2_baseline.sqlite personaplex_main_b2_baseline.nsys-rep
    python analyze_nsys.py personaplex_main_b2_baseline.sqlite

    A directory of traces (graphs on vs off, one file per N) prints one table.
    Rows are NVTX wall time, not kernel sums. Kernel sums are not comparable
    when the capture used the default --cuda-graph-trace=graph.

    python analyze_nsys.py /tmp/pplex-nsys

    Pair graphs off vs on by filename. A shared case key is what remains after
    removing eager/off or graphs/on, so eager_n16 and graphs_n16 compare.

    The server records ticks 20-120 via torch.cuda.profiler.start/stop in gpu_ar_model_runner.
    Segment kernel counts use launch correlation (runtime correlationId).
"""

import argparse
import sqlite3
import subprocess
from pathlib import Path

import regex as re

ANCHOR_MARKER = "personaplex_sample_and_depformer"
SEGMENT_MARKERS = (
    "personaplex_temporal_forward",
    "personaplex_sample_and_depformer",
)
GRAPH_LAUNCH_NAMES = ("cudaGraphLaunch", "cudaGraphLaunchKernel")


def _table_exists(cursor, table_name):
    cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    return cursor.fetchone() is not None


def _load_runtime_rows(cursor):
    if _table_exists(cursor, "StringIds"):
        cursor.execute(
            """
            SELECT r.start, r.end, r.correlationId, COALESCE(s.value, '')
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            LEFT JOIN StringIds s ON s.id = r.nameId
            ORDER BY r.start ASC
            """
        )
    else:
        cursor.execute("SELECT start, end, correlationId, '' FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start ASC")
    return [(int(start), int(end), int(corr_id), str(name)) for start, end, corr_id, name in cursor.fetchall()]


def _load_kernel_by_correlation(cursor):
    cursor.execute("SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL")
    grouped: dict[int, list[tuple[int, int]]] = {}
    for start, end, corr_id in cursor.fetchall():
        grouped.setdefault(int(corr_id), []).append((int(start), int(end)))
    return grouped


def _load_nvtx_ranges(cursor, segment_markers):
    placeholder = ",".join("?" for _ in segment_markers)
    cursor.execute(
        f"""SELECT text, start, end FROM NVTX_EVENTS WHERE text IN ({placeholder}) ORDER BY start ASC""",
        segment_markers,
    )
    events: dict[str, list[tuple[int, int]]] = {marker: [] for marker in segment_markers}
    for text, start, end in cursor.fetchall():
        events[str(text)].append((int(start), int(end)))
    return events


def _correlated_for_nvtx(
    nvtx_start: int,
    nvtx_end: int,
    runtime_rows: list[tuple[int, int, int, str]],
    kernel_by_corr: dict[int, list[tuple[int, int]]],
) -> tuple[int, int, int]:
    kernels = 0
    launches = 0
    graph_replays = 0
    for runtime_start, _, corr_id, name in runtime_rows:
        if runtime_start < nvtx_start or runtime_start > nvtx_end:
            continue
        launches += 1
        if any(token in name for token in GRAPH_LAUNCH_NAMES):
            graph_replays += 1
        kernels += len(kernel_by_corr.get(corr_id, []))
    return kernels, launches, graph_replays


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _ms(start: int, end: int) -> float:
    return (end - start) / 1e6


def _stage0_step_ms(temporal: list[tuple[int, int]], depformer: list[tuple[int, int]]):
    """Wall time from the start of temporal forward to the end of the depformer range."""
    if not temporal or not depformer or len(temporal) != len(depformer):
        return []
    steps = []
    for (temporal_start, _), (_, depformer_end) in zip(sorted(temporal), sorted(depformer)):
        if depformer_end >= temporal_start:
            steps.append(_ms(temporal_start, depformer_end))
    return steps


def _trace_wall_row(db_path, tick_ms):
    """Per-step Stage 0 wall times from NVTX. Does not use kernel-sum GPU active time."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except sqlite3.Error as exc:
        print(f"{db_path.name}: cannot open ({exc})")
        return None
    if not _table_exists(cursor, "NVTX_EVENTS"):
        print(f"{db_path.name}: no NVTX_EVENTS table")
        conn.close()
        return None
    ranges = _load_nvtx_ranges(cursor, SEGMENT_MARKERS)
    temporal = ranges.get("personaplex_temporal_forward", [])
    depformer = ranges.get(ANCHOR_MARKER, [])
    temporal_ms = [_ms(start, end) for start, end in temporal]
    depformer_ms = [_ms(start, end) for start, end in depformer]
    step_ms = _stage0_step_ms(temporal, depformer)
    replays = None
    if _table_exists(cursor, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        runtime_rows = _load_runtime_rows(cursor)
        kernel_by_corr = {}
        if _table_exists(cursor, "CUPTI_ACTIVITY_KIND_KERNEL"):
            kernel_by_corr = _load_kernel_by_correlation(cursor)
        total_replays = 0
        for start, end in depformer:
            _, _, graph_n = _correlated_for_nvtx(start, end, runtime_rows, kernel_by_corr)
            total_replays += graph_n
        replays = total_replays / len(depformer) if depformer else 0.0
    conn.close()
    step_med = _median(step_ms)
    return {
        "trace": db_path.stem,
        "ticks": len(depformer),
        "temporal_med_ms": _median(temporal_ms),
        "depformer_med_ms": _median(depformer_ms),
        "stage0_step_med_ms": step_med,
        "stage0_step_max_ms": max(step_ms) if step_ms else None,
        "step_over_tick": (step_med / tick_ms) if step_med is not None and tick_ms else None,
        "depformer_replays_per_tick": replays,
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
    paths: list[Path] = []
    for sqlite_path in sorted(directory.rglob("*.sqlite")):
        paths.append(sqlite_path)
    known = {path.with_suffix("") for path in paths}
    for nsys_rep in sorted(directory.rglob("*.nsys-rep")):
        if nsys_rep.with_suffix("") in known:
            continue
        exported = _export_sqlite(nsys_rep)
        if exported is not None:
            paths.append(exported)
    return paths


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


_OFF_TOKENS = {"eager_depformer", "graphs_off", "graph_off", "eager", "off"}
_ON_TOKENS = {"graphs_on", "graph_on", "graphs", "on"}


def _mode_and_key(trace_name) -> tuple[str, str] | None:
    """Split `eager_n16` / `graphs_n16` into a mode and the shared case key."""
    stem = Path(trace_name).name.lower()
    mode = None
    for token in _OFF_TOKENS:
        if token in stem:
            mode = "off"
            stem = stem.replace(token, " ")
            break
    if mode is None:
        for token in _ON_TOKENS:
            if token in stem:
                mode = "on"
                stem = stem.replace(token, " ")
                break
    if mode is None:
        return None
    key = re.sub(r"[^a-z0-9]+", "", stem) or trace_name
    return mode, key


def _case_sort_key(case_key: str) -> tuple[int, str]:
    numbers = re.findall(r"\d+", case_key)
    return (int[numbers[-1]] if numbers else 10**9, case_key)


def _print_table(headers, rendered) -> None:
    widths = [len(header) for header in headers]
    for line in rendered:
        for index, cell in enumerate(line):
            widths[index] = max(widths[index], len(cell))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    for line in rendered:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(line)))


def _print_on_off_table(rows):
    grouped = {}
    for row in rows:
        parsed = _mode_and_key(str(row["trace"]))
        if parsed is None:
            continue
        mode, case_key = parsed
        grouped.setdefault(case_key, {})[mode] = row
    if not grouped:
        print("No graphs-on/off pair.")
        return
    headers = (
        "case",
        "off_step_med_ms",
        "on_step_med_ms",
        "delta_ms",
        "off_step/80",
        "on_step/80",
        "off_depformer_med_ms",
        "on_depformer_med_ms",
        "off_replays/tick",
        "on_replays/tick",
    )
    rendered = []
    for case_key in sorted(grouped, key=_case_sort_key):
        pair = grouped[case_key]
        off = pair.get("off")
        on = pair.get("on")
        off_step = off.get("stage0_step_med_ms") if off else None
        on_step = on.get("stage0_step_med_ms") if on else None
        delta = None
        if isinstance(off_step, float) and isinstance(on_step, float):
            delta = on_step - off_step
        rendered.append(
            [
                case_key,
                _fmt(off_step),
                _fmt(on_step),
                _fmt(delta),
                _fmt(off.get("step_over_tick") if off else None),
                _fmt(on.get("step_over_tick") if on else None),
                _fmt(off.get("depformer_med_ms") if off else None),
                _fmt(on.get("depformer_med_ms") if off else None, 1),
                _fmt(off.get("depformer_replays_per_tick") if off else None, 1),
                _fmt(on.get("depformer_replays_per_tick") if on else None, 1),
            ]
        )
    print("Graph off vs on. delta_ms is on minus off (negative means the graphed depformer step is faster).")
    _print_table(headers, rendered)


def summarize_nsys_dir(directory, tick_ms: float = 80.0) -> None:
    """One row per trace, plus an on/off table when filenames mark the pair."""
    rows = []
    for db_path in _sqlite_paths(directory):
        row = _trace_wall_row(db_path, tick_ms)
        if row is not None:
            row["trace"] = str(db_path.relative_to(directory).with_suffix(""))
            rows.append(row)
    if not rows:
        print(f"No nsys sqlite traces in {directory}")
        return
    print(
        "Stage 0 step is NVTX wall time from personaplex temporal_forward start "
        "to personaplex_sample_and_depformer end."
    )
    _print_on_off_table(rows)
    print()
    headers = (
        "trace",
        "ticks",
        "temporal_med_ms",
        "depformer_med_ms",
        "stage0_step_med_ms",
        "stage0_step_max_ms",
        "step/80",
        "depformer_replays/tick",
    )
    keys = (
        "trace",
        "ticks",
        "temporal_med_ms",
        "depformer_med_ms",
        "stage0_step_med_ms",
        "stage0_step_max_ms",
        "step_over_tick",
        "depformer_replays_per_tick",
    )
    rendered = [[_fmt(row[key], 1 if key == "depformer_replays_per_tick" else 2) for key in keys] for row in rows]
    _print_table(headers, rendered)


def analyze_nsys_sqlite(db_path, tick_ms=80.0):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error opening database: {e}")
        return

    # ==========================================
    # PART 1: Global GPU Idle & Kernel Stats
    # ==========================================
    # Query the start and end times of all kernels (in nanoseconds)
    query_global = """
    SELECT start, end
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    ORDER BY start ASC
    """
    try:
        cursor.execute(query_global)
        kernels = cursor.fetchall()
    except sqlite3.OperationalError:
        print("Error: Could not find CUPTI_ACTIVITY_KIND_KERNEL table. Ensure the trace includes CUDA (-t cuda).")
        conn.close()
        return

    if not kernels:
        print("No kernel data found in the database.")
        conn.close()
        return

    total_kernels = len(kernels)

    # Get the total span of the captured window
    trace_start = kernels[0][0]
    trace_end = max(k[1] for k in kernels)
    trace_duration_ns = trace_end - trace_start
    trace_duration_ms = trace_duration_ns / 1e6

    # Merge overlapping kernel intervals to calculate absolute active GPU time
    active_ns = 0
    current_start = -1
    current_end = -1

    for start, end in kernels:
        if start > current_end:
            # No overlap, accumulate the previous interval
            if current_start != -1:
                active_ns += current_end - current_start
            current_start = start
            current_end = end
        else:
            # Overlap exists, extend the end time of the current interval
            current_end = max(current_end, end)

    # Accumulate the final interval
    if current_start != -1:
        active_ns += current_end - current_start

    active_ms = active_ns / 1e6
    idle_ms = trace_duration_ms - active_ms
    idle_fraction = (idle_ms / trace_duration_ms) * 100

    nvtx_ranges = _load_nvtx_ranges(cursor, SEGMENT_MARKERS)
    observed_anchor_ticks = len(nvtx_ranges.get(ANCHOR_MARKER, []))
    trace_span_over_nominal_ticks = trace_duration_ms / tick_ms if tick_ms else 0.0

    runtime_rows = _load_runtime_rows(cursor)
    kernel_by_corr = _load_kernel_by_correlation(cursor)

    print("=" * 50)
    print(" Nsys Profile Automated Analysis Report")
    print("=" * 50)
    print(f"Total Trace Duration      : {trace_duration_ms:.2f} ms")
    print(f"Observed Anchor Ticks           : {observed_anchor_ticks} ({ANCHOR_MARKER})")
    print(
        f"Trace span / nominal tick : {trace_span_over_nominal_ticks:.1f}"
        f"(metadata only; not observed tick count; nominal={tick_ms} ms)"
    )
    print("-" * 50)
    print(f"Total Kernel records     : {len(kernels)}")
    print("-" * 50)
    print(f"Absolute GPU Active Time  : {active_ms:.2f} ms")
    print(f"Absolute GPU Idle Time    : {idle_ms:.2f} ms")
    print(f"GPU Idle Fraction         : {idle_fraction:.2f}%")
    print("=" * 50)

    # ==========================================
    # PART 2: NVTX Specific Kernel Launch Counts
    # ==========================================
    # Map kernels precisely to the marked NVTX ranges
    print("\n" + "=" * 50)
    print(" NVTX Segment-Specific Kernel Analysis(launch-correlated)")
    print("=" * 50)
    for marker in SEGMENT_MARKERS:
        occurrences = nvtx_ranges.get(marker, [])
        total_kernels = 0
        total_launches = 0
        total_graph_replays = 0
        for start, end in occurrences:
            kernels_n, launches_n, graph_n = _correlated_for_nvtx(start, end, runtime_rows, kernel_by_corr)
            total_kernels += kernels_n
            total_launches += launches_n
            total_graph_replays += graph_n
        observed = len(occurrences)
        avg_kernels = total_kernels / observed if observed else 0.0
        avg_graph = total_graph_replays / observed if observed else 0.0

        print(f"Segment (NVTX)       : {marker}")
        print(f"Observed Occurrences  : {observed} times")
        print(f"Total Kernels Records : {total_kernels}")
        print(f"Total CUDA Runtime calls : {total_launches}")
        print(f"Total Graph Replays  : {total_graph_replays}")
        print(f"Avg Correlated Kernels   : {avg_kernels:.1f} kernels")
        print(f"Avg Graph Replays / Call: {avg_graph:.1f} replays")
        print("-" * 50)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Nsys SQLite export for GPU Idle fraction and NVTX kernel counts."
    )
    parser.add_argument("path", help="One .sqlite file, or a directory of .sqlite / .nsys-rep traces.")
    parser.add_argument("--tick_ms", type=float, default=80.0, help="Duplex tick cycle duration in ms (default: 80.0)")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        summarize_nsys_dir(target, args.tick_ms)
    else:
        analyze_nsys_sqlite(str(target), args.tick_ms)
