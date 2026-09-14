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

    The server records ticks 20-120 via torch.cuda.profiler.start/stop in gpu_ar_model_runner.
    Segment kernel counts use launch correlation (runtime correlationId).
"""

import argparse
import sqlite3

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
    parser.add_argument("db_path", help="Path to the SQLite database file (e.g., my_profile.sqlite)")
    parser.add_argument("--tick_ms", type=float, default=80.0, help="Duplex tick cycle duration in ms (default: 80.0)")
    args = parser.parse_args()

    analyze_nsys_sqlite(args.db_path, args.tick_ms)
