import sqlite3
import sys
import argparse

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
                active_ns += (current_end - current_start)
            current_start = start
            current_end = end
        else:
            # Overlap exists, extend the end time of the current interval
            current_end = max(current_end, end)
            
    # Accumulate the final interval
    if current_start != -1:
        active_ns += (current_end - current_start)

    active_ms = active_ns / 1e6
    idle_ms = trace_duration_ms - active_ms
    idle_fraction = (idle_ms / trace_duration_ms) * 100

    estimated_ticks = trace_duration_ms / tick_ms

    print("=" * 50)
    print(" Nsys Profile Automated Analysis Report")
    print("=" * 50)
    print(f"Total Trace Duration      : {trace_duration_ms:.2f} ms")
    print(f"Estimated Ticks           : {estimated_ticks:.1f} (based on {tick_ms}ms/tick)")
    print("-" * 50)
    print(f"Total Kernel Launches     : {total_kernels}")
    print(f"Avg Launches per Tick     : {total_kernels / estimated_ticks:.0f} launches / tick")
    print("-" * 50)
    print(f"Absolute GPU Active Time  : {active_ms:.2f} ms")
    print(f"Absolute GPU Idle Time    : {idle_ms:.2f} ms")
    print(f"GPU Idle Fraction         : {idle_fraction:.2f}%")
    print("=" * 50)

    # ==========================================
    # PART 2: NVTX Specific Kernel Launch Counts
    # ==========================================
    # Map kernels precisely to the marked NVTX ranges
    query_nvtx = """
    SELECT 
        n.text AS nvtx_marker,
        COUNT(DISTINCT n.start) AS num_ticks,
        COUNT(k.start) AS total_kernels,
        CAST(COUNT(k.start) AS FLOAT) / COUNT(DISTINCT n.start) AS avg_kernels_per_tick
    FROM 
        NVTX_EVENTS n
    LEFT JOIN 
        CUPTI_ACTIVITY_KIND_KERNEL k 
        ON k.start >= n.start AND k.end <= n.end
    WHERE 
        LOWER(n.text) LIKE '%temporal_forward%' 
        OR LOWER(n.text) LIKE '%sample_and_depformer%'
        OR LOWER(n.text) LIKE '%depformer%'
    GROUP BY 
        n.text;
    """
    
    try:
        cursor.execute(query_nvtx)
        nvtx_rows = cursor.fetchall()
        
        print("\n" + "=" * 50)
        print(" NVTX Segment-Specific Kernel Analysis")
        print("=" * 50)
        
        if not nvtx_rows:
            print("No matching NVTX markers found. Check range names or trace capture.")
        else:
            for row in nvtx_rows:
                marker_name = row[0]
                num_ticks = row[1]
                segment_kernels = row[2]
                avg_kernels = row[3]
                
                print(f"Segment (NVTX)       : {marker_name}")
                print(f"Sampled Occurrences  : {num_ticks} times")
                print(f"Total Kernels inside : {segment_kernels}")
                print(f"Avg Launches / Call  : {avg_kernels:.1f} launches")
                print("-" * 50)
                
    except sqlite3.OperationalError as e:
        print(f"\nFailed to query NVTX events (table might not exist): {e}")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 Nsys SQLite 导出文件中的 GPU Idle 和 Kernel 发射开销。")
    parser.add_argument("db_path", help="SQLite 数据库文件路径 (如 my_profile.sqlite)")
    parser.add_argument("--tick_ms", type=float, default=80.0, help="全双工的时钟周期(默认80ms)")
    args = parser.parse_args()
    
    analyze_nsys_sqlite(args.db_path, args.tick_ms)