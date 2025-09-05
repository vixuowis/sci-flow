# streamlit_obs_sim.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import deque
from datetime import datetime

def _init_state(series_names, max_points, init_vals):
    """Initialize session_state containers (deques) if absent or when max_points changes."""
    if 'max_points' not in st.session_state or st.session_state.max_points != max_points:
        # create new deques with given maxlen
        st.session_state.max_points = max_points
        st.session_state.timestamps = deque(maxlen=max_points)
        st.session_state.deques = {}
        for k, v in init_vals.items():
            # start each deque with one initial value
            st.session_state.deques[k] = deque([v], maxlen=max_points)
        st.session_state.timestamps.append(pd.Timestamp.now())

    if 'running' not in st.session_state:
        st.session_state.running = False

def _reset_state(init_vals, max_points):
    st.session_state.max_points = max_points
    st.session_state.timestamps = deque([pd.Timestamp.now()], maxlen=max_points)
    st.session_state.deques = {k: deque([v], maxlen=max_points) for k, v in init_vals.items()}
    st.session_state.running = False

def run_scene3():
    st.title("无人机直播流传输流量监控")

    # --- Sidebar controls ---
    st.sidebar.header("Simulation controls")
    interval = st.sidebar.slider("更新间隔 (秒)", min_value=0.1, max_value=5.0, value=0.5, step=0.1,
                                 help="每隔多少秒生成一个新数据点")
    max_points = st.sidebar.number_input("最大保存点数 (超过则丢弃最早数据)", min_value=20, max_value=20000, value=300, step=10)
    st.sidebar.markdown("**启动 / 停止 / 重置**")
    start_btn = st.sidebar.button("Start")
    stop_btn = st.sidebar.button("Stop")
    reset_btn = st.sidebar.button("Reset")

    st.sidebar.markdown("---")
    st.sidebar.caption("提示：关闭浏览器标签或停止脚本可终止持续运行。")

    # --- Define simulated series and initial values ---
    # We'll simulate:
    # - throughput_MB_s: instant throughput in MB/s (positive, fluctuating)
    # - packets_k_per_s: packets per second in thousands (derived from throughput)
    # - active_streams: how many active streams (small integer, fluctuating)
    # - total_bytes_GB: cumulative total bytes transferred in GB (monotonic)
    # - total_files: cumulative file count (monotonic)
    init_vals = {
        "throughput_MB_s": 5.0,      # start at 5 MB/s
        "packets_k_per_s": 0.0,      # will be computed from throughput
        "active_streams": 2.0,       # ~2 streams
        "total_bytes_GB": 0.0,       # cumulative
        "total_files": 0.0,          # cumulative files transferred
    }

    # noise / behaviour parameters
    params = {
        "throughput_sigma": 0.6,       # volatility of throughput
        "active_streams_sigma": 0.25,  # volatility of stream count
        "file_creation_prob_per_tick": 0.06,  # probability a new file appears each tick
        "avg_file_size_MB": 12.0,      # average size of a new file (MB) if created
        "avg_packet_size_bytes": 1200  # used to derive packet rate
    }

    # initialize session state containers
    _init_state(list(init_vals.keys()), int(max_points), init_vals)

    # handle control buttons
    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
    if reset_btn:
        _reset_state(init_vals, int(max_points))

    # If user changed max_points via UI, ensure deques' maxlen matches new value
    if st.session_state.max_points != int(max_points):
        # preserve existing data but re-create deques with new maxlen
        old_ts = list(st.session_state.timestamps)
        old_data = {k: list(v) for k, v in st.session_state.deques.items()}
        _reset_state(init_vals, int(max_points))
        # re-fill as much as fits
        for i, t in enumerate(old_ts[-int(max_points):]):
            st.session_state.timestamps.append(pd.Timestamp(t))
        for k in old_data:
            vals = old_data[k][-int(max_points):]
            st.session_state.deques[k].clear()
            for v in vals:
                st.session_state.deques[k].append(v)

    # Layout: two charts + latest metrics
    col1, col2 = st.columns([3, 1])
    chart_container = col1.empty()
    stats_container = col2.empty()
    # cumulative_container = st.expander("累计指标（历史）", expanded=False)
    cumulative_placeholder = st.empty()
    # helper to build DataFrame from deques
    def build_df(keys):
        ts = list(st.session_state.timestamps)
        data = {k: list(st.session_state.deques[k]) for k in keys}
        if len(ts) == 0:
            return pd.DataFrame(data)
        df = pd.DataFrame(data, index=pd.to_datetime(ts))
        return df

    # Main streaming loop (runs while st.session_state.running is True)
    # NOTE: in Streamlit the loop will block the script run; stop button may not interrupt immediately depending on environment.
    try:
        while st.session_state.running:
            # last values
            last_throughput = st.session_state.deques["throughput_MB_s"][-1]
            last_active = st.session_state.deques["active_streams"][-1]
            last_total_bytes_gb = st.session_state.deques["total_bytes_GB"][-1]
            last_total_files = st.session_state.deques["total_files"][-1]

            # --- simulate throughput (MB/s) with random walk, keep >= 0.1 and reasonable cap ---
            new_throughput = float(last_throughput + np.random.normal(loc=0.0, scale=params["throughput_sigma"]))
            new_throughput = max(0.05, min(new_throughput, 200.0))  # clamp between 0.05 and 200 MB/s

            # --- packets per second derived from throughput (with small noise) ---
            bytes_per_s = new_throughput * 1_000_000.0  # MB/s -> bytes/s
            packets_per_s = bytes_per_s / float(params["avg_packet_size_bytes"])
            packets_per_s = packets_per_s * (1.0 + np.random.normal(0.0, 0.02))  # small multiplicative noise
            packets_k_per_s = packets_per_s / 1000.0  # convert to kpackets/s for nicer scale

            # --- active streams (slow-changing) ---
            new_active = float(last_active + np.random.normal(loc=0.0, scale=params["active_streams_sigma"]))
            new_active = max(0.0, min(new_active, 20.0))

            # --- cumulative bytes (GB) increased by throughput * interval ---
            # delta_GB = throughput_MB_s * interval_seconds / 1000
            delta_gb = new_throughput * float(interval) / 1000.0
            new_total_bytes_gb = float(last_total_bytes_gb + delta_gb)

            # --- file arrivals (random) ---
            new_total_files = float(last_total_files)
            if np.random.rand() < params["file_creation_prob_per_tick"]:
                # one new file appears; file size drawn from exponential-ish distribution around avg_file_size_MB
                size_mb = max(0.1, np.random.exponential(scale=params["avg_file_size_MB"]))
                # add its bytes to cumulative as well (already accounted by throughput, but we simulate file count separately)
                new_total_files += 1.0
                # optional: bump total bytes as if instantaneous file transfer happened
                # (we'll not double count—keep cumulative from throughput for realism)

            # --- append to deques (they auto-drop old entries by maxlen) ---
            now = pd.Timestamp.now()
            st.session_state.timestamps.append(now)
            st.session_state.deques["throughput_MB_s"].append(new_throughput)
            st.session_state.deques["packets_k_per_s"].append(packets_k_per_s)
            st.session_state.deques["active_streams"].append(new_active)
            st.session_state.deques["total_bytes_GB"].append(new_total_bytes_gb)
            st.session_state.deques["total_files"].append(new_total_files)

            # --- build DataFrames for plotting ---
            df_rates = build_df(["throughput_MB_s", "packets_k_per_s", "active_streams"])
            df_cum = build_df(["total_bytes_GB", "total_files"])

            # --- redraw charts ---
            with chart_container.container():
                st.subheader("瞬时指标（最近 {} 点）".format(st.session_state.max_points))
                st.line_chart(df_rates)

            with cumulative_placeholder.container():
                st.subheader("累计指标（最近 {} 点）".format(st.session_state.max_points))
                st.line_chart(df_cum)

            # --- show latest small stats on the right column ---
            latest_stats = {
                "Throughput (MB/s)": f"{new_throughput:.2f}",
                "Packets (k/s)": f"{packets_k_per_s:.2f}",
                "Active streams": f"{new_active:.1f}",
                "Total bytes (GB)": f"{new_total_bytes_gb:.4f}",
                "Total files": f"{int(new_total_files)}",
            }
            # using a simple table-like display
            stats_container.write("### 最新值")
            for k, v in latest_stats.items():
                stats_container.write(f"- **{k}**: {v}")

            # sleep for specified interval
            time.sleep(float(interval))

            # Loop will continue until st.session_state.running becomes False.
            # Note: Due to how Streamlit handles requests, stopping may not be instantaneous in some environments.
            # When the user clicks "Stop", the script will be rerun and session_state.running will be False on next run.
            # We still check the flag every tick above.

        # When not running (initially or after stop), show a static snapshot
        if not st.session_state.running:
            df_rates = build_df(["throughput_MB_s", "packets_k_per_s", "active_streams"])
            df_cum = build_df(["total_bytes_GB", "total_files"])
            with chart_container.container():
                st.subheader("瞬时指标（暂停）")
                st.line_chart(df_rates)
            with cumulative_placeholder.container():
                st.subheader("累计指标（暂停）")
                st.line_chart(df_cum)
            # display last values
            last_vals = {k: (st.session_state.deques[k][-1] if len(st.session_state.deques[k])>0 else None) for k in init_vals.keys()}
            stats_container.write("### 当前快照")
            for k, v in last_vals.items():
                if v is None:
                    stats_container.write(f"- **{k}**: -")
                else:
                    if isinstance(v, float):
                        stats_container.write(f"- **{k}**: {v:.4f}")
                    else:
                        stats_container.write(f"- **{k}**: {v}")

    except Exception as e:
        st.error(f"模拟运行时发生异常: {e}")
        st.session_state.running = False

if __name__ == "__main__":
    run_scene3()
