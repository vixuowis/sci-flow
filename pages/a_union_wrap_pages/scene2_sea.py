# app.py
"""
Streamlit app: Download insitu_subset.parquet from OBS public URL,
and visualize the selected variable as 3D bars on the map by minute.
"""

import io
import math
import requests
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---------- Configuration (modifiable in sidebar) ----------
DEFAULT_PQ_URL = "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/%E6%B5%B7%E6%B4%8B%E7%89%A7%E5%9C%BA-%E6%96%87%E6%9C%AC%2B%E6%97%B6%E5%BA%8F/insitu_subset.parquet"

# ---------- Helper functions ----------
@st.cache_data(ttl=3600)
def load_parquet_from_url(url: str) -> pd.DataFrame:
    """Download parquet file from URL and return as pandas.DataFrame (exceptions will be shown on the page)."""
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    bio = io.BytesIO(r.content)
    # Use pandas directly, depends on pyarrow or fastparquet
    df = pd.read_parquet(bio)
    return df

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and add columns for easier visualization."""
    df = df.copy()
    # time -> datetime (UTC)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    # Tolerant handling of longitude/latitude column names (your data has longitude / latitude)
    if "longitude" not in df.columns or "latitude" not in df.columns:
        raise KeyError("Missing 'longitude' or 'latitude' column.")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    # value -> numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # Small-sample filtering: drop rows with NaN in lon/lat, NaT in time, or NaN in value
    df = df.dropna(subset=["longitude", "latitude", "time", "value"])
    # Align by minute for frame selection
    df["minute"] = df["time"].dt.floor("min")
    return df

def aggregate_for_minute(df_var: pd.DataFrame, minute_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Aggregate data for the given minute_ts (timezone-aware pandas.Timestamp) by platform_id + lon/lat:
    Returns a DataFrame with columns: platform_id, longitude, latitude, value (mean), count, depth (mean).
    """
    dfm = df_var[df_var["minute"] == minute_ts]
    if dfm.empty:
        return pd.DataFrame(columns=["platform_id","longitude","latitude","value","count","depth"])
    grp = dfm.groupby(["platform_id", "longitude", "latitude"], as_index=False).agg(
        value=("value", "mean"),
        count=("value", "count"),
        depth=("depth", "mean")
    )
    return grp

def compute_height_and_color(df_plot: pd.DataFrame, v_low=None, v_high=None, target_p99_height=1000.0):
    """
    Compute bar height (meters) and RGBA color column (list).
    Mapping strategy:
     - Linearly map value in v_low..v_high to 0..target_p99_height (v_high defaults to 99% quantile)
     - Color interpolated linearly from blue (low) to red (high)
    """
    if df_plot.empty:
        return df_plot
    vals = df_plot["value"].astype(float)
    if v_low is None:
        v_low = float(vals.quantile(0.01))
    if v_high is None:
        v_high = float(vals.quantile(0.99))
    # Prevent v_high == v_low
    if v_high <= v_low:
        v_high = v_low + 1.0
    # Normalize and compute height
    t = ((vals - v_low) / (v_high - v_low)).clip(0.0, 1.0)
    df_plot["height"] = (t * target_p99_height).astype(float)
    # Linear interpolation of color (blue->red)
    def lerp(u):
        r = int(255 * u)
        g = int(50 + 150 * (1 - abs(u - 0.5) * 2))  # keep a bit of green for mid values
        b = int(255 * (1 - u))
        return [r, g, b, 180]
    df_plot["color"] = [lerp(u) for u in t]
    return df_plot
def run_scene2():
  # ---------- Page layout ----------
  st.set_page_config(page_title="INSITU Parquet 变量地图柱状图", layout="wide")
  st.title("📦 INSITU Parquet → 按分钟 3D 柱状变量展示")

  # Sidebar: URL / load settings / filtering
  st.sidebar.header("数据源与筛选")
  pq_url = st.sidebar.text_input("Parquet 文件 URL（OBS）", value=DEFAULT_PQ_URL)
  load_btn = st.sidebar.button("重新加载数据")

  # Try loading (cached)
  try:
      with st.spinner("从 URL 加载 Parquet 文件..."):
          df_raw = load_parquet_from_url(pq_url)
  except Exception as e:
      st.error(f"加载 parquet 失败：{e}")
      st.stop()

  # Show basic info
  st.sidebar.markdown(f"**数据大小:** {df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")
  st.expander("查看原始表头与前 5 行", expanded=False).write(df_raw.head())

  # Preprocessing
  try:
      df = prepare_dataframe(df_raw)
  except Exception as e:
      st.error(f"预处理失败：{e}")
      st.stop()

  # Variable selection
  vars_available = sorted(df["variable"].dropna().unique().astype(str))
  if not vars_available:
      st.error("数据中未找到可用的 'variable' 值。")
      st.stop()

  variable = st.sidebar.selectbox("选择 variable", vars_available, index=0)
  df_var = df[df["variable"].astype(str) == variable].copy()
  st.sidebar.markdown(f"**选中变量行数:** {len(df_var)}")

  # value_qc filter (if exists)
  if "value_qc" in df_var.columns:
      qcs = sorted(df_var["value_qc"].dropna().unique().tolist())
      qc_sel = st.sidebar.multiselect("value_qc（过滤）", qcs, default=qcs)
      if qc_sel:
          df_var = df_var[df_var["value_qc"].isin(qc_sel)]
  else:
      st.sidebar.caption("数据中无 value_qc 列，跳过 QC 过滤。")

  # Time frame selection (by minute)
  minutes_sorted = sorted(df_var["minute"].dropna().unique())
  if not minutes_sorted:
      st.warning("所选变量在数据中没有可用时间点。")
      st.stop()

  # Select time frame index in sidebar (shown as readable time)
  min_idx = 0
  max_idx = len(minutes_sorted) - 1
  frame_idx = st.sidebar.slider(
      "选择时间帧（按分钟）",
      min_value=min_idx,
      max_value=max_idx,
      value=min_idx,
      step=1,
      format="%d"  # slider internally still returns index
  )
  selected_minute = pd.to_datetime(minutes_sorted[frame_idx])
  st.sidebar.markdown(f"**当前分钟：** {selected_minute} (UTC)")

  # Optional: time window aggregation (minutes)
  agg_minutes = st.sidebar.number_input("聚合窗口（分钟，中心化于所选分钟；0 即仅该分钟）", min_value=0, max_value=60, value=0, step=1)

  # Filter time range according to aggregation window
  if agg_minutes and agg_minutes > 0:
      half = int(agg_minutes // 2)
      start_ts = selected_minute - pd.Timedelta(minutes=half)
      end_ts = selected_minute + pd.Timedelta(minutes=half)
      df_window = df_var[(df_var["time"] >= start_ts) & (df_var["time"] <= end_ts)]
      st.write(f"聚合时间范围：{start_ts} — {end_ts}，行数 {len(df_window)}")
      # Take mean per platform
      df_group = df_window.groupby(["platform_id", "longitude", "latitude"], as_index=False).agg(
          value=("value", "mean"), count=("value", "count"), depth=("depth", "mean")
      )
  else:
      df_group = aggregate_for_minute(df_var, selected_minute)

  if df_group.empty:
      st.warning("当前帧没有数据，尝试切换到其它时间帧或放宽时间窗口。")
      st.stop()

  # Compute height and color
  df_plot = compute_height_and_color(df_group, target_p99_height=1000.0)

  # Compute map center and radius
  center_lat = float(df_plot["latitude"].mean())
  center_lon = float(df_plot["longitude"].mean())

  # Adaptive radius based on lat/lon span (avoid all bars overlapping)
  lat_span = df_plot["latitude"].max() - df_plot["latitude"].min()
  lon_span = df_plot["longitude"].max() - df_plot["longitude"].min()
  max_span_deg = max(lat_span, lon_span, 0.01)
  # 1 deg ~ 111km (approx.)
  radius_m = 40000

  # Build pydeck Layer
  layer = pdk.Layer(
      "ColumnLayer",
      data=df_plot,
      get_position=["longitude", "latitude"],
      get_elevation="height",
      elevation_scale=500.0,
      radius=radius_m,
      get_fill_color="color",
      pickable=True,
      auto_highlight=True,
  )

  view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=4, pitch=45)

  tooltip = {
      "html": "<b>{platform_id}</b><br/>value: {value:.3f}<br/>count: {count}<br/>lat: {latitude}, lon: {longitude}<br/>depth: {depth}",
      "style": {"color": "white"},
  }

  deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style=None)

  st.subheader(f"地图：变量 {variable}（时间：{selected_minute} UTC）")
  st.pydeck_chart(deck, use_container_width=True)

  # Show table and stats on the right or below
  with st.expander("当前帧数据表格（聚合后）", expanded=True):
      st.dataframe(df_plot.sort_values("value", ascending=False).reset_index(drop=True))

  # Simple statistics & distribution
  col1, col2 = st.columns([2, 1])
  with col1:
      st.metric("站点数量（当前帧）", f"{len(df_plot)}")
      st.write("值统计（原始均值）")
      desc = df_plot["value"].describe().to_frame().T
      st.table(desc)
  with col2:
      st.write("值分布直方图")
      st.bar_chart(df_plot["value"].dropna())

  st.caption("说明：柱高为 value 的归一化高度（99% 映射到 ~1000m）。若你希望用原始物理单位直接映射高度，请在代码中调整 compute_height_and_color 的映射策略。")

  # Export current aggregated table as CSV
  csv_bytes = df_plot.to_csv(index=False).encode("utf-8")
  st.download_button("下载当前帧聚合数据（CSV）", csv_bytes, file_name=f"{variable}_{selected_minute.strftime('%Y%m%dT%H%M')}.csv")