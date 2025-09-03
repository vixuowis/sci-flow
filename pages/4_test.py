# app.py
# Streamlit 页面：按时间动态在地图上展示各站点振幅的 3D 柱状图（每分钟均匀30帧）

import io
import os
import re
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from obspy import read

# ==============================
# 1) 全局 URL 列表（请替换为你的可访问 mseed 链接）
#    文件名必须形如：IU_AFI_-13.91_-171.78_waveforms.mseed
# ==============================
url_list = [
    "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/%E5%9C%B0%E8%B4%A8%E7%81%BE%E5%AE%B3-%E6%96%87%E6%9C%AC%2B%E9%81%A5%E6%84%9F%2B%E6%97%B6%E5%BA%8F/IRIS/dataset_earthquake/IU_AFI_-13.91_-171.78_waveforms.mseed",
    "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/%E5%9C%B0%E8%B4%A8%E7%81%BE%E5%AE%B3-%E6%96%87%E6%9C%AC%2B%E9%81%A5%E6%84%9F%2B%E6%97%B6%E5%BA%8F/IRIS/dataset_earthquake/IU_CASY_-66.28_110.54_waveforms.mseed",
]

# ------------------------------
# 工具函数
# ------------------------------
FNAME_RE = re.compile(
    r"(?P<net>[A-Z0-9]+)_(?P<sta>[A-Z0-9]+)_(?P<lat>[-0-9.]+)_(?P<lon>[-0-9.]+)_waveforms\.mseed$"
)

def parse_meta_from_url(url: str):
    """从文件名解析网络、台站、经纬度"""
    fname = os.path.basename(url)
    m = FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"文件名不符合规则: {fname}")
    d = m.groupdict()
    d["lat"] = float(d["lat"])
    d["lon"] = float(d["lon"])
    return d  # {net, sta, lat, lon}

def pick_best_trace(stream):
    """优先选 Z 分量，采样率最高的 Trace；若无 Z，则选采样率最高的"""
    z_traces = [tr for tr in stream if tr.stats.channel.endswith("Z")]
    candidates = z_traces if z_traces else list(stream)
    # 采样率最高者
    return max(candidates, key=lambda tr: float(getattr(tr.stats, "sampling_rate", 0.0)))

def to_pandas_time(utc_list):
    """ObsPy UTCDateTime 列表 -> pandas.Timestamp(UTC)"""
    return pd.to_datetime([t.datetime for t in utc_list], utc=True)

def evenly_sample_per_minute(df, n=30):
    """
    对单个站点的 (time, amplitude) 按分钟分组，
    每分钟均匀选 n 个样本；给每个样本加 slot(0..n-1)
    """
    df = df.sort_values("time")
    df["minute"] = df["time"].dt.floor("min")

    def _sampler(g):
        if len(g) == 0:
            return g
        if len(g) <= n:
            out = g.copy().reset_index(drop=True)
            out["slot"] = np.arange(len(out))
            return out
        idx = np.linspace(0, len(g) - 1, num=n, dtype=int)
        out = g.iloc[idx].copy()
        out["slot"] = np.arange(len(out))
        return out

    out = df.groupby("minute", group_keys=False).apply(_sampler)
    return out.reset_index(drop=True)

# ------------------------------
# 数据加载（缓存）
# ------------------------------
@st.cache_data(show_spinner=True)
def load_one_url(url: str) -> pd.DataFrame:
    meta = parse_meta_from_url(url)
    # 下载
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    # 读取 mseed，选择最佳 trace
    st_obj = read(io.BytesIO(r.content))
    tr = pick_best_trace(st_obj)

    times = to_pandas_time(tr.times("utcdatetime"))
    amps = tr.data.astype(float)

    df = pd.DataFrame({"time": times, "amplitude": amps})
    df["net"] = meta["net"]
    df["sta"] = meta["sta"]
    df["lat"] = meta["lat"]
    df["lon"] = meta["lon"]

    # 每分钟均匀采样 30 个点（不足则尽可能多）
    df_s = evenly_sample_per_minute(df, n=30)
    return df_s

@st.cache_data(show_spinner=True)
def load_all(urls):
    parts = []
    for u in urls:
        try:
            parts.append(load_one_url(u))
        except Exception as e:
            st.warning(f"加载失败 {u}: {e}")
    if not parts:
        return pd.DataFrame(columns=["time","amplitude","net","sta","lat","lon","minute","slot"])
    df_all = pd.concat(parts, ignore_index=True)

    # 构造帧索引：按 minute 升序、slot 升序
    minutes_sorted = np.array(sorted(df_all["minute"].dropna().unique()))
    minute_index = {m: i for i, m in enumerate(minutes_sorted)}
    df_all["minute_idx"] = df_all["minute"].map(minute_index).astype("Int64")
    df_all["frame_idx"] = df_all["minute_idx"] * 30 + df_all["slot"].astype(int)

    # 供展示使用的帧列表（实际存在的帧）
    frames = (
        df_all[["minute", "slot", "frame_idx"]]
        .drop_duplicates()
        .sort_values(["frame_idx"])
        .reset_index(drop=True)
    )

    # 地图中心（所有站点的经纬度均值）
    center_lat = float(df_all["lat"].mean())
    center_lon = float(df_all["lon"].mean())
    coords = df_all[["lat", "lon"]].drop_duplicates().to_numpy()

    return df_all, frames, center_lat, center_lon, coords

# ------------------------------
# 页面
# ------------------------------
st.set_page_config(page_title="地震波形柱状图（动态）", layout="wide")
st.title("🌍 按时间动态展示：各台站振幅 3D 柱状图")

# 可选：把 URL 列表展示出来，便于确认
with st.expander("查看 URL 列表"):
    st.write(url_list)

if not url_list:
    st.info("请先在代码顶部填入 mseed 文件的 URL 列表。")
    st.stop()

with st.spinner("加载与解析 mseed 数据中…"):
    df_all, frames, center_lat, center_lon, coords = load_all(tuple(url_list))

if df_all.empty or frames.empty:
    st.error("没有可展示的数据（可能所有链接都加载失败或文件为空）。")
    st.stop()

# 时间帧滑块（跨分钟与 slot）
min_frame = int(frames["frame_idx"].min())
max_frame = int(frames["frame_idx"].max())
frame_val = st.slider("选择时间帧（每分钟均匀 30 帧）", min_value=min_frame, max_value=max_frame, value=min_frame, step=1)

# 当前帧对应的 minute/slot 信息
row = frames.loc[frames["frame_idx"] == frame_val].iloc[0]
cur_minute = pd.to_datetime(row["minute"])
cur_slot = int(row["slot"])
st.markdown(f"**当前分钟：** {cur_minute}  &nbsp;&nbsp; **帧内 slot：** {cur_slot+1}/30")

# 取当前帧数据（这一分钟里、这个 slot 的每个站点各一条）
df_now = df_all[(df_all["minute"] == cur_minute) & (df_all["slot"] == cur_slot)].copy()

if df_now.empty:
    st.warning("该帧无数据，试试其它帧。")
    st.stop()

# 可视化字段：柱高用 |amplitude|；颜色按正负区分
df_now["height"] = np.abs(df_now["amplitude"].astype(float))
df_now["r"] = np.where(df_now["amplitude"] >= 0, 255, 30)   # 正=偏红，负=偏蓝
df_now["g"] = 50
df_now["b"] = np.where(df_now["amplitude"] >= 0, 60, 255)

# 动态高度缩放：让 99 分位高度 ~ 1000m
p99 = float(df_now["height"].quantile(0.99)) if len(df_now) > 1 else float(df_now["height"].max())
elev_scale = 10.0

# 半径（米）：全局地理范围大时可适当加大
radius_m = 200000

tooltip = {
    "html": "<b>{net}.{sta}</b><br/>amp: {amplitude}<br/>lat: {lat}, lon: {lon}<br/>{minute} slot {slot}",
    "style": {"color": "white"},
}

# HexagonLayer 展示振幅 -> 柱高
hex_layer = pdk.Layer(
    "HexagonLayer",
    data=df_now,
    get_position="[lon, lat]",
    radius=radius_m,              # 六边形半径 (米)，可调
    elevation_scale=0.05,      # 控制柱子高度（可调大）
    elevation_range=[0, 5000], # 柱子高度范围
    get_weight="amplitude",    # 振幅作为权重
    pickable=True,
    extruded=True,
    coverage=1,
)

# ScatterplotLayer 展示站点位置
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_now,
    get_position="[lon, lat]",
    get_color="[200, 30, 0, 160]",  # 红色点
    get_radius=15000,               # 圆点大小 (米)
)

view_state = pdk.ViewState(
    latitude=df_now["lat"].mean(),
    longitude=df_now["lon"].mean(),
    zoom=2,
    pitch=50,
)

deck = pdk.Deck(
    layers=[hex_layer, scatter_layer],
    initial_view_state=view_state,
    map_style=None,   # Streamlit 自带地图样式
    tooltip={"text": "{net}.{sta}\nAmp: {amplitude}"}
)

st.pydeck_chart(deck, use_container_width=True)

with st.expander("数据切片（当前帧）"):
    st.dataframe(
        df_now[["net","sta","lat","lon","time","amplitude","minute","slot"]]
        .sort_values(["net","sta"])
        .reset_index(drop=True)
    )

st.caption("提示：柱高为 |amplitude|，颜色区分正负；若地图不显示，请确保 URL 可访问，且文件名包含经纬度。")
