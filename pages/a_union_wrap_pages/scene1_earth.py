# app.py
# Streamlit page: Display 3D bar charts of station amplitudes on the map dynamically over time (30 frames per minute)

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
# 1) Global URL list (replace with your accessible mseed links)
#    Filenames must follow the format: IU_AFI_-13.91_-171.78_waveforms.mseed
# ==============================

import os
import re
from typing import List, Tuple, Optional
from urllib.parse import urlparse, quote, unquote

from obs import ObsClient  # pip install esdk-obs-python

# -----------------------
# Configure here or read from environment
# -----------------------
ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"
BUCKET = "gaoyuan-49d0"
PREFIX = "地质灾害-文本+遥感+时序/IRIS/dataset_earthquake/" 

# # Fill in credentials if needed. Leave empty if OBS is publicly accessible (or use ENV mechanism).
# ACCESS_KEY = None  # or "your AK"
# SECRET_KEY = None  # or "your SK"

# -----------------------
# Internal implementation
# -----------------------

def _create_obs_client(endpoint: str, access_key: Optional[str], secret_key: Optional[str]) -> ObsClient:
    """Create ObsClient: use AK/SK if provided, otherwise only pass server (can rely on ENV/ECS strategy)."""
    if access_key and secret_key:
        return ObsClient(access_key_id=access_key, secret_access_key=secret_key, server=endpoint)
    else:
        # If environment variables or ECS authorization are available, this constructor also works (see official security_provider_policy option).
        return ObsClient(server=endpoint)


def list_all_objects_under_prefix(client: ObsClient, bucket: str, prefix: str, max_keys: int = 1000) -> List[str]:
    """
    Use listObjects + marker to fetch all object keys under prefix (recursive).
    Return object key list (object name, without bucket).
    Note: Each call returns up to max_keys (limit 1000).
    Reference: listObjects / is_truncated / next_marker.
    """
    # Ensure prefix is in object name format (not URL encoded)
    prefix = prefix.lstrip('/')
    keys = []
    marker = None

    while True:
        resp = client.listObjects(bucket, prefix=prefix, marker=marker, max_keys=max_keys)
        if resp.status >= 300:
            raise RuntimeError(f"listObjects failed: status={resp.status}, reason={resp.reason}, msg={getattr(resp, 'errorMessage', '')}")

        body = resp.body
        contents = getattr(body, "contents", []) or []
        for c in contents:
            # Support attribute access or dict access (compatible with SDK return format)
            key = getattr(c, "key", None) or (c.get("key") if isinstance(c, dict) else None)
            if key:
                keys.append(key)

        # Pagination check
        if not getattr(body, "is_truncated", False):
            break

        # Next marker
        next_marker = getattr(body, "next_marker", None) or getattr(body, "nextMarker", None)
        if next_marker:
            marker = next_marker
        else:
            # Fallback: use last key value
            if contents:
                last_key = None
                last = contents[-1]
                last_key = getattr(last, "key", None) or (last.get("key") if isinstance(last, dict) else None)
                if last_key:
                    marker = last_key
                else:
                    break
            else:
                break

    return keys


def filter_and_dedup_mseed(keys: List[str], prefix: str, max_depth: Optional[int] = None) -> List[str]:
    """
    From keys (full object key list), filter .mseed files and deduplicate by rules:
    - Ignore files with depth exceeding max_depth (if provided).
    - For groups like `xx.mseed`, `xx_1.mseed`, `xx_2.mseed`, only keep the first choice:
        * Prefer the original name `xx.mseed` without _number (if exists).
        * Otherwise, choose the lexicographically first one.
    Return selected object keys (relative to bucket).
    """
    prefix = prefix.rstrip('/') + '/' if prefix and not prefix.endswith('/') else prefix
    mseed_keys = []
    for k in keys:
        if not k.lower().endswith('.mseed'):
            continue
        # Depth limit (relative path to prefix)
        if prefix and k.startswith(prefix):
            rel = k[len(prefix):]
        else:
            rel = k
        if max_depth is not None:
            # If rel is empty or ends with '/', depth = segment count
            depth = 0 if rel == "" else rel.count('/')
            if depth > max_depth:
                continue
        mseed_keys.append(k)

    # Group by stem: remove ending "_number" (before .mseed extension)
    groups = {}
    for k in mseed_keys:
        fname = os.path.basename(k)
        # Stem: strip trailing _1,_2 when the last part is numeric
        stem = re.sub(r'(_\d+)(?=\.mseed$)', '', fname, flags=re.IGNORECASE)
        groups.setdefault(stem, []).append(k)

    selected = []
    for stem, lst in groups.items():
        # Prefer exact match: stem + ".mseed"
        plain = stem if stem.lower().endswith('.mseed') else f"{stem}.mseed"
        exact = None
        for k in lst:
            if os.path.basename(k) == plain:
                exact = k
                break
        if exact:
            chosen = exact
        else:
            chosen = sorted(lst)[0]  # Otherwise choose the lexicographically first one
        selected.append(chosen)

    # Return sorted result (for stability)
    return sorted(selected)


def build_public_urls(endpoint: str, bucket: str, keys: List[str]) -> List[str]:
    """
    Construct public access URL list from endpoint and bucket.
    - If endpoint hostname already contains bucket (e.g. user provided 'https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com'), use it directly.
    - Otherwise, construct base as {scheme}://{bucket}.{netloc}.
    Encode object key with urllib.parse.quote (preserve '/').
    """
    endpoint = endpoint.rstrip('/')
    parsed = urlparse(endpoint if '://' in endpoint else 'https://' + endpoint)
    scheme = parsed.scheme or 'https'
    netloc = parsed.netloc or parsed.path

    if netloc.startswith(bucket + "."):
        base = f"{scheme}://{netloc}"
    else:
        base = f"{scheme}://{bucket}.{netloc}"

    urls = [f"{base}/{quote(k, safe='/')}" for k in keys]
    return urls


def build_mseed_url_list(
    endpoint: str,
    bucket: str,
    prefix: str,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    max_depth: Optional[int] = None,
) -> List[str]:
    """
    High-level function: list all mseed files under prefix (deduplicated), return accessible URL list.
    """
    # Ensure prefix is decoded (in case user passed URL-encoded value)
    prefix = unquote(prefix)
    client = _create_obs_client(endpoint, access_key, secret_key)
    try:
        keys = list_all_objects_under_prefix(client, bucket, prefix)
        mseed_keys = filter_and_dedup_mseed(keys, prefix, max_depth=max_depth)
        urls = build_public_urls(endpoint, bucket, mseed_keys)
        return urls
    finally:
        try:
            client.close()
        except Exception:
            pass

def run_scene1():
    url_list = build_mseed_url_list(ENDPOINT, BUCKET, PREFIX, None, None, max_depth=None)

    # ------------------------------
    # Utility functions
    # ------------------------------
    FNAME_RE = re.compile(
        r"(?P<net>[A-Z0-9]+)_(?P<sta>[A-Z0-9]+)_(?P<lat>[-0-9.]+)_(?P<lon>[-0-9.]+)_waveforms\.mseed$"
    )

    def parse_meta_from_url(url: str):
        """Parse network, station, and lat/lon from filename"""
        fname = os.path.basename(url)
        m = FNAME_RE.match(fname)
        if not m:
            raise ValueError(f"Filename does not match pattern: {fname}")
        d = m.groupdict()
        d["lat"] = float(d["lat"])
        d["lon"] = float(d["lon"])
        return d  # {net, sta, lat, lon}

    def pick_best_trace(stream):
        """Prefer Z component with highest sampling rate; if no Z, choose highest sampling rate overall"""
        z_traces = [tr for tr in stream if tr.stats.channel.endswith("Z")]
        candidates = z_traces if z_traces else list(stream)
        # Pick the one with highest sampling rate
        return max(candidates, key=lambda tr: float(getattr(tr.stats, "sampling_rate", 0.0)))

    def to_pandas_time(utc_list):
        """ObsPy UTCDateTime list -> pandas.Timestamp(UTC)"""
        return pd.to_datetime([t.datetime for t in utc_list], utc=True)

    def evenly_sample_per_minute(df, n=30):
        """
        For one station's (time, amplitude), group by minute,
        uniformly select n samples per minute; add slot (0..n-1) to each sample.
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
        # 确保minute列没有丢失

        return out.reset_index(drop=True)

    # ------------------------------
    # Data loading (cached)
    # ------------------------------
    @st.cache_data(show_spinner=True)
    def load_one_url(url: str) -> pd.DataFrame:
        meta = parse_meta_from_url(url)
        # Download
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        # Read mseed, select best trace
        st_obj = read(io.BytesIO(r.content))
        tr = pick_best_trace(st_obj)

        times = to_pandas_time(tr.times("utcdatetime"))
        amps = tr.data.astype(float)

        df = pd.DataFrame({"time": times, "amplitude": amps})
        df["net"] = meta["net"]
        df["sta"] = meta["sta"]
        df["lat"] = meta["lat"]
        df["lon"] = meta["lon"]

        # Uniformly sample 30 points per minute (or fewer if not enough)
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

        # Build frame index: ascending by minute, then slot
        minutes_sorted = np.array(sorted(df_all["minute"].dropna().unique()))
        minute_index = {m: i for i, m in enumerate(minutes_sorted)}
        df_all["minute_idx"] = df_all["minute"].map(minute_index).astype("Int64")
        df_all["frame_idx"] = df_all["minute_idx"] * 30 + df_all["slot"].astype(int)

        # Frame list for display (frames that actually exist)
        frames = (
            df_all[["minute", "slot", "frame_idx"]]
            .drop_duplicates()
            .sort_values(["frame_idx"])
            .reset_index(drop=True)
        )

        # Map center (mean lat/lon of all stations)
        center_lat = float(df_all["lat"].mean())
        center_lon = float(df_all["lon"].mean())
        coords = df_all[["lat", "lon"]].drop_duplicates().to_numpy()

        return df_all, frames, center_lat, center_lon, coords

    # ------------------------------
    # Page
    # ------------------------------
    st.set_page_config(page_title="地震波形柱状图（动态）", layout="wide")
    st.title("🌍 按时间动态展示：各台站振幅 3D 柱状图")

    # Optional: show URL list for confirmation
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

    # Time frame slider (across minutes and slots)
    min_frame = int(frames["frame_idx"].min())
    max_frame = int(frames["frame_idx"].max())
    frame_val = st.slider("选择时间帧（每分钟均匀 30 帧）", min_value=min_frame, max_value=max_frame, value=min_frame, step=1)

    # Current frame's minute/slot info
    row = frames.loc[frames["frame_idx"] == frame_val].iloc[0]
    cur_minute = pd.to_datetime(row["minute"])
    cur_slot = int(row["slot"])
    st.markdown(f"**当前分钟：** {cur_minute}  &nbsp;&nbsp; **帧内 slot：** {cur_slot+1}/30")

    # Select current frame data (one record per station for this minute and slot)
    df_now = df_all[(df_all["minute"] == cur_minute) & (df_all["slot"] == cur_slot)].copy()

    if df_now.empty:
        st.warning("该帧无数据，试试其它帧。")
        st.stop()

    # Visualization fields: bar height = |amplitude|; color distinguishes sign
    df_now["height"] = np.abs(df_now["amplitude"].astype(float))
    df_now["r"] = np.where(df_now["amplitude"] >= 0, 255, 30)   # Positive = reddish, Negative = bluish
    df_now["g"] = 50
    df_now["b"] = np.where(df_now["amplitude"] >= 0, 60, 255)

    # Dynamic height scaling: make 99th percentile height ~ 1000m
    p99 = float(df_now["height"].quantile(0.99)) if len(df_now) > 1 else float(df_now["height"].max())
    elev_scale = 1000.0

    # Radius (meters): can increase if global geographic range is large
    radius_m = 100000

    tooltip = {
        "html": "<b>{net}.{sta}</b><br/>amp: {amplitude}<br/>lat: {lat}, lon: {lon}<br/>{minute} slot {slot}",
        "style": {"color": "white"},
    }

    layer = pdk.Layer(
        "ColumnLayer",
        data=df_now,
        get_position="[lon, lat]",     # Note: longitude first
        get_elevation="height",
        elevation_scale=elev_scale,     # Adjust height
        radius=radius_m,
        get_fill_color="[r, g, b, 180]",
        pickable=True,
        auto_highlight=True,
    )

    # View: centered at all stations
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=2,
        pitch=45,
        bearing=0,
    )

    # Use map_style=None (no token required)
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None,
    )

    st.pydeck_chart(deck, use_container_width=True)

    with st.expander("数据切片（当前帧）"):
        st.dataframe(
            df_now[["net","sta","lat","lon","time","amplitude","minute","slot"]]
            .sort_values(["net","sta"])
            .reset_index(drop=True)
        )

    st.caption("提示：柱高为 |amplitude|，颜色区分正负；若地图不显示，请确保 URL 可访问，且文件名包含经纬度。")