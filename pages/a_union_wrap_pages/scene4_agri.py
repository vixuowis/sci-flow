# app.py
# Streamlit page: dynamically display 3D bar charts of amplitudes at each station on the map over time (30 frames per minute)

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
#    Filenames must be in the form: IU_AFI_-13.91_-171.78_waveforms.mseed
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
PREFIX = "农业智慧-文本+图像+时序/outputs/"
# -----------------------
# Utility functions
# -----------------------
def _create_obs_client(endpoint: str, access_key: Optional[str], secret_key: Optional[str]) -> ObsClient:
    if access_key and secret_key:
        return ObsClient(access_key_id=access_key, secret_access_key=secret_key, server=endpoint)
    else:
        return ObsClient(server=endpoint)


def list_all_objects_under_prefix(client: ObsClient, bucket: str, prefix: str, max_keys: int = 1000) -> List[str]:
    """List all object keys under a prefix with pagination"""
    prefix = prefix.lstrip('/')
    keys = []
    marker = None

    while True:
        resp = client.listObjects(bucket, prefix=prefix, marker=marker, max_keys=max_keys)
        if resp.status >= 300:
            raise RuntimeError(f"listObjects failed: status={resp.status}, reason={resp.reason}")

        body = resp.body
        contents = getattr(body, "contents", []) or []
        for c in contents:
            key = getattr(c, "key", None) or (c.get("key") if isinstance(c, dict) else None)
            if key:
                keys.append(key)

        if not getattr(body, "is_truncated", False):
            break

        marker = getattr(body, "next_marker", None) or getattr(body, "nextMarker", None)
        if not marker and contents:
            last_key = getattr(contents[-1], "key", None) or contents[-1].get("key")
            marker = last_key

        if not marker:
            break

    return keys


def filter_and_dedup_parquet(keys: List[str], prefix: str) -> List[str]:
    """Filter parquet files and remove duplicates (xx.parquet vs xx_1.parquet)"""
    parquet_keys = [k for k in keys if k.lower().endswith(".parquet")]

    groups = {}
    for k in parquet_keys:
        fname = os.path.basename(k)
        stem = re.sub(r'(_\d+)(?=\.parquet$)', '', fname, flags=re.IGNORECASE)
        groups.setdefault(stem, []).append(k)

    selected = []
    for stem, lst in groups.items():
        plain = stem if stem.lower().endswith('.parquet') else f"{stem}.parquet"
        exact = None
        for k in lst:
            if os.path.basename(k) == plain:
                exact = k
                break
        chosen = exact if exact else sorted(lst)[0]
        selected.append(chosen)

    return sorted(selected)


def build_public_urls(endpoint: str, bucket: str, keys: List[str]) -> List[str]:
    """Construct a list of public URLs"""
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


def build_parquet_url_list(
    endpoint: str,
    bucket: str,
    prefix: str,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> List[str]:
    prefix = unquote(prefix)
    client = _create_obs_client(endpoint, access_key, secret_key)
    try:
        keys = list_all_objects_under_prefix(client, bucket, prefix)
        parquet_keys = filter_and_dedup_parquet(keys, prefix)
        urls = build_public_urls(endpoint, bucket, parquet_keys)
        return urls
    finally:
        try:
            client.close()
        except Exception:
            pass
# # If credentials are required, fill them in; if OBS is publicly accessible, leave them empty and only pass server (or use ENV mechanism)
# ACCESS_KEY = None  # or "your AK"
# SECRET_KEY = None  # or "your SK"
def run_scene4():
    
    # url_list = [
    #     "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/.../IU_AFI_-13.91_-171.78_waveforms.mseed",
    #     "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/.../IU_CASY_-66.28_110.54_waveforms.mseed",
    # ]
    url_list = build_parquet_url_list(ENDPOINT, BUCKET, PREFIX, None, None)

    # -----------------------
    # Config: variables to display (customize as needed)
    # -----------------------
    VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
    ]

    # -----------------------
    # Helper: parse site code and lat/lon from parquet filename
    # Example: ACS_-28.893060_136.169440_agdata.parquet
    # -----------------------
    PARQUET_FNAME_RE = re.compile(
        r"(?P<code>[A-Za-z0-9\-]+)_(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+).*\.parquet$"
    )


    def parse_meta_from_parquet_url(url: str):
        # First get the unencoded basename
        path = urlparse(url).path
        path = unquote(path)
        fname = os.path.basename(path)
        m = PARQUET_FNAME_RE.search(fname)
        if not m:
            # If unable to parse lat/lon, return None so caller can skip or warn
            return None
        d = m.groupdict()
        d["lat"] = float(d["lat"])
        d["lon"] = float(d["lon"])
        d["code"] = d["code"]
        return d


    # -----------------------
    # Load a single parquet (cached)
    # -----------------------
    @st.cache_data(show_spinner=True)
    def load_one_parquet(url: str) -> pd.DataFrame:
        """
        Download and read a single parquet file, return DataFrame with lat/lon/code columns added.
        Raise an exception if file read or parsing fails.
        """
        meta = parse_meta_from_parquet_url(url)
        if meta is None:
            raise ValueError(f"Unable to parse lat/lon from filename: {url}")

        # Try downloading and reading with pandas (pyarrow supports buffer)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)

        try:
            # Requires pyarrow or fastparquet installed
            df = pd.read_parquet(buf)
        except Exception as e:
            # Fallback: let pandas read directly from URL (in some environments)
            try:
                df = pd.read_parquet(url)
            except Exception:
                raise RuntimeError(f"Failed to read parquet: {url}. Error: {e}")

        # Ensure time column exists and parse as pandas.Timestamp (keep hourly resolution)
        if "time" not in df.columns:
            raise ValueError(f"No 'time' column found in parquet: {url}")
        # Parse time; keep timezone-naive or use utc=True depending on preference
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        # Add metadata
        df["lat"] = meta["lat"]
        df["lon"] = meta["lon"]
        df["code"] = meta["code"]
        # Keep only time and selected variables (other columns can remain as well)
        return df


    @st.cache_data(show_spinner=True)
    def load_all_parquets(urls):
        """
        Batch load multiple parquet files, merge into a single DataFrame.
        Return (df_all, available_variables)
        """
        parts = []
        failed = []
        for u in urls:
            try:
                parts.append(load_one_parquet(u))
            except Exception as e:
                failed.append((u, str(e)))
        if failed:
            for u, e in failed:
                st.warning(f"加载失败：{u} -> {e}")
        if not parts:
            return pd.DataFrame(), []

        df_all = pd.concat(parts, ignore_index=True)
        # Normalize column names (strip whitespace)
        df_all.columns = [c.strip() for c in df_all.columns]

        # Which target variables exist in the dataset
        avail_vars = [v for v in VARIABLES if v in df_all.columns]

        return df_all, avail_vars


    # -----------------------
    # Page body
    # -----------------------
    st.set_page_config(page_title="Parquet Map Viewer", layout="wide")
    st.title("Scene 4")

    # Ensure url_list is available
    if not url_list:
        st.error(
            "未检测到 `url_list`（parquet URLs）。\n\n请在脚本顶部设置 `url_list = build_parquet_url_list(...)` 或直接赋值为 URL 列表。"
        )
        st.stop()

    with st.spinner("加载 parquet 文件（可能较多，请耐心）..."):
        df_all, avail_vars = load_all_parquets(tuple(url_list))

    if df_all.empty:
        st.error("未加载到任何 parquet 数据。请检查 URL 列表与访问权限。")
        st.stop()

    if not avail_vars:
        st.error(f"在加载的数据中未找到以下任何目标变量：{VARIABLES}")
        st.stop()

    # Timeline: sort unique hourly timestamps
    times = sorted(df_all["time"].dropna().unique())
    if len(times) == 0:
        st.error("未找到任何时间点。")
        st.stop()

    # Left: control panel
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("控制面板")

        # Time slider (move by index)
        time_idx = st.slider(
            "选择时间（拖动）",
            min_value=0,
            max_value=len(times) - 1,
            value=0,
            step=1,
            format="%d",
        )
        selected_time = pd.to_datetime(times[time_idx])
        st.markdown(f"**当前时间：** {selected_time}")

        # Variable slider (simulate variable selection by index)
        var_idx = st.slider(
            "选择变量（拖动） —— 变量索引",
            min_value=0,
            max_value=len(avail_vars) - 1,
            value=0,
            step=1,
        )
        selected_var = avail_vars[var_idx]
        st.markdown(f"**当前变量：** {selected_var}")

        # Control max bar height (meters) for interactive zooming
        max_elev_m = st.slider("最大柱高 (米)：", min_value=100, max_value=20000, value=5000, step=100)

        # Option to invert colors (small feature)
        invert_color = st.checkbox("反转颜色（低值为红，高值为蓝）", value=False)

    with col2:
        st.subheader("地图与柱状展示")

        # Current time slice (one row per station)
        df_now = df_all[df_all["time"] == selected_time].copy()
        if df_now.empty:
            st.warning("当前时间没有数据点。尝试移动时间滑块。")
            st.stop()

        # Extract values to display (may contain NaNs)
        values = pd.to_numeric(df_now[selected_var], errors="coerce")
        df_now["value"] = values

        # Normalize (0..1); if all values identical, set to 0.5
        vmin = float(df_all[selected_var].min(skipna=True))
        vmax = float(df_all[selected_var].max(skipna=True))
        if np.isclose(vmax, vmin):
            df_now["norm"] = 0.5
        else:
            df_now["norm"] = (df_now["value"] - vmin) / (vmax - vmin)
            df_now["norm"] = df_now["norm"].clip(0, 1)

        # elevation: map norm to [0, max_elev_m]
        df_now["elevation"] = df_now["norm"].fillna(0.0) * float(max_elev_m)

        # Color mapping: blue->red (or reversed)
        if not invert_color:
            df_now["r"] = (df_now["norm"] * 255).fillna(100).astype(int)
            df_now["g"] = (50 + (1 - df_now["norm"]) * 100).fillna(100).astype(int)
            df_now["b"] = (255 - df_now["norm"] * 255).fillna(200).astype(int)
        else:
            df_now["r"] = (255 - df_now["norm"] * 255).fillna(200).astype(int)
            df_now["g"] = (50 + (1 - df_now["norm"]) * 100).fillna(100).astype(int)
            df_now["b"] = (df_now["norm"] * 255).fillna(100).astype(int)

        # Prepare pydeck data (longitude first)
        df_now["lon"] = pd.to_numeric(df_now["lon"])
        df_now["lat"] = pd.to_numeric(df_now["lat"])

        # Tooltip configuration
        tooltip = {
            "html": "<b>{code}</b><br/>value: {value}<br/>lat: {lat}, lon: {lon}",
            "style": {"color": "white"},
        }

        # ColumnLayer: fixed base radius, bar height controlled by 'elevation' field (elevation_scale=1)
        col_layer = pdk.Layer(
            "ColumnLayer",
            data=df_now,
            get_position="[lon, lat]",
            get_elevation="elevation",
            elevation_scale=1000,
            radius=100000,  # Same base radius for all cylinders (meters), adjust as needed
            get_fill_color="[r, g, b, 180]",
            pickable=True,
            auto_highlight=True,
        )

        # ScatterplotLayer: mark station locations
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=df_now,
            get_position="[lon, lat]",
            get_color="[r, g, b, 180]",
            get_radius=6000,
            pickable=True,
        )

        # View center: use mean lat/lon of current data, fallback if none
        try:
            center_lat = float(df_now["lat"].mean())
            center_lon = float(df_now["lon"].mean())
        except Exception:
            center_lat, center_lon = 0.0, 0.0

        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3, pitch=45)

        deck = pdk.Deck(layers=[col_layer, scatter], initial_view_state=view_state, tooltip=tooltip, map_style=None)

        st.pydeck_chart(deck, use_container_width=True)

        # Show current frame data table
        with st.expander("当前时间切片数据（表格）", expanded=False):
            show_cols = ["code", "lat", "lon", "value", "elevation"]
            available = [c for c in show_cols if c in df_now.columns]
            st.dataframe(df_now[available].sort_values("code").reset_index(drop=True))

    st.caption(
        f"说明：变量集合 = {avail_vars}. \n柱高为选中变量按整个数据集归一化后的值乘以你设定的 最大柱高({max_elev_m}米)。"
    )