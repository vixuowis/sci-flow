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

import os
import re
from typing import List, Tuple, Optional
from urllib.parse import urlparse, quote, unquote

from obs import ObsClient  # pip install esdk-obs-python

# -----------------------
# 请在此处配置或从环境读取
# -----------------------

ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"
BUCKET = "gaoyuan-49d0"
PREFIX = "农业智慧-文本+图像+时序/outputs/"
# -----------------------
# 工具函数
# -----------------------
def _create_obs_client(endpoint: str, access_key: Optional[str], secret_key: Optional[str]) -> ObsClient:
    if access_key and secret_key:
        return ObsClient(access_key_id=access_key, secret_access_key=secret_key, server=endpoint)
    else:
        return ObsClient(server=endpoint)


def list_all_objects_under_prefix(client: ObsClient, bucket: str, prefix: str, max_keys: int = 1000) -> List[str]:
    """分页列出 prefix 下的所有对象 key"""
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
    """筛选 parquet 文件并去重（xx.parquet vs xx_1.parquet）"""
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
    """构造公网 URL 列表"""
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
# # 如果需要凭证则填写，若 OBS 公开可访问可留空并只传 server（或使用 ENV 机制）
# ACCESS_KEY = None  # 或 "你的AK"
# SECRET_KEY = None  # 或 "你的SK"
def run_scene4():
    
    # url_list = [
    #     "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/%E5%9C%B0%E8%B4%A8%E7%81%BE%E5%AE%B3-%E6%96%87%E6%9C%AC%2B%E9%81%A5%E6%84%9F%2B%E6%97%B6%E5%BA%8F/IRIS/dataset_earthquake/IU_AFI_-13.91_-171.78_waveforms.mseed",
    #     "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/%E5%9C%B0%E8%B4%A8%E7%81%BE%E5%AE%B3-%E6%96%87%E6%9C%AC%2B%E9%81%A5%E6%84%9F%2B%E6%97%B6%E5%BA%8F/IRIS/dataset_earthquake/IU_CASY_-66.28_110.54_waveforms.mseed",
    # ]
    url_list = build_parquet_url_list(ENDPOINT, BUCKET, PREFIX, None, None)

    # -----------------------
    # 配置：要展示的变量（按你的要求）
    # -----------------------
    VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
    ]

    # -----------------------
    # 辅助：从 parquet 文件名解析站点与经纬度
    # 例如： ACS_-28.893060_136.169440_agdata.parquet
    # -----------------------
    PARQUET_FNAME_RE = re.compile(
        r"(?P<code>[A-Za-z0-9\-]+)_(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+).*\.parquet$"
    )


    def parse_meta_from_parquet_url(url: str):
        # 先取未编码的 basename
        path = urlparse(url).path
        path = unquote(path)
        fname = os.path.basename(path)
        m = PARQUET_FNAME_RE.search(fname)
        if not m:
            # 如果无法解析经纬度，返回 None，让调用方跳过或警告
            return None
        d = m.groupdict()
        d["lat"] = float(d["lat"])
        d["lon"] = float(d["lon"])
        d["code"] = d["code"]
        return d


    # -----------------------
    # 加载单个 parquet（缓存）
    # -----------------------
    @st.cache_data(show_spinner=True)
    def load_one_parquet(url: str) -> pd.DataFrame:
        """
        下载并读取单个 parquet 文件，返回 DataFrame 并加上 lat/lon/code 列。
        若文件读取失败或解析失败，抛出异常。
        """
        meta = parse_meta_from_parquet_url(url)
        if meta is None:
            raise ValueError(f"无法从文件名解析经纬度: {url}")

        # 尝试下载并用 pandas 读取（pyarrow 支持从 buffer）
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)

        try:
            # 要求环境中安装 pyarrow 或 fastparquet
            df = pd.read_parquet(buf)
        except Exception as e:
            # 再尝试直接让 pandas 从 URL 读取（某些环境）
            try:
                df = pd.read_parquet(url)
            except Exception:
                raise RuntimeError(f"读取 parquet 失败: {url}. Error: {e}")

        # 确保 time 列存在并解析为 pandas.Timestamp（保留小时分辨率）
        if "time" not in df.columns:
            raise ValueError(f"parquet 中未找到 time 列: {url}")
        # 解析时间；保持为 timezone-naive 或使用 utc=True 视你偏好而定
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        # 添加元信息
        df["lat"] = meta["lat"]
        df["lon"] = meta["lon"]
        df["code"] = meta["code"]
        # 保留只有时间和关注变量（其它列也保留无妨）
        return df


    @st.cache_data(show_spinner=True)
    def load_all_parquets(urls):
        """
        批量加载多个 parquet，合并为单个 DataFrame。
        返回 (df_all, available_variables)
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
        # 标准化列名（去空格）
        df_all.columns = [c.strip() for c in df_all.columns]

        # 哪些我们关心的变量在数据里存在
        avail_vars = [v for v in VARIABLES if v in df_all.columns]

        return df_all, avail_vars


    # -----------------------
    # 页面主体
    # -----------------------
    st.set_page_config(page_title="Parquet Map Viewer", layout="wide")
    st.title("Scene 4")

    # 确认 url_list 可用
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

    # 时间轴：按小时唯一时间点排序
    times = sorted(df_all["time"].dropna().unique())
    if len(times) == 0:
        st.error("未找到任何时间点。")
        st.stop()

    # 左侧：控制面板
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("控制面板")

        # 时间滑块（按 index 拖动）
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

        # 变量滑块（用索引的滑块来模拟“拖动 bar 选择变量”）
        var_idx = st.slider(
            "选择变量（拖动） —— 变量索引",
            min_value=0,
            max_value=len(avail_vars) - 1,
            value=0,
            step=1,
        )
        selected_var = avail_vars[var_idx]
        st.markdown(f"**当前变量：** {selected_var}")

        # 控制最大柱高（米），用于交互式放大/缩小
        max_elev_m = st.slider("最大柱高 (米)：", min_value=100, max_value=20000, value=5000, step=100)

        # 颜色反转选项（小功能）
        invert_color = st.checkbox("反转颜色（低值为红，高值为蓝）", value=False)

    with col2:
        st.subheader("地图与柱状展示")

        # 取当前时间的数据（每个站点一行）
        df_now = df_all[df_all["time"] == selected_time].copy()
        if df_now.empty:
            st.warning("当前时间没有数据点。尝试移动时间滑块。")
            st.stop()

        # 取出要展示的值（可能为 NaN）
        values = pd.to_numeric(df_now[selected_var], errors="coerce")
        df_now["value"] = values

        # 计算归一化（0..1）；若所有值相同则设为 0.5
        vmin = float(df_all[selected_var].min(skipna=True))
        vmax = float(df_all[selected_var].max(skipna=True))
        if np.isclose(vmax, vmin):
            df_now["norm"] = 0.5
        else:
            df_now["norm"] = (df_now["value"] - vmin) / (vmax - vmin)
            df_now["norm"] = df_now["norm"].clip(0, 1)

        # elevation: 将 norm 映射到 [0, max_elev_m]
        df_now["elevation"] = df_now["norm"].fillna(0.0) * float(max_elev_m)

        # 颜色映射：从蓝->红（或反转）
        if not invert_color:
            df_now["r"] = (df_now["norm"] * 255).fillna(100).astype(int)
            df_now["g"] = (50 + (1 - df_now["norm"]) * 100).fillna(100).astype(int)
            df_now["b"] = (255 - df_now["norm"] * 255).fillna(200).astype(int)
        else:
            df_now["r"] = (255 - df_now["norm"] * 255).fillna(200).astype(int)
            df_now["g"] = (50 + (1 - df_now["norm"]) * 100).fillna(100).astype(int)
            df_now["b"] = (df_now["norm"] * 255).fillna(100).astype(int)

        # 准备 pydeck 数据（经度在前）
        df_now["lon"] = pd.to_numeric(df_now["lon"])
        df_now["lat"] = pd.to_numeric(df_now["lat"])

        # tooltip 配置
        tooltip = {
            "html": "<b>{code}</b><br/>value: {value}<br/>lat: {lat}, lon: {lon}",
            "style": {"color": "white"},
        }

        # ColumnLayer：底面半径固定，柱高由 elevation 字段控制（elevation_scale=1）
        col_layer = pdk.Layer(
            "ColumnLayer",
            data=df_now,
            get_position="[lon, lat]",
            get_elevation="elevation",
            elevation_scale=1000,
            radius=100000,  # 所有圆柱底面相同（米），按需要调整
            get_fill_color="[r, g, b, 180]",
            pickable=True,
            auto_highlight=True,
        )

        # ScatterplotLayer：标出站点位置
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=df_now,
            get_position="[lon, lat]",
            get_color="[r, g, b, 180]",
            get_radius=6000,
            pickable=True,
        )

        # 视图中心：以当前数据的平均经纬度为中心，若无则 fallback
        try:
            center_lat = float(df_now["lat"].mean())
            center_lon = float(df_now["lon"].mean())
        except Exception:
            center_lat, center_lon = 0.0, 0.0

        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3, pitch=45)

        deck = pdk.Deck(layers=[col_layer, scatter], initial_view_state=view_state, tooltip=tooltip, map_style=None)

        st.pydeck_chart(deck, use_container_width=True)

        # 展示当前帧数据表
        with st.expander("当前时间切片数据（表格）", expanded=False):
            show_cols = ["code", "lat", "lon", "value", "elevation"]
            available = [c for c in show_cols if c in df_now.columns]
            st.dataframe(df_now[available].sort_values("code").reset_index(drop=True))

    st.caption(
        f"说明：变量集合 = {avail_vars}. \n柱高为选中变量按整个数据集归一化后的值乘以你设定的 最大柱高({max_elev_m}米)。"
    )