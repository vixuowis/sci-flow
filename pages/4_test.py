# app.py
# Streamlit 页面：按时间动态在地图上展示各站点振幅的 3D 柱状图（每分钟均匀30帧）
from session_fix import page_setup
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

from integration_helper import check_authentication, add_auth_sidebar
user_info = check_authentication()
add_auth_sidebar()

from typing import List, Tuple, Optional
from urllib.parse import urlparse, quote, unquote

from obs import ObsClient  # pip install esdk-obs-python

# -----------------------
# 请在此处配置或从环境读取
# -----------------------
ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"
BUCKET = "gaoyuan-49d0"
PREFIX = "地质灾害-文本+遥感+时序/IRIS/dataset_earthquake/" 

# # 如果需要凭证则填写，若 OBS 公开可访问可留空并只传 server（或使用 ENV 机制）
# ACCESS_KEY = None  # 或 "你的AK"
# SECRET_KEY = None  # 或 "你的SK"

# -----------------------
# 内部实现
# -----------------------

def _create_obs_client(endpoint: str, access_key: Optional[str], secret_key: Optional[str]) -> ObsClient:
    """创建 ObsClient：若提供 AK/SK 则使用，否则只传 server（可借助 ENV/ECS 策略）。"""
    if access_key and secret_key:
        return ObsClient(access_key_id=access_key, secret_access_key=secret_key, server=endpoint)
    else:
        # 如果环境变量或 ECS 授权可用，下面的构造也能工作（参考官方 security_provider_policy 选项）
        return ObsClient(server=endpoint)


def list_all_objects_under_prefix(client: ObsClient, bucket: str, prefix: str, max_keys: int = 1000) -> List[str]:
    """
    使用 listObjects + marker 分页获取 prefix 下的所有 object keys（递归）。
    返回 object key 列表（object name，不含 bucket）。
    注意：每次最多 max_keys（上限 1000）。
    参考文档：listObjects / is_truncated / next_marker。:contentReference[oaicite:3]{index=3}
    """
    # 保证 prefix 是对象名形式（非 URL 编码）
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
            # 支持属性访问或字典访问（兼容 SDK 返回格式）
            key = getattr(c, "key", None) or (c.get("key") if isinstance(c, dict) else None)
            if key:
                keys.append(key)

        # 翻页判断
        if not getattr(body, "is_truncated", False):
            break

        # next marker
        next_marker = getattr(body, "next_marker", None) or getattr(body, "nextMarker", None)
        if next_marker:
            marker = next_marker
        else:
            # 兼容处理：使用最后一个 key 的值
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
    从 keys（完整 object key 列表）筛选 .mseed 文件，并按规则去重：
    - 忽略 depth 超出 max_depth 的文件（若提供）
    - 对于类似 `xx.mseed`, `xx_1.mseed`, `xx_2.mseed` 的组，只保留 first choice:
        * 优先选择没有 _数字 的原名 `xx.mseed`（如果存在）
        * 否则选择该组中排序最靠前的一个
    返回选中的 object keys（相对于 bucket）。
    """
    prefix = prefix.rstrip('/') + '/' if prefix and not prefix.endswith('/') else prefix
    mseed_keys = []
    for k in keys:
        if not k.lower().endswith('.mseed'):
            continue
        # depth 限制（相对于 prefix 的相对路径）
        if prefix and k.startswith(prefix):
            rel = k[len(prefix):]
        else:
            rel = k
        if max_depth is not None:
            # 如果 rel 为空或以 / 结尾，计算深度为 segments 数
            depth = 0 if rel == "" else rel.count('/')
            if depth > max_depth:
                continue
        mseed_keys.append(k)

    # group by stem: 去掉结尾的 "_数字"（仅在 .mseed 扩展名前）
    groups = {}
    for k in mseed_keys:
        fname = os.path.basename(k)
        # stem 去除 _1,_2 形式的尾号（仅最后一段为数字时）
        stem = re.sub(r'(_\d+)(?=\.mseed$)', '', fname, flags=re.IGNORECASE)
        groups.setdefault(stem, []).append(k)

    selected = []
    for stem, lst in groups.items():
        # 优先找纯名 stem + ".mseed"
        plain = stem if stem.lower().endswith('.mseed') else f"{stem}.mseed"
        exact = None
        for k in lst:
            if os.path.basename(k) == plain:
                exact = k
                break
        if exact:
            chosen = exact
        else:
            chosen = sorted(lst)[0]  # 否则选择字典序最先的做代表
        selected.append(chosen)

    # 返回排序过的结果（便于稳定）
    return sorted(selected)


def build_public_urls(endpoint: str, bucket: str, keys: List[str]) -> List[str]:
    """
    根据 endpoint 与 bucket 构造公开访问 URL 列表。
    - 若 endpoint 主机名已包含 bucket（例如 user 给的是 'https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com'），直接用该 endpoint 作为 base；
    - 否则按照 {scheme}://{bucket}.{netloc} 构造 base。
    对 object key 做 urllib.parse.quote 编码（保留 '/').
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
    高层函数：列出 prefix 下的所有 mseed（去重），返回可访问 URL 列表。
    """
    # 确保 prefix 是解码后的路径（若用户不慎传了 URL 编码，先解码）
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

# url_list = [
#     "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/%E5%9C%B0%E8%B4%A8%E7%81%BE%E5%AE%B3-%E6%96%87%E6%9C%AC%2B%E9%81%A5%E6%84%9F%2B%E6%97%B6%E5%BA%8F/IRIS/dataset_earthquake/IU_AFI_-13.91_-171.78_waveforms.mseed",
#     "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/%E5%9C%B0%E8%B4%A8%E7%81%BE%E5%AE%B3-%E6%96%87%E6%9C%AC%2B%E9%81%A5%E6%84%9F%2B%E6%97%B6%E5%BA%8F/IRIS/dataset_earthquake/IU_CASY_-66.28_110.54_waveforms.mseed",
# ]
url_list = build_mseed_url_list(ENDPOINT, BUCKET, PREFIX, None, None, max_depth=None)

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
    try:
        # 新版pandas兼容方式
        out = df.groupby("minute", group_keys=False).apply(_sampler, include_groups=False)
    except TypeError:
        # 旧版pandas兜底方式
        out = df.groupby("minute", group_keys=False).apply(_sampler)
    #out = df.groupby("minute", group_keys=False).apply(_sampler)
    # 确保minute列没有丢失
    if "minute" not in out.columns and "minute" in df.columns:
        out = out.reset_index()
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
elev_scale = 1000.0

# 半径（米）：全局地理范围大时可适当加大
radius_m = 100000

tooltip = {
    "html": "<b>{net}.{sta}</b><br/>amp: {amplitude}<br/>lat: {lat}, lon: {lon}<br/>{minute} slot {slot}",
    "style": {"color": "white"},
}

layer = pdk.Layer(
    "ColumnLayer",
    data=df_now,
    get_position="[lon, lat]",     # 注意顺序：经度在前
    get_elevation="height",
    elevation_scale=elev_scale,     # 调整高度
    radius=radius_m,
    get_fill_color="[r, g, b, 180]",
    pickable=True,
    auto_highlight=True,
)

# 视角：以所有站点中心为基准
view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=2,
    pitch=45,
    bearing=0,
)

# 使用 map_style=None（不需要 token）
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
