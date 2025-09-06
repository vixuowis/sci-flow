# app.py
"""
Streamlit 页面：从华为云 OBS 列出 JPG，读取 EXIF GPS 并按 DJI index 排序绘制轨迹（折线 + 点）
"""

import io
import os
import re
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse, quote, unquote

import requests
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from PIL import Image, ExifTags

# OBS SDK
from obs import ObsClient  # pip install esdk-obs-python

# -----------------------
# 配置（可改为从环境变量读取或通过 Streamlit UI 修改）
# -----------------------
ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"
BUCKET = "gaoyuan-49d0"
PREFIX = "森林数据-rgb+激光雷达/海珠湿地公园L2激光雷达数据20250310/海珠湿地公园1号测区正射/DCIM/DJI_202503101211_001_湿地1号场地/"

# -----------------------
# 辅助函数：OBS 列表 -> 返回 object keys（不含 bucket）
# -----------------------
def _create_obs_client(endpoint: str, access_key: Optional[str], secret_key: Optional[str]) -> ObsClient:
    if access_key and secret_key:
        return ObsClient(access_key_id=access_key, secret_access_key=secret_key, server=endpoint)
    else:
        return ObsClient(server=endpoint)


def list_all_objects_under_prefix(client: ObsClient, bucket: str, prefix: str, max_keys: int = 1000) -> List[str]:
    prefix = prefix.lstrip('/')
    keys: List[str] = []
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
        # 翻页判断
        if not getattr(body, "is_truncated", False):
            break
        next_marker = getattr(body, "next_marker", None) or getattr(body, "nextMarker", None)
        if next_marker:
            marker = next_marker
        else:
            if contents:
                last = contents[-1]
                last_key = getattr(last, "key", None) or (last.get("key") if isinstance(last, dict) else None)
                if last_key:
                    marker = last_key
                else:
                    break
            else:
                break
    return keys


def build_public_urls(endpoint: str, bucket: str, keys: List[str]) -> List[str]:
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


def build_image_key_url_list(
    endpoint: str,
    bucket: str,
    prefix: str,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    exts: List[str] = (".jpg", ".jpeg"),
) -> List[Tuple[str, str]]:
    """
    返回 [(object_key, public_url), ...]，只含图片扩展名（不区分大小写）。
    """
    prefix = unquote(prefix)
    client = _create_obs_client(endpoint, access_key, secret_key)
    try:
        keys = list_all_objects_under_prefix(client, bucket, prefix)
    finally:
        try:
            client.close()
        except Exception:
            pass

    img_keys = [k for k in keys if os.path.splitext(k)[1].lower() in exts]
    urls = build_public_urls(endpoint, bucket, img_keys)
    return list(zip(img_keys, urls))


# -----------------------
# EXIF -> GPS 解析
# -----------------------
# build mapping tables once
TAG_LABELS = ExifTags.TAGS
GPSTAGS = ExifTags.GPSTAGS


def _ratio_to_float(r) -> float:
    """
    兼容 PIL 里常见的 IFDRational 或 tuple((num,den)) 等格式。
    """
    try:
        # IFDRational has numerator, denominator
        if hasattr(r, "numerator") and hasattr(r, "denominator"):
            return float(r.numerator) / float(r.denominator) if r.denominator != 0 else float(r.numerator)
        # tuple like (num, den)
        if isinstance(r, (tuple, list)) and len(r) >= 2:
            num, den = r[0], r[1]
            return float(num) / float(den) if den != 0 else float(num)
        # numeric or string
        return float(r)
    except Exception:
        try:
            return float(str(r))
        except Exception:
            return 0.0


def _dms_to_dd(dms) -> float:
    """
    dms expected like [(deg_num,deg_den),(min_num,min_den),(sec_num,sec_den)] or IFDRational objects
    """
    try:
        d = _ratio_to_float(dms[0])
        m = _ratio_to_float(dms[1])
        s = _ratio_to_float(dms[2])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return 0.0


def extract_gps_from_pil_image(img: Image.Image) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    从 PIL.Image 中提取 (lat, lon, datetime_str)；若无 GPS 返回 (None, None, None)
    """
    try:
        exif_raw = img._getexif()
        if not exif_raw:
            return None, None, None
        exif = {}
        for tag_id, value in exif_raw.items():
            tag = TAG_LABELS.get(tag_id, tag_id)
            exif[tag] = value
        # datetime 原始标签
        dt = exif.get("DateTimeOriginal") or exif.get("DateTime") or None

        gps_raw = exif.get("GPSInfo")
        if not gps_raw:
            return None, None, dt

        gps = {}
        for t, v in gps_raw.items():
            name = GPSTAGS.get(t, t)
            gps[name] = v

        lat = None
        lon = None
        lat_ref = gps.get("GPSLatitudeRef")
        lon_ref = gps.get("GPSLongitudeRef")
        lat_vals = gps.get("GPSLatitude")
        lon_vals = gps.get("GPSLongitude")
        if lat_vals and lon_vals and lat_ref and lon_ref:
            lat = _dms_to_dd(lat_vals)
            lon = _dms_to_dd(lon_vals)
            if isinstance(lat_ref, bytes):
                lat_ref = lat_ref.decode('utf-8', errors='ignore')
            if isinstance(lon_ref, bytes):
                lon_ref = lon_ref.decode('utf-8', errors='ignore')
            if str(lat_ref).upper() in ("S", "SOUTH"):
                lat = -abs(lat)
            if str(lon_ref).upper() in ("W", "WEST"):
                lon = -abs(lon)
            return lat, lon, dt
        else:
            return None, None, dt
    except Exception:
        return None, None, None


# -----------------------
# 从 URL 下载并解析 EXIF（缓存）
# -----------------------
@st.cache_data(show_spinner=True)
def fetch_image_and_extract_gps(url: str) -> Dict[str, Any]:
    """
    下载图片并返回 dict: {url, success(bool), lat, lon, datetime, error}
    """
    out = {"url": url, "success": False, "lat": None, "lon": None, "datetime": None, "error": None}
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        lat, lon, dt = extract_gps_from_pil_image(img)
        out.update({"success": True, "lat": lat, "lon": lon, "datetime": dt})
        if lat is None or lon is None:
            out["success"] = False
            out["error"] = "no GPS EXIF"
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


# -----------------------
# 解析 DJI 文件名中的 index（尽量兼容多种命名）
# -----------------------
def parse_dji_index_from_basename(basename: str) -> Optional[int]:
    """
    例如： DJI_20250310121732_0003_D.JPG -> index = 3
    若无法解析返回 None
    """
    # 常见模式
    m = re.search(r'DJI_(?P<dt>\d{14})_(?P<index>\d+)_', basename, flags=re.I)
    if m:
        return int(m.group("index"))
    # 退而求其次：最后一个下划线后的数字（文件扩展名前）
    m2 = re.search(r'_(?P<index>\d+)(?:\.[^.]+)$', basename)
    if m2:
        return int(m2.group("index"))
    return None


# -----------------------
# 主流程：在 Streamlit 页面中展示
# -----------------------
def run_scene5():
    st.set_page_config(page_title="OBS JPG EXIF 轨迹绘制", layout="wide")
    st.title("📸 从 OBS 图片 EXIF 绘制轨迹（折线 + 点）")

    st.markdown(
        "说明：脚本会列出指定 OBS 前缀下的 JPG/JPEG 文件，读取 EXIF GPS，然后按 DJI index 排序并绘制轨迹。"
    )

    # 允许用户在 UI 中覆盖默认配置
    with st.expander("OBS 配置（可在此覆盖）", expanded=False):
        endpoint = st.text_input("OBS Endpoint", value=ENDPOINT)
        bucket = st.text_input("Bucket", value=BUCKET)
        prefix = st.text_input("Prefix (object key prefix)", value=PREFIX)


    if st.button("列出并加载图片 EXIF"):
        with st.spinner("从 OBS 列出文件并读取 EXIF（可能较慢，视图片数量而定）..."):
            try:
                pairs = build_image_key_url_list(endpoint, bucket, prefix, None, None)
            except Exception as e:
                st.error(f"列出 OBS 对象失败：{e}")
                st.stop()

            if not pairs:
                st.info("在该 prefix 下未找到 JPG/JPEG 文件。请检查前缀或 OBS 权限。")
                st.stop()

            # 构造基础表格：key / url / basename / index
            rows = []
            for key, url in pairs:
                basename = os.path.basename(key)
                idx = parse_dji_index_from_basename(basename)
                rows.append({"key": key, "url": url, "basename": basename, "index": idx})

            df_candidates = pd.DataFrame(rows)
            
            # --- 新逻辑开始 ---
            # 1. 筛选出能成功解析出 index 的文件
            df_indexed = df_candidates.dropna(subset=['index']).copy()
            if df_indexed.empty:
                st.error("在指定路径下，没有找到任何文件名符合 'DJI_..._NNNN_...' 格式的图片。")
                st.dataframe(df_candidates) # 显示所有找到的文件以供调试
                st.stop()
            
            df_indexed['index'] = df_indexed['index'].astype(int)

            # 2. 创建一个从 index 到文件信息的映射，用于快速查找
            # 如果有重复的 index，只保留第一个
            indexed_files_map = {
                r['index']: r.to_dict()
                for _, r in df_indexed.sort_values('index').iterrows()
            }

            meta_rows = []
            skipped = []

            # 3. 找到序列的起始点（最小的 index）
            current_index = df_indexed['index'].min()
            start_index = current_index

            st.info(f"发现有效DJI图片文件，将从最小索引 index={start_index} 开始连续读取...")

            # 4. 严格按连续索引处理，直到序列中断
            while current_index in indexed_files_map:
                file_data = indexed_files_map[current_index]
                basename = file_data["basename"]
                url = file_data["url"]
                
                # st.text(f"正在读取图片: {basename} (index={current_index})")

                # 下载并解析 EXIF
                meta = fetch_image_and_extract_gps(url)
                if not meta["success"] or meta["lat"] is None or meta["lon"] is None:
                    skipped.append({"basename": basename, "error": meta.get("error", "no GPS EXIF")})
                    # 如果序列中的某个文件没有GPS，可以选择中断或跳过
                    # 此处选择中断，因为轨迹需要连续的位置信息
                    st.warning(f"图片 {basename} 无法获取有效GPS信息，序列处理中断。")
                    break
                
                meta_rows.append(
                    {
                        "basename": basename,
                        "key": file_data["key"],
                        "url": url,
                        "index": int(current_index),
                        "lat": float(meta["lat"]),
                        "lon": float(meta["lon"]),
                        "datetime": meta["datetime"],
                    }
                )
                
                # 准备检查下一个连续的 index
                current_index += 1
            
            st.success(f"图片序列读取完成。序列从 index {start_index} 开始，在 index {current_index - 1} 之后中断，因为未在 OBS 中找到 index 为 {current_index} 的文件或其中有文件GPS信息缺失。")
            # --- 新逻辑结束 ---

            if not meta_rows:
                st.error("没有任何图片被成功处理（序列为空或第一个文件就失败了）。")
                if skipped:
                    st.dataframe(pd.DataFrame(skipped))
                st.stop()

            df = pd.DataFrame(meta_rows)
            # 此时的 df 已经是按 index 严格排序的了，无需再次排序
            df["order"] = np.arange(len(df))  # 用于确保顺序

            # 地图中心
            center_lat = float(df["lat"].mean())
            center_lon = float(df["lon"].mean())

            # 构造路径（经度在前）
            path = [[float(lon), float(lat)] for lon, lat in zip(df["lon"], df["lat"])]

            # pydeck layers (这部分代码无需改动)
            layers = []

            # 折线（路径）
            if len(path) >= 2:
                path_data = [{"path": path, "name": "trajectory"}]
                layers.append(
                    pdk.Layer(
                        "PathLayer",
                        data=path_data,
                        get_path="path",
                        get_width=5,
                        width_min_pixels=2,
                        rounded=True,
                        get_color=[0, 120, 200],
                        pickable=False,
                    )
                )

            # 点层（可点击查看信息）
            df_points = df.copy()
            # label 用于 tooltip
            df_points["label"] = df_points["basename"].astype(str) + " | idx:" + df_points["index"].fillna(-1).astype(int).astype(str)
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_points,
                    get_position="[lon, lat]",
                    get_radius=50,  # 半径（米）
                    radius_min_pixels=4,
                    radius_max_pixels=12,
                    get_fill_color=[200, 30, 30],
                    pickable=True,
                    auto_highlight=True,
                )
            )

            # 文本标签（显示序号或时间）
            df_points["label_text"] = df_points["order"].astype(str)  # 用 order 作为点编号
            layers.append(
                pdk.Layer(
                    "TextLayer",
                    data=df_points,
                    get_position="[lon, lat]",
                    get_text="label_text",
                    get_size=14,
                    get_angle=0,
                    get_alignment_baseline="'bottom'",
                    pickable=False,
                )
            )

            tooltip = {
                "html": "<b>{basename}</b><br/>{datetime}<br/>idx: {index} <br/>lat: {lat} , lon: {lon}",
                "style": {"color": "white"},
            }

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=14 if len(df) < 100 else 10,
                pitch=45,
                bearing=0,
            )

            deck = pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None, tooltip=tooltip)
            st.pydeck_chart(deck, use_container_width=True)

            with st.expander("轨迹点数据（按 index 排序）", expanded=True):
                st.dataframe(df[["basename", "index", "order", "lat", "lon", "datetime"]].reset_index(drop=True))

            if skipped:
                st.warning(f"处理过程中有 {len(skipped)} 张图片被跳过或导致中断。")
                st.dataframe(pd.DataFrame(skipped))

    else:
        st.info("点击上方按钮“列出并加载图片 EXIF”开始。")

if __name__ == "__main__":
    run_scene5()