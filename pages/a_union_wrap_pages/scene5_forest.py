# app.py
"""
Streamlit page: List JPGs from Huawei OBS, read EXIF GPS, and plot trajectory (polyline + points) sorted by DJI index
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
# Configuration (can be modified to read from environment variables or Streamlit UI)
# -----------------------
ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"
BUCKET = "gaoyuan-49d0"
PREFIX = "森林数据-rgb+激光雷达/海珠湿地公园L2激光雷达数据20250310/海珠湿地公园1号测区正射/DCIM/DJI_202503101211_001_湿地1号场地/"

# -----------------------
# Helper function: OBS list -> return object keys (excluding bucket)
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
        # pagination check
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
    Return [(object_key, public_url), ...], only containing image extensions (case-insensitive).
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
# EXIF -> GPS parsing
# -----------------------
# build mapping tables once
TAG_LABELS = ExifTags.TAGS
GPSTAGS = ExifTags.GPSTAGS


def _ratio_to_float(r) -> float:
    """
    Compatible with common formats in PIL such as IFDRational or tuple((num,den)).
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
    Extract (lat, lon, datetime_str) from PIL.Image; if no GPS return (None, None, None)
    """
    try:
        exif_raw = img._getexif()
        if not exif_raw:
            return None, None, None
        exif = {}
        for tag_id, value in exif_raw.items():
            tag = TAG_LABELS.get(tag_id, tag_id)
            exif[tag] = value
        # original datetime tag
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
# Download from URL and parse EXIF (with cache)
# -----------------------
@st.cache_data(show_spinner=True)
def fetch_image_and_extract_gps(url: str) -> Dict[str, Any]:
    """
    Download image and return dict: {url, success(bool), lat, lon, datetime, error}
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
# Parse DJI filename index (try to be compatible with multiple naming formats)
# -----------------------
def parse_dji_index_from_basename(basename: str) -> Optional[int]:
    """
    Example: DJI_20250310121732_0003_D.JPG -> index = 3
    Return None if cannot parse
    """
    # common pattern
    m = re.search(r'DJI_(?P<dt>\d{14})_(?P<index>\d+)_', basename, flags=re.I)
    if m:
        return int(m.group("index"))
    # fallback: last number after underscore before file extension
    m2 = re.search(r'_(?P<index>\d+)(?:\.[^.]+)$', basename)
    if m2:
        return int(m2.group("index"))
    return None


# -----------------------
# Main workflow: Display in Streamlit page
# -----------------------
def run_scene5():
    st.set_page_config(page_title="OBS JPG EXIF 轨迹绘制", layout="wide")
    st.title("📸 从 OBS 图片 EXIF 绘制轨迹（折线 + 点）")

    st.markdown(
        "说明：脚本会列出指定 OBS 前缀下的 JPG/JPEG 文件，读取 EXIF GPS，然后按 DJI index 排序并绘制轨迹。"
    )

    # Allow user to override default config via UI
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

            # build base table: key / url / basename / index
            rows = []
            for key, url in pairs:
                basename = os.path.basename(key)
                idx = parse_dji_index_from_basename(basename)
                rows.append({"key": key, "url": url, "basename": basename, "index": idx})

            df_candidates = pd.DataFrame(rows)
            
            # --- new logic start ---
            # 1. Filter out files with valid index
            df_indexed = df_candidates.dropna(subset=['index']).copy()
            if df_indexed.empty:
                st.error("在指定路径下，没有找到任何文件名符合 'DJI_..._NNNN_...' 格式的图片。")
                st.dataframe(df_candidates) # show all found files for debugging
                st.stop()
            
            df_indexed['index'] = df_indexed['index'].astype(int)

            # 2. Create a mapping from index to file info for quick lookup
            # If duplicate index exists, keep the first one
            indexed_files_map = {
                r['index']: r.to_dict()
                for _, r in df_indexed.sort_values('index').iterrows()
            }

            meta_rows = []
            skipped = []

            # 3. Find the starting point of the sequence (minimum index)
            current_index = df_indexed['index'].min()
            start_index = current_index

            st.info(f"发现有效DJI图片文件，将从最小索引 index={start_index} 开始连续读取...")

            # 4. Process strictly in consecutive order until sequence breaks
            while current_index in indexed_files_map:
                file_data = indexed_files_map[current_index]
                basename = file_data["basename"]
                url = file_data["url"]
                
                # st.text(f"正在读取图片: {basename} (index={current_index})")

                # Download and parse EXIF
                meta = fetch_image_and_extract_gps(url)
                if not meta["success"] or meta["lat"] is None or meta["lon"] is None:
                    skipped.append({"basename": basename, "error": meta.get("error", "no GPS EXIF")})
                    # If a file in sequence has no GPS, we can either break or skip
                    # Here we choose to break, since trajectory requires continuous location info
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
                
                # prepare to check next consecutive index
                current_index += 1
            
            st.success(f"图片序列读取完成。序列从 index {start_index} 开始，在 index {current_index - 1} 之后中断，因为未在 OBS 中找到 index 为 {current_index} 的文件或其中有文件GPS信息缺失。")
            # --- new logic end ---

            if not meta_rows:
                st.error("没有任何图片被成功处理（序列为空或第一个文件就失败了）。")
                if skipped:
                    st.dataframe(pd.DataFrame(skipped))
                st.stop()

            df = pd.DataFrame(meta_rows)
            # df is already strictly ordered by index, no need to sort again
            df["order"] = np.arange(len(df))  # ensure sequence order

            # map center
            center_lat = float(df["lat"].mean())
            center_lon = float(df["lon"].mean())

            # build path (longitude first)
            path = [[float(lon), float(lat)] for lon, lat in zip(df["lon"], df["lat"])]

            # pydeck layers (no change needed)
            layers = []

            # polyline (trajectory)
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

            # point layer (clickable to view info)
            df_points = df.copy()
            # label for tooltip
            df_points["label"] = df_points["basename"].astype(str) + " | idx:" + df_points["index"].fillna(-1).astype(int).astype(str)
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_points,
                    get_position="[lon, lat]",
                    get_radius=50,  # radius (meters)
                    radius_min_pixels=4,
                    radius_max_pixels=12,
                    get_fill_color=[200, 30, 30],
                    pickable=True,
                    auto_highlight=True,
                )
            )

            # text labels (show sequence number or time)
            df_points["label_text"] = df_points["order"].astype(str)  # use order as point index
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