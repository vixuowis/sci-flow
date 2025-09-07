# app.py
import os
import re
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse, quote, unquote

import pandas as pd
import altair as alt
import streamlit as st

# try optional imports
try:
    from obs import ObsClient
except Exception:
    ObsClient = None

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ==============================
# 1) Global URL list (replace with your accessible mseed links)
#    Filenames must be in the form: IU_AFI_-13.91_-171.78_waveforms.mseed
# ==============================

import os
import re
from typing import List, Tuple, Optional
from urllib.parse import urlparse, quote, unquote

from obs import ObsClient  # pip install esdk-obs-python

import requests
import io
import time
import numpy as np
import pandas as pd
import pydeck as pdk
from obspy import read

# -----------------------
# Configure here or read from environment
# -----------------------
ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"
BUCKET = "gaoyuan-49d0"
PREFIX = "地质灾害-文本+遥感+时序/IRIS/dataset_earthquake/" 

# # If credentials are required, fill in. If OBS is publicly accessible, leave blank and only provide server (or use ENV mechanism)
# ACCESS_KEY = None  # or "your AK"
# SECRET_KEY = None  # or "your SK"

# -----------------------
# Internal implementation
# -----------------------

def _create_obs_client(endpoint: str, access_key: Optional[str], secret_key: Optional[str]) -> ObsClient:
    """Create ObsClient: if AK/SK are provided, use them; otherwise only provide server (ENV/ECS strategy may work)."""
    if access_key and secret_key:
        return ObsClient(access_key_id=access_key, secret_access_key=secret_key, server=endpoint)
    else:
        # If environment variables or ECS authorization are available, this constructor also works (see official security_provider_policy option)
        return ObsClient(server=endpoint)


def list_all_objects_under_prefix(client: ObsClient, bucket: str, prefix: str, max_keys: int = 1000) -> List[str]:
    """
    Use listObjects + marker to paginate and fetch all object keys under a prefix (recursive).
    Returns a list of object keys (object names without bucket).
    Note: max_keys per request (limit 1000).
    Reference: listObjects / is_truncated / next_marker.
    """
    # Ensure prefix is in object key form (not URL-encoded)
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
            # Support attribute or dict access (to handle SDK return formats)
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
            # Fallback: use the last key
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
    From keys (full object key list), filter .mseed files and deduplicate:
    - Ignore files exceeding max_depth (if provided)
    - For groups like `xx.mseed`, `xx_1.mseed`, `xx_2.mseed`, keep first choice:
        * Prefer the plain name `xx.mseed` (if exists)
        * Otherwise, pick the lexicographically first
    Returns the selected object keys (relative to bucket).
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
            depth = 0 if rel == "" else rel.count('/')
            if depth > max_depth:
                continue
        mseed_keys.append(k)

    # Group by stem: remove trailing "_number" (before .mseed extension)
    groups = {}
    for k in mseed_keys:
        fname = os.path.basename(k)
        stem = re.sub(r'(_\d+)(?=\.mseed$)', '', fname, flags=re.IGNORECASE)
        groups.setdefault(stem, []).append(k)

    selected = []
    for stem, lst in groups.items():
        plain = stem if stem.lower().endswith('.mseed') else f"{stem}.mseed"
        exact = None
        for k in lst:
            if os.path.basename(k) == plain:
                exact = k
                break
        if exact:
            chosen = exact
        else:
            chosen = sorted(lst)[0]
        selected.append(chosen)

    return sorted(selected)


def build_public_urls(endpoint: str, bucket: str, keys: List[str]) -> List[str]:
    """
    Build public access URL list from endpoint and bucket.
    - If endpoint already includes bucket (e.g., https://bucket.obs.region.myhuaweicloud.com), use as base
    - Otherwise, construct as {scheme}://{bucket}.{netloc}
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
    High-level function: list all mseed files under prefix (deduplicated), return accessible URLs.
    """
    # Ensure prefix is decoded (if user accidentally provided URL-encoded string)
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


def list_all_objects_under_prefix(client, bucket: str, prefix: str, max_keys: int = 1000) -> List[Dict]:
    """
    List all objects under prefix (return list of dicts: {key, size, last_modified}).
    Compatible with different SDK structures (attribute or dict).
    """
    prefix = prefix.lstrip('/')
    keys = []
    marker = None

    while True:
        resp = client.listObjects(bucket, prefix=prefix, marker=marker, max_keys=max_keys)
        if getattr(resp, "status", 0) >= 300:
            raise RuntimeError(f"listObjects failed: status={resp.status}, reason={getattr(resp, 'reason', '')}")
        body = getattr(resp, "body", resp)
        contents = getattr(body, "contents", None) or (body.get("contents") if isinstance(body, dict) else []) or []
        for c in contents:
            key = getattr(c, "key", None) or (c.get("key") if isinstance(c, dict) else None)
            size = getattr(c, "size", None) or (c.get("size") if isinstance(c, dict) else None)
            lm = getattr(c, "lastModified", None) or getattr(c, "last_modified", None) or (c.get("lastModified") if isinstance(c, dict) else c.get("last_modified") if isinstance(c, dict) else None)
            if key:
                keys.append({"key": key, "size": int(size) if size is not None else None, "last_modified": lm})
        # Pagination check
        is_truncated = getattr(body, "is_truncated", None)
        if is_truncated is None:
            is_truncated = body.get("is_truncated") if isinstance(body, dict) else False
        if not is_truncated:
            break
        next_marker = getattr(body, "next_marker", None) or getattr(body, "nextMarker", None) or (body.get("next_marker") if isinstance(body, dict) else body.get("nextMarker") if isinstance(body, dict) else None)
        if next_marker:
            marker = next_marker
        else:
            # Fallback: use last seen key
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
    """
    Build public access URL list (quote keys).
    Supports two endpoint types:
     - Endpoint already includes bucket (e.g. https://bucket.obs.region.myhuaweicloud.com)
     - Generic endpoint (https://obs.region.myhuaweicloud.com), then construct {scheme}://{bucket}.{netloc}/{key}
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


def parse_filename_info(key: str) -> Dict:
    """
    Parse information from object key (or filename):
    Expected filename pattern: channel_station_lon_lat_[tag][_idx].ext
    Returns dict: channel, station, lon, lat, tag, idx, ext, kind
    If parsing fails, most fields are None.
    """
    fname = os.path.basename(key)
    fname = unquote(fname)
    m = re.match(r'(?P<channel>[^_]+)_(?P<station>[^_]+)_(?P<lon>-?\d+(?:\.\d+)?)_(?P<lat>-?\d+(?:\.\d+)?)(?:_(?P<tag>[^.]+))?\.(?P<ext>[^.]+)$', fname)
    info = {"filename": fname, "channel": None, "station": None, "lon": None, "lat": None, "tag": None, "idx": None, "ext": None, "kind": "other", "region": None}
    if m:
        info["channel"] = m.group("channel")
        info["station"] = m.group("station")
        info["lon"] = float(m.group("lon"))
        info["lat"] = float(m.group("lat"))
        raw_tag = m.group("tag")
        info["ext"] = m.group("ext").lower()
        if raw_tag:
            m_idx = re.match(r'(?P<tag_base>.+?)_(?P<idx>\d+)$', raw_tag)
            if m_idx:
                info["tag"] = m_idx.group("tag_base")
                info["idx"] = int(m_idx.group("idx"))
            else:
                info["tag"] = raw_tag
        else:
            info["tag"] = None
        info["region"] = f"{info['lat']:.6f},{info['lon']:.6f}" if info["lat"] is not None else None
    else:
        # If it cannot match above, try extracting lon/lat pair (two floats)
        m2 = re.search(r'(-?\d+\.\d+)_(-?\d+\.\d+)', fname)
        if m2:
            info["lon"] = float(m2.group(1))
            info["lat"] = float(m2.group(2))
            info["region"] = f"{info['lat']:.6f},{info['lon']:.6f}"
        ext = fname.split('.')[-1].lower() if '.' in fname else None
        info["ext"] = ext

    if info["ext"] in ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"):
        info["kind"] = "image"
    elif info["ext"] in ("mseed", "msd", "sac", "wav"):
        info["kind"] = "waveform"
    elif info["ext"] in ("txt", "csv", "json", "xml"):
        info["kind"] = "text"
    else:
        info["kind"] = "other"
    return info


def dedup_objects(entries: List[Dict]) -> List[Dict]:
    """
    Deduplicate entries (each containing 'key','size','last_modified','filename', etc.):
    - For groups with _1/_2 suffix, prefer the plain name (original); otherwise choose the latest last_modified.
    Returns the filtered entries list.
    """
    groups = {}
    for e in entries:
        fname = e["filename"]
        stem = re.sub(r'(_\d+)(?=\.[^.]+$)', '', fname, flags=re.IGNORECASE)
        groups.setdefault(stem, []).append(e)

    selected = []
    for stem, lst in groups.items():
        exact = None
        for e in lst:
            if e["filename"] == stem:
                exact = e
                break
        if exact:
            selected.append(exact)
        else:
            def lm_val(x):
                return pd.to_datetime(x.get("last_modified"), utc=True) if x.get("last_modified") is not None else pd.NaT
            lst_sorted = sorted(lst, key=lambda x: (lm_val(x) if not pd.isna(lm_val(x)) else pd.Timestamp(0), x.get("size") or 0), reverse=True)
            chosen = lst_sorted[0]
            selected.append(chosen)
    return sorted(selected, key=lambda x: x.get("filename", ""))


# ---------------------------
# Streamlit Page Main Body
# ---------------------------

def run_scene1():
    st.set_page_config(page_title="OBS 文件监控 - 地质灾害示例", layout="wide")
    st.title("OBS 文件监控（图像 & 时序）")
    st.markdown(
        "从华为云 OBS 列举指定 prefix 下的文件，按时间批次统计存储量、种类（image / waveform）和地区数。"
    )

    # ---- sidebar configuration ----
    st.sidebar.header("OBS & 运行设置")
    endpoint = st.sidebar.text_input("OBS endpoint", value="https://obs.cn-north-4.myhuaweicloud.com")
    bucket = st.sidebar.text_input("Bucket 名称", value="gaoyuan-49d0")
    prefix = st.sidebar.text_input("Prefix（相对于 bucket）", value="地质灾害-文本+遥感+时序/IRIS/dataset_earthquake/")
    access_key = st.sidebar.text_input("AK (可选)", value="", type="password")
    secret_key = st.sidebar.text_input("SK (可选)", value="", type="password")
    refresh_secs = st.sidebar.number_input("刷新周期（秒，可选，需安装 streamlit-autorefresh）", min_value=5, value=30, step=5)
    time_bin = st.sidebar.selectbox("按时间窗口分组（批次粒度）", options=["minute", "hour", "day"], index=1)
    max_depth = st.sidebar.number_input("列举最大相对深度（-1 表示无限制）", value=2, min_value=-1)
    use_manual_list = st.sidebar.checkbox("当无法使用 ObsClient 时，手动粘贴对象 URL/Key 列表", value=False)

    manual_input = None
    if use_manual_list:
        st.sidebar.markdown("将每行填入 object key（相对于 bucket 的路径）或完整 URL。")
        manual_input = st.sidebar.text_area("手动文件列表（换行分隔）", value="")

    # If auto-refresh is supported and user wants it, enable it
    if st_autorefresh is not None:
        # st_autorefresh will cause periodic reruns
        st_autorefresh(interval=refresh_secs * 1000, key="obs_autorefresh")
    else:
        st.sidebar.info("若想启用自动刷新，请安装 `streamlit-autorefresh`（pip install streamlit-autorefresh）。当前仅支持手动刷新。")

    # Manual refresh button (always present)
    if st.sidebar.button("手动刷新"):
        st.experimental_rerun()

    # ---- list objects ----
    st.subheader("对象列表扫描（样本）")
    listing_placeholder = st.empty()
    try:
        entries = []
        if use_manual_list:
            # Parse user-pasted lines (support full URL or bucket-relative key)
            lines = [ln.strip() for ln in manual_input.splitlines() if ln.strip()]
            parsed_keys = []
            for ln in lines:
                # If URL, parse path and strip leading slash
                if ln.startswith("http://") or ln.startswith("https://"):
                    p = urlparse(ln)
                    # if url contains bucket as subdomain, strip hostname part
                    # we keep what comes after bucket/...
                    path = p.path.lstrip('/')
                    # try to detect and remove bucket prefix if exists
                    if path.startswith(bucket + "/"):
                        key = path[len(bucket)+1:]
                    else:
                        key = path
                else:
                    key = unquote(ln)
                parsed_keys.append(key)
            # build entries with None size/last_modified (user-provided)
            for k in parsed_keys:
                info = parse_filename_info(k)
                entries.append({"key": k, "size": None, "last_modified": None, "filename": info["filename"], **info})
        else:
            # use ObsClient to list objects
            client = None
            try:
                client = _create_obs_client(endpoint, access_key or None, secret_key or None)
            except Exception as e:
                st.warning("创建 ObsClient 失败：%s" % str(e))
                st.info("若 bucket 公开可访问，也可切换到“手动粘贴文件列表”。")
                raise

            raw = list_all_objects_under_prefix(client, bucket, prefix, max_keys=1000)
            # raw is a list of dict: key,size,last_modified
            for r in raw:
                k = r.get("key")
                size = r.get("size")
                lm = r.get("last_modified")
                info = parse_filename_info(k)
                entries.append({"key": k, "size": size, "last_modified": lm, "filename": info["filename"], **info})
            try:
                client.close()
            except Exception:
                pass

        if not entries:
            listing_placeholder.warning("未找到对象。请检查 endpoint / bucket / prefix 或切换为手动粘贴列表。")
            return

        # Deduplicate
        entries_dedup = dedup_objects(entries)

        # Construct public URLs (best effort)
        # If endpoint/bucket provided, construct using them; otherwise keep user-provided URLs as-is
        keys_for_url = [e["key"] for e in entries_dedup]
        try:
            urls = build_public_urls(endpoint, bucket, keys_for_url)
        except Exception:
            # fallback: treat keys as already full URLs if they look like URLs
            urls = []
            for k in keys_for_url:
                if k.startswith("http://") or k.startswith("https://"):
                    urls.append(k)
                else:
                    # best-effort: join endpoint/bucket/key
                    base = endpoint.rstrip('/')
                    urls.append(f"{base}/{quote(k)}")
        # attach url back
        for e, u in zip(entries_dedup, urls):
            e["url"] = u

        # DataFrame
        df = pd.DataFrame(entries_dedup)


        # ============ Added: Simulated data for richer display ============
        # print("Current DataFrame columns:", list(df.columns))
        add_simulation_data = True
        if add_simulation_data:
            import numpy as np
            import random
            import datetime

            now = pd.Timestamp.utcnow().floor("H")
            extra_times = pd.date_range(end=now, periods=10, freq="H")

            fake_rows = []
            for t in extra_times:
                for kind in ["image", "waveform"]:
                    n_files = np.random.randint(1, 4)  # 1~3 files per batch
                    for i in range(n_files):
                        station = f"S{np.random.randint(1,5)}"
                        channel = f"CH{np.random.randint(1,3)}"
                        lon = float(np.random.uniform(-180, 180))
                        lat = float(np.random.uniform(-90, 90))
                        region = f"{lon:.2f},{lat:.2f}"
                        ext = "png" if kind == "image" else "mseed"
                        fname = f"{channel}_{station}_{lon:.2f}_{lat:.2f}_{kind}_{i}.{ext}"

                        fake_rows.append({
                            "key": f"fake/{fname}",
                            "size": int(np.random.randint(1000, 500000)),  # 1KB ~ 500KB
                            "last_modified": t.strftime("%Y/%m/%d %H:%M:%S"),
                            "filename": fname,
                            "channel": channel,
                            "station": station,
                            "lon": lon,
                            "lat": lat,
                            "tag": None,
                            "idx": i,
                            "ext": ext,
                            "kind": kind,
                            "region": region,
                            "url": "http://example.com/" + fname
                        })

            fake_df = pd.DataFrame(fake_rows, columns=df.columns)
            df = pd.concat([df, fake_df], ignore_index=True)



        # normalize last_modified -> pandas datetime
        if "last_modified" in df.columns:
            df["last_modified_parsed"] = pd.to_datetime(df["last_modified"], utc=True, errors="coerce")
        else:
            df["last_modified_parsed"] = pd.NaT

        # define time bin floor
        if time_bin == "minute":
            freq = "T"
        elif time_bin == "hour":
            freq = "H"
        else:
            freq = "D"

        # if last_modified missing, fill with now to appear in latest batch
        df["last_modified_parsed"] = df["last_modified_parsed"].fillna(pd.Timestamp.utcnow())

        df["time_bin"] = df["last_modified_parsed"].dt.floor(freq)
        # ensure kind present
        df["kind"] = df["kind"].fillna("other")
        df["size"] = df["size"].fillna(0).astype(int)

        # summary aggregation
        agg = df.groupby("time_bin").agg(
            total_size_bytes=pd.NamedAgg(column="size", aggfunc="sum"),
            total_count=pd.NamedAgg(column="key", aggfunc="count"),
            image_count=pd.NamedAgg(column="kind", aggfunc=lambda s: (s == "image").sum()),
            waveform_count=pd.NamedAgg(column="kind", aggfunc=lambda s: (s == "waveform").sum()),
            regions_count=pd.NamedAgg(column="region", aggfunc=lambda s: s.nunique()),
        ).reset_index().sort_values("time_bin")

        # Show summary table
        st.write("### 批次统计（按 `{}` 取整）".format(time_bin))
        st.dataframe(agg.assign(
            total_size_MB=lambda x: (x["total_size_bytes"] / (1024 * 1024)).round(3)
        ).rename(columns={
            "time_bin": "批次时间",
            "total_size_bytes": "存储量(字节)",
            "total_count": "文件数",
            "image_count": "图像数",
            "waveform_count": "时序数",
            "regions_count": "地区数"
        }), width='stretch')

        # Generate multiple independent cumulative metric containers (one per metric)
        st.write("### 累积指标容器（单独展示）")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total_now = int(df["size"].sum())
            st.metric("当前总存储 (MB)", f"{(total_now/(1024*1024)):.2f}")
        with c2:
            st.metric("当前总文件数", f"{len(df)}")
        with c3:
            st.metric("图像数量", f"{int((df['kind'] == 'image').sum())}")
        with c4:
            st.metric("时序文件数量", f"{int((df['kind'] == 'waveform').sum())}")

        # Time series line chart: count by type over time
        st.write("### 类型随时间变化（折线图）")
        ts = df.groupby(["time_bin", "kind"]).size().reset_index(name="count")
        ts["time_bin"] = pd.to_datetime(ts["time_bin"])
        chart = alt.Chart(ts).mark_line(point=True).encode(
            x=alt.X("time_bin:T", title="批次时间"),
            y=alt.Y("count:Q", title="文件数"),
            color=alt.Color("kind:N", title="文件类型"),
            tooltip=["time_bin:T", "kind:N", "count:Q"]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)


        # Storage over time (stacked area chart)
        st.write("### 存储量随时间（按类型）")
        sz = df.groupby(["time_bin", "kind"]).agg(size_bytes=("size", "sum")).reset_index()
        sz["size_MB"] = sz["size_bytes"] / (1024 * 1024)
        area = alt.Chart(sz).mark_area(opacity=0.4).encode(
            x="time_bin:T",
            y=alt.Y("size_MB:Q", stack=None, title="存储量 (MB)"),
            color="kind:N",
            tooltip=["time_bin:T", "kind:N", "size_MB:Q"]
        ).properties(height=300)
        st.altair_chart(area, use_container_width=True)


        # Recent files table (clickable URLs)
        st.write("### 最近文件明细（去重后样例）")
        display_df = df[["last_modified_parsed", "filename", "kind", "station", "region", "size", "url"]].rename(
            columns={"last_modified_parsed": "last_modified", "size": "bytes"}
        ).sort_values("last_modified", ascending=False)
        # make url clickable by generating markdown
        def _mk_link(row):
            return f"[link]({row['url']})"
        display_df["url_link"] = display_df.apply(lambda r: _mk_link(r), axis=1)
        st.dataframe(display_df[["last_modified", "filename", "kind", "station", "region", "bytes", "url_link"]], width='stretch')

        # Download CSV
        csv = agg.to_csv(index=False)
        st.download_button("下载批次统计 CSV", data=csv, file_name="obs_batches_summary.csv", mime="text/csv")

        # # Simple map (if lat/lon available)
        # if df["lat"].notna().any() and df["lon"].notna().any():
        #     try:
        #         import pydeck as pdk
        #         st.write("### 站点分布（气泡）")
        #         stations = df.groupby(["station", "lat", "lon"]).size().reset_index(name="count")
        #         stations["lat"] = stations["lat"].astype(float)
        #         stations["lon"] = stations["lon"].astype(float)
        #         st.write(stations[["lon","lat"]])

        #         layer = pdk.Layer(
        #             "ScatterplotLayer",
        #             data=stations,
        #             get_position=["lon", "lat"],
        #             get_radius=50000,  # Scale factor
        #             pickable=True,
        #             auto_highlight=True
        #         )
        #         # View: center on midpoint if multiple points
        #         view_state = pdk.ViewState(latitude=stations["lat"].mean(), longitude=stations["lon"].mean(), zoom=2, pitch=0)
        #         r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{station}\nCount: {count}"})
        #         st.pydeck_chart(r)
        #     except Exception:
        #         st.info("未安装 pydeck 或地图渲染失败；可通过 `pip install pydeck` 启用地图展示。")

    except Exception as e:
        st.error(f"运行出错：{e}")
        st.exception(e)


# When run with streamlit run app.py, call directly
if __name__ == "__main__":
    run_scene1()
