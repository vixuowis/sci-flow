import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta
from collections import deque
import io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# -------------------- Helpers --------------------

def _now():
    return datetime.now()

# 高原地区
REGIONS = ["青藏高原", "帕米尔高原", "安第斯高原", "云贵高原", "伊朗高原"]

# realistic ranges for plateau metrics
TEMP_RANGE = (-15.0, 20.0)       # deg C (高原冷凉)
PRECIP_RANGE = (0.0, 20.0)       # mm/日
VEG_RANGE = (0.1, 0.8)           # NDVI 植被指数

random.seed(42)
np.random.seed(42)

def run_scene3():
    def generate_text_note(region, temp, precip, veg_index):
        notes = [
            "草地退化明显",
            "高寒植被稀疏",
            "冰川融水补给",
            "干旱少雨，土壤贫瘠",
            "局部有植被恢复迹象",
            "气温骤降导致冻害",
        ]
        if veg_index < 0.2:
            return f"{random.choice(notes)}; NDVI:{veg_index:.2f} 极低"
        if precip > 15:
            return f"{random.choice(notes)}; 降水:{precip:.1f} mm"
        return f"{random.choice(notes)}; 气温:{temp:.1f}°C"

    def make_small_image(timestamp: datetime, region: str, temp: float, precip: float, veg_index: float):
        """Create a small PNG in-memory showing plateau visualization + metadata text."""
        W, H = 480, 320
        img = Image.new("RGB", (W, H), color=(245, 240, 230))
        d = ImageDraw.Draw(img)

        # draw mountain-like pattern
        for x in range(0, W, 5):
            height = int(100 + 80 * (1 - veg_index) + 20 * np.sin(x / 30.0))
            d.line([(x, H), (x, H - height)], fill=(120, 100, 80))

        # overlay text
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        text_lines = [
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            f"{region}",
            f"T:{temp:.1f}°C  P:{precip:.1f}mm  NDVI:{veg_index:.2f}",
        ]
        tx, ty = 8, 8
        for line in text_lines:
            d.text((tx, ty), line, fill=(10, 10, 10), font=font)
            ty += 18

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

    def compute_storage_mb(modalities: dict, temp_count=1):
        size = 0.0
        size += 0.02 * temp_count
        if modalities.get("text", False):
            size += 0.002 * 30
        if modalities.get("image", False):
            size += 1.5 + random.random() * 1.0  # 高原图像小一些
        size += 0.01
        return round(size, 3)

    def generate_batch(ts: datetime = None):
        if ts is None:
            ts = _now()
        region = random.choice(REGIONS)
        temp = float(np.random.uniform(*TEMP_RANGE))
        precip = float(np.random.uniform(*PRECIP_RANGE))
        veg_index = float(np.random.uniform(*VEG_RANGE))

        modalities = {
            "sensor": True,
            "image": random.random() < 0.6,
            "text": random.random() < 0.9,
        }

        note = generate_text_note(region, temp, precip, veg_index) if modalities["text"] else ""
        image_bytes = make_small_image(ts, region, temp, precip, veg_index) if modalities["image"] else None
        storage_mb = compute_storage_mb(modalities)

        batch = {
            "timestamp": ts,
            "region": region,
            "temp": temp,
            "precip": precip,
            "veg_index": veg_index,
            "modalities": modalities,
            "note": note,
            "image_bytes": image_bytes,
            "storage_mb": storage_mb,
        }
        return batch

    # -------------------- Session init --------------------
    if "batches" not in st.session_state:
        st.session_state["max_batches_keep"] = 200
        st.session_state["batches"] = deque(maxlen=st.session_state["max_batches_keep"])
        now = _now()
        N_init = 30
        for i in range(N_init, 0, -1):
            ts = now - timedelta(minutes=10 * i)
            st.session_state["batches"].append(generate_batch(ts))

    if "auto" not in st.session_state:
        st.session_state["auto"] = False

    # -------------------- UI --------------------
    st.set_page_config(page_title="高原脆弱场景数据可视化", layout="wide")
    st.title("高原生态环境数据可视化")

    # Controls
    with st.sidebar:
        st.header("控制面板")
        refresh_interval = st.slider("自动获取周期（秒，0 表示关闭自动）", 0, 60, 8, 1)
        gen_per_tick = st.slider("每周期获取批次数", 1, 5, 1)
        max_keep = st.slider("内存中保留的最大批次数", 20, 1000, 200, 10)
        st.session_state["max_batches_keep"] = max_keep

    # Auto generation
    if st.session_state["auto"] and refresh_interval > 0:
        for _ in range(gen_per_tick):
            st.session_state["batches"].append(generate_batch())
        time.sleep(refresh_interval)
        st.rerun()

    # -------------------- Dataframe / Summaries --------------------
    batches_list = list(st.session_state["batches"])
    if len(batches_list) == 0:
        st.warning("当前无批次数据。")
        st.stop()

    rows = []
    for b in batches_list:
        rows.append({
            "timestamp": b["timestamp"],
            "region": b["region"],
            "temp": b["temp"],
            "precip": b["precip"],
            "veg_index": b["veg_index"],
            "has_image": bool(b["modalities"].get("image", False)),
            "has_text": bool(b["modalities"].get("text", False)),
            "storage_mb": b["storage_mb"],
            "note": b["note"],
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("时间序列：气温 / 降水量 / NDVI")
        st.line_chart(df.set_index("timestamp")[["temp", "precip", "veg_index"]])

        st.subheader("模态计数随时间变化（image / text / sensor）")
        modality_counts = df.copy()
        modality_counts["sensor"] = 1
        modality_cum = modality_counts.set_index("timestamp")[["has_image", "has_text", "sensor"]].rename(
            columns={"has_image": "image", "has_text": "text"}
        ).cumsum()
        st.line_chart(modality_cum)

    with col_b:
        st.subheader("按高原地区统计存储量（MB）")
        st.bar_chart(df.groupby("region")["storage_mb"].sum().sort_values(ascending=False))

        st.subheader("最近 5 批详细信息")
        st.dataframe(df.tail(5))

    st.subheader("所有批次数据")
    st.dataframe(df.set_index("timestamp"))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("下载 CSV（全部数据）", data=csv, file_name="sim_plateau_batches.csv", mime="text/csv")