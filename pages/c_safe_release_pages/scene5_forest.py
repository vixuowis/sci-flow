"""
Streamlit app: Simulated forest ecology data ingestion & dashboard
- Fully simulated data (no external sources required)
- Modalities: sensor (numeric), image (small generated image), text (notes)
- Every "tick" the app can automatically generate new batches (or you can manually add)
- Shows:
    * Batch table (rows = batch timestamps)
    * Summary columns: storage(MB), modalities present, region
    * Time-series line chart for air temperature / soil moisture / vegetation index (different colored lines)
    * Time-series of modality counts (image/text/sensor)
    * Small image gallery for batches that include images
    * Download CSV
"""

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

REGIONS = [
    "Amazon_Rainforest",
    "Congo_Basin",
    "Himalayan_Forests",
    "Alpine_Forests",
    "Boreal_Siberia"
]

# realistic ranges for forest ecology metrics
TEMP_RANGE = (5.0, 35.0)       # °C
SOIL_MOISTURE_RANGE = (5.0, 60.0)  # % volumetric water content
VEG_INDEX_RANGE = (0.1, 0.9)   # NDVI (Normalized Difference Vegetation Index)

random.seed(42)
np.random.seed(42)

def run_scene5():
  def generate_text_note(region, temp, soil_moisture, veg_index):
      notes = [
          "lush canopy, high biodiversity",
          "dry understory, reduced growth",
          "signs of recent rainfall",
          "drought stress observed",
          "leaf fall detected",
          "wildlife activity noted",
      ]
      if veg_index > 0.7:
          return f"{random.choice(notes)}; dense vegetation NDVI:{veg_index:.2f}"
      if soil_moisture < 15:
          return f"{random.choice(notes)}; soil dry:{soil_moisture:.1f}%"
      return f"{random.choice(notes)}; temp:{temp:.1f}°C"

  def make_small_image(timestamp: datetime, region: str, temp: float, soil_moisture: float, veg_index: float):
      """Create a small PNG in-memory showing simple visualization + metadata text."""
      W, H = 480, 320
      img = Image.new("RGB", (W, H), color=(220, 255, 220))
      d = ImageDraw.Draw(img)

      # draw green "forest canopy" texture
      xs = np.linspace(0, 4 * np.pi, W)
      amplitude = int(20 + veg_index * 80)
      ymid = 160
      for x in range(W - 1):
          y = int(ymid + amplitude * np.sin(xs[x] + soil_moisture / 10.0))
          d.line([(x, y), (x + 1, y)], fill=(34, 139, 34))

      # overlay text
      try:
          font = ImageFont.truetype("DejaVuSans.ttf", 14)
      except Exception:
          font = ImageFont.load_default()

      text_lines = [
          timestamp.strftime("%Y-%m-%d %H:%M:%S"),
          f"{region}",
          f"T:{temp:.2f}°C  SM:{soil_moisture:.1f}%  NDVI:{veg_index:.2f}",
      ]
      tx, ty = 8, 8
      for line in text_lines:
          d.text((tx, ty), line, fill=(10, 30, 10), font=font)
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
          size += 2.0 + random.random() * 1.5
      size += 0.01
      return round(size, 3)

  def generate_batch(ts: datetime = None):
      if ts is None:
          ts = _now()
      region = random.choice(REGIONS)
      temp = float(np.random.uniform(*TEMP_RANGE))
      soil_moisture = float(np.random.uniform(*SOIL_MOISTURE_RANGE))
      veg_index = float(np.random.uniform(*VEG_INDEX_RANGE))

      modalities = {
          "sensor": True,
          "image": random.random() < 0.6,
          "text": random.random() < 0.9,
      }

      note = generate_text_note(region, temp, soil_moisture, veg_index) if modalities["text"] else ""
      image_bytes = make_small_image(ts, region, temp, soil_moisture, veg_index) if modalities["image"] else None
      storage_mb = compute_storage_mb(modalities)

      return {
          "timestamp": ts,
          "region": region,
          "temp": temp,
          "soil_moisture": soil_moisture,
          "veg_index": veg_index,
          "modalities": modalities,
          "note": note,
          "image_bytes": image_bytes,
          "storage_mb": storage_mb,
      }

  # -------------------- Session init --------------------
  if "batches" not in st.session_state:
      st.session_state["max_batches_keep"] = 200
      st.session_state["batches"] = deque(maxlen=st.session_state["max_batches_keep"])
      now = _now()
      for i in range(30, 0, -1):
          ts = now - timedelta(minutes=10 * i)
          st.session_state["batches"].append(generate_batch(ts))

  if "auto" not in st.session_state:
      st.session_state["auto"] = False

  # -------------------- UI --------------------
  st.set_page_config(page_title="Simulated Forest Ecology Dashboard", layout="wide")
  st.title("森林生态数据可控发布示例")

  with st.sidebar:
      st.header("控制面板")
      refresh_interval = st.slider("自动获取周期（秒，0 表示关闭自动）", 0, 60, 8, 1)
      gen_per_tick = st.slider("每周期获取批次数", 1, 5, 1)
      max_keep = st.slider("内存中保留的最大批次数（max）", 20, 1000, 200, 10)
      st.session_state["max_batches_keep"] = max_keep

  if st.session_state["auto"] and refresh_interval > 0:
      for _ in range(gen_per_tick):
          st.session_state["batches"].append(generate_batch())
      time.sleep(refresh_interval)
      st.rerun()

  # Convert to DataFrame
  df = pd.DataFrame(list(st.session_state["batches"]))
  df["timestamp"] = pd.to_datetime(df["timestamp"])
  df = df.sort_values("timestamp")

  summary_df = df[["timestamp", "region", "storage_mb", "temp", "soil_moisture", "veg_index", "note"]].set_index("timestamp")

  # Charts
  col_a, col_b = st.columns([2, 1])
  with col_a:
      st.subheader("时间序列：气温 / 土壤湿度 / 植被指数")
      ts_chart_df = df.set_index("timestamp")[["temp", "soil_moisture", "veg_index"]]
      st.line_chart(ts_chart_df)

      st.subheader("模态计数随时间变化")
      modality_counts = df.copy()
      modality_counts["sensor"] = 1
      modality_counts_ts = modality_counts.set_index("timestamp")[["modalities"]].copy()
      modality_counts_ts = pd.DataFrame({
          "image": [m["image"] for m in df["modalities"]],
          "text": [m["text"] for m in df["modalities"]],
          "sensor": [1 for _ in df["modalities"]],
      }, index=df["timestamp"])
      st.line_chart(modality_counts_ts.cumsum())

  with col_b:
      st.subheader("按区域统计存储量（MB）")
      st.bar_chart(df.groupby("region")["storage_mb"].sum())

      st.subheader("最近 5 批详细信息")
      st.dataframe(summary_df.tail(5))

  st.subheader("所有批次")
  st.dataframe(summary_df)

  csv = df.to_csv(index=False).encode("utf-8")
  st.download_button("下载 CSV", data=csv, file_name="sim_forest_batches.csv", mime="text/csv")

  st.subheader("含影像的批次示例（最近 8 条）")
  img_candidates = [b for b in reversed(list(st.session_state["batches"])) if b.get("image_bytes")]
  if img_candidates:
      gallery = img_candidates[:8]
      cols = st.columns(min(4, len(gallery)))
      for i, g in enumerate(gallery):
          with cols[i % len(cols)]:
              st.image(g["image_bytes"], caption=f"{g['timestamp']} | {g['region']}")
              if g.get("note"):
                  st.caption(g["note"])

  st.markdown("---")
  col1, col2, col3 = st.columns(3)
  with col1:
      st.metric("总批次数", len(df))
      st.metric("含影像批次数", int(sum(m["image"] for m in df["modalities"])))
  with col2:
      st.metric("总存储量（MB）", round(float(df['storage_mb'].sum()), 2))
      st.metric("平均NDVI", round(float(df['veg_index'].mean()), 2))
  with col3:
      st.metric("区域数", df['region'].nunique())
      st.metric("最近批次时间", df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'))