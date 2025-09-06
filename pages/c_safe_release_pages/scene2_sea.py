"""
Streamlit app: Simulated ocean data ingestion & dashboard
- Fully simulated data (no external sources required)
- Modalities: sensor (numeric), image (small generated image), text (notes)
- Every "tick" the app can automatically generate new batches (or you can manually add)
- Shows:
    * Batch table (rows = batch timestamps)
    * Summary columns: storage(MB), modalities present, region
    * Time-series line chart for sea temperature / salinity / biomass (different colored lines)
    * Time-series of modality counts (image/text/sensor)
    * Small image gallery for batches that include images
    * Download CSV

Run with:
    streamlit run scene2_sea_sim.py

Dependencies:
    pip install streamlit pandas numpy matplotlib pillow

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

REGIONS = ["North_Sea", "South_Pacific", "East_China_Sea", "Gulf_of_Mexico", "Indian_Ocean"]

# realistic ranges for sea metrics
TEMP_RANGE = (2.0, 30.0)       # deg C
SALINITY_RANGE = (30.0, 38.0)  # PSU
BIOMASS_RANGE = (0.1, 50.0)    # mg/m^3 (very approximate)

random.seed(42)
np.random.seed(42)

def run_scene2():
  def generate_text_note(region, temp, salinity, biomass):
      notes = [
          "clear surface, low turbidity",
          "slightly turbid, possible runoff",
          "algae bloom observed",
          "high turbidity after storm",
          "sampling indicates healthy biomass",
          "sudden temp spike near surface",
      ]
      # weight selection by conditions
      if biomass > 30:
          return f"{random.choice(notes)}; high biomass:{biomass:.1f} mg/m^3"
      if temp > 26:
          return f"{random.choice(notes)}; warm surface:{temp:.1f}°C"
      return f"{random.choice(notes)}; salinity:{salinity:.1f} PSU"


  def make_small_image(timestamp: datetime, region: str, temp: float, salinity: float, biomass: float):
      """Create a small PNG in-memory showing simple visualization + metadata text.
      """
      # create a 320x200 image
      W, H = 480, 320
      img = Image.new("RGB", (W, H), color=(230, 245, 255))
      d = ImageDraw.Draw(img)

      # draw a simple gradient-like wave using sin
      xs = np.linspace(0, 4 * np.pi, W)
      amplitude = int(30 + (biomass / (BIOMASS_RANGE[1] + 1)) * 60)
      ymid = 140
      for x in range(W - 1):
          y = int(ymid + amplitude * np.sin(xs[x] + temp / 5.0))
          d.line([(x, y), (x + 1, y)], fill=(20, 80, 160))

      # overlay text (timestamp, region, metrics)
      try:
          font = ImageFont.truetype("DejaVuSans.ttf", 14)
      except Exception:
          font = ImageFont.load_default()

      text_lines = [
          timestamp.strftime("%Y-%m-%d %H:%M:%S"),
          f"{region}",
          f"T:{temp:.2f}°C  S:{salinity:.2f} PSU  B:{biomass:.2f} mg/m^3",
      ]
      tx = 8
      ty = 8
      for line in text_lines:
          d.text((tx, ty), line, fill=(10, 10, 10), font=font)
          ty += 18

      # encode to bytes
      buf = io.BytesIO()
      img.save(buf, format="PNG")
      buf.seek(0)
      return buf.read()


  def compute_storage_mb(modalities: dict, temp_count=1):
      """Very rough simulated storage size per batch in MB.
      - sensor data small
      - text tiny
      - image big (a few MB)
      """
      size = 0.0
      # base for sensors: one reading per metric
      size += 0.02 * temp_count
      if modalities.get("text", False):
          size += 0.002 * 30  # assume 30 chars
      if modalities.get("image", False):
          size += 2.5 + random.random() * 2.0  # 2.5-4.5 MB
      # small overhead
      size += 0.01
      return round(size, 3)


  def generate_batch(ts: datetime = None):
      if ts is None:
          ts = _now()
      region = random.choice(REGIONS)
      temp = float(np.random.uniform(*TEMP_RANGE))
      salinity = float(np.random.uniform(*SALINITY_RANGE))
      biomass = float(np.random.uniform(*BIOMASS_RANGE))

      # modality presence probabilities
      modalities = {
          "sensor": True,
          "image": random.random() < 0.6,  # 60% batches include an image
          "text": random.random() < 0.9,
      }

      note = generate_text_note(region, temp, salinity, biomass) if modalities["text"] else ""
      image_bytes = make_small_image(ts, region, temp, salinity, biomass) if modalities["image"] else None
      storage_mb = compute_storage_mb(modalities)

      batch = {
          "timestamp": ts,
          "region": region,
          "temp": temp,
          "salinity": salinity,
          "biomass": biomass,
          "modalities": modalities,
          "note": note,
          "image_bytes": image_bytes,
          "storage_mb": storage_mb,
      }
      return batch


  # -------------------- Session init --------------------

  if "batches" not in st.session_state:
      # create a deque to hold batches (maxlen configurable)
      st.session_state["max_batches_keep"] = 200
      st.session_state["batches"] = deque(maxlen=st.session_state["max_batches_keep"]) 
      # simulate historical batches (e.g., 30 past batches every 10 minutes)
      now = _now()
      N_init = 30
      for i in range(N_init, 0, -1):
          ts = now - timedelta(minutes=10 * i)
          st.session_state["batches"].append(generate_batch(ts))

  if "auto" not in st.session_state:
      st.session_state["auto"] = False

  # -------------------- UI --------------------

  st.set_page_config(page_title="Simulated Ocean Data Dashboard", layout="wide")
  st.title("海洋数据可控发布示例")

  # Controls
  with st.sidebar:
      st.header("控制面板")
      refresh_interval = st.slider("自动获取周期（秒，0 表示关闭自动）", min_value=0, max_value=60, value=8, step=1)
      gen_per_tick = st.slider("每周期获取批次数", min_value=1, max_value=5, value=1)
      max_keep = st.slider("内存中保留的最大批次数（max）", min_value=20, max_value=1000, value=200, step=10)
      st.session_state["max_batches_keep"] = max_keep

      
  # Auto generation: blocks for a short time then rerun (simple approach)
  if st.session_state["auto"] and refresh_interval > 0:
      # generate some batches, sleep for refresh_interval, then rerun so UI updates
      for _ in range(gen_per_tick):
          st.session_state["batches"].append(generate_batch())
      # brief pause to avoid tight loop
      time.sleep(refresh_interval)
      # rerun to refresh the UI after sleep
      st.rerun()

  # -------------------- Dataframe / Summaries --------------------

  # Convert deque to DataFrame
  batches_list = list(st.session_state["batches"])
  if len(batches_list) == 0:
      st.warning("当前无批次数据。请在左侧点击 '手动生成 1 批' 或开始自动生成。")
      st.stop()

  # flatten to DataFrame
  rows = []
  for b in batches_list:
      rows.append({
          "timestamp": b["timestamp"],
          "region": b["region"],
          "temp": b["temp"],
          "salinity": b["salinity"],
          "biomass": b["biomass"],
          "has_image": bool(b["modalities"].get("image", False)),
          "has_text": bool(b["modalities"].get("text", False)),
          "storage_mb": b["storage_mb"],
          "note": b["note"],
      })

  df = pd.DataFrame(rows)
  # ensure timestamp is datetime and sorted
  df["timestamp"] = pd.to_datetime(df["timestamp"])
  df = df.sort_values("timestamp")

  # summary table has timestamps as rows (vertical) and a few horizontal columns
  summary_df = df[["timestamp", "region", "storage_mb", "has_image", "has_text", "temp", "salinity", "biomass"]].copy()
  summary_df["types_present"] = summary_df.apply(lambda r: ", ".join([t for t, v in [("temp", True), ("salinity", True), ("biomass", True)] if v]), axis=1)
  summary_df = summary_df[["timestamp", "region", "storage_mb", "has_image", "has_text", "temp", "salinity", "biomass"]]
  summary_df = summary_df.set_index("timestamp")

  # Top area: charts and key summaries
  col_a, col_b = st.columns([2, 1])
  with col_a:
      st.subheader("时间序列：海水温度 / 盐度 / 生物量")
      ts_chart_df = df.set_index("timestamp")[ ["temp", "salinity", "biomass"] ]
      st.line_chart(ts_chart_df)

      st.subheader("模态计数随时间变化（image / text / sensor）")
      modality_counts = df.copy()
      modality_counts["sensor"] = 1  # every batch has sensor
      modality_counts_ts = modality_counts.set_index("timestamp")[ ["has_image", "has_text", "sensor"] ]
      modality_counts_ts = modality_counts_ts.rename(columns={"has_image": "image", "has_text": "text"})
      # accumulate counts over time (cumulative) for clearer trend
      modality_cum = modality_counts_ts.cumsum()
      st.line_chart(modality_cum)

  with col_b:
      st.subheader("按海区统计存储量（MB）")
      region_storage = df.groupby("region")["storage_mb"].sum().sort_values(ascending=False)
      st.bar_chart(region_storage)

      st.subheader("最近 5 批详细信息")
      st.dataframe(summary_df.tail(5))

  # Middle area: full table with download
  st.subheader("所有批次（时间纵列） — 存储量 + 模态 + 海区")
  st.dataframe(summary_df)

  csv = df.to_csv(index=False).encode("utf-8")
  st.download_button("下载 CSV（全部数据）", data=csv, file_name="sim_ocean_batches.csv", mime="text/csv")

  # Image gallery and text notes for recent image-bearing batches
  st.subheader("含影像的批次示例（最近 8 条）")
  img_candidates = [b for b in reversed(batches_list) if b.get("image_bytes")]
  if len(img_candidates) == 0:
      st.info("当前没有含影像的批次。可通过左侧面板提高影像出现概率或手动生成更多批次。")
  else:
      gallery = img_candidates[:8]
      cols = st.columns(min(4, len(gallery)))
      for i, g in enumerate(gallery):
          with cols[i % len(cols)]:
              st.image(g["image_bytes"], caption=f"{g['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | {g['region']}")
              if g.get("note"):
                  st.caption(g.get("note"))

  # Bottom: aggregate summaries and small table
  st.markdown("---")
  col1, col2, col3 = st.columns(3)
  with col1:
      st.metric("总批次数", len(df))
      st.metric("含影像批次数", int(df['has_image'].sum()))
  with col2:
      st.metric("总存储量（MB）", round(float(df['storage_mb'].sum()), 2))
      st.metric("平均生物量", round(float(df['biomass'].mean()), 2))
  with col3:
      st.metric("海区数", df['region'].nunique())
      st.metric("最近批次时间", df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'))


  # End
