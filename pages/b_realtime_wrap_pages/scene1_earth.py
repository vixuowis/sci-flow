import streamlit as st
import numpy as np
import time
import requests
from obspy import read
from io import BytesIO

def run_scene1():
    # Download mseed file from Huawei Cloud URL
    url = "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/%E5%9C%B0%E8%B4%A8%E7%81%BE%E5%AE%B3-%E6%96%87%E6%9C%AC%2B%E9%81%A5%E6%84%9F%2B%E6%97%B6%E5%BA%8F/IRIS/dataset_earthquake/IU_AFI_-13.91_-171.78_waveforms.mseed"
    resp = requests.get(url)
    resp.raise_for_status()
    
    # Read mseed using ObsPy
    st.write("读取 mseed 文件中...")
    st_stream = read(BytesIO(resp.content))
    tr = st_stream[0]  # Take the first trace
    data = tr.data.astype(np.float32)
    times = np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts)

    st.write(f"台站: {tr.stats.network}.{tr.stats.station}  "
             f"采样率: {tr.stats.sampling_rate} Hz  "
             f"总点数: {tr.stats.npts}")

    # ==== Downsample to reduce points ====
    downsample_factor = 50  # Take one every N points
    data = data[::downsample_factor]
    times = times[::downsample_factor]

    # st.write(f"下采样后点数: {len(data)}")

    # Dynamic plotting
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    chart = st.line_chart([])

    chunk_size = 200  # Number of points to add per iteration
    total_chunks = len(data) // chunk_size

    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        new_values = data[start:end]

        chart.add_rows({"Amplitude": new_values})
        status_text.text(f"{(i+1) * 100 // total_chunks}% Complete")
        progress_bar.progress((i+1) * 100 // total_chunks)
        time.sleep(0.05)

    progress_bar.empty()
    st.button("Re-run")


    