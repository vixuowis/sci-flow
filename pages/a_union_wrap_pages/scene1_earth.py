
# ==========================
# get obs file list tree test
import io
import requests

from obs import ObsClient

from utils import build_tree, list_all_objects

from streamlit_tree_select import tree_select

import streamlit as st

import pandas as pd
import pydeck as pdk
from numpy.random import default_rng as rng


def run_scene1():
  st.title("🐙 OBS 文件树展示")
  st.subheader("选择 OBS 文件或文件夹")

  # 配置 OBS
  ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"
  BUCKET = "gaoyuan-49d0"
  PREFIX = "石冰川数据-遥感+无人机"

  client = ObsClient(server=ENDPOINT)
  all_objs = list_all_objects(PREFIX, client, BUCKET, max_depth=3)  # 可调 max_depth
  client.close()

  # 构建树节点
  nodes = build_tree(all_objs, prefix=PREFIX)

  # 展示
  return_select = tree_select(nodes)
  st.write("你选择的节点：", return_select)

  # ==========================

  url = "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/%E7%9F%B3%E5%86%B0%E5%B7%9D%E6%95%B0%E6%8D%AE-%E9%81%A5%E6%84%9F%2B%E6%97%A0%E4%BA%BA%E6%9C%BA/%E9%B2%81%E6%9C%97%E7%9F%B3%E5%86%B0%E5%B7%9D_1107/001/MSS/SJY01_MSS_20241107_121209_002811_062_001_L1A.jpg"# 确保请求成功
  resp = requests.get(url, timeout=10)
  # resp.raise_for_status()               # 确保请求成功
  st.image(io.BytesIO(resp.content), caption="the image")


  df = pd.DataFrame(
      rng(0).standard_normal((1000, 2)) / [50, 50] + [37.76, -122.4],
      columns=["lat", "lon"],
  )

  st.pydeck_chart(
      pdk.Deck(
          map_style=None,  # Use Streamlit theme to pick map style
          initial_view_state=pdk.ViewState(
              latitude=37.76,
              longitude=-122.4,
              zoom=11,
              pitch=50,
          ),
          layers=[
              pdk.Layer(
                  "HexagonLayer",
                  data=df,
                  get_position="[lon, lat]",
                  radius=200,
                  elevation_scale=4,
                  elevation_range=[0, 1000],
                  pickable=True,
                  extruded=True,
              ),
              pdk.Layer(
                  "ScatterplotLayer",
                  data=df,
                  get_position="[lon, lat]",
                  get_color="[200, 30, 0, 160]",
                  get_radius=200,
              ),
          ],
      )
  )
