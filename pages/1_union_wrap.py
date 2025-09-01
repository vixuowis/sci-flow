import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError

st.title("联合封装")

st.markdown("# Mapping Demo")
st.sidebar.header("Mapping Demo")
st.write(
    """This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
to display geospatial data."""
)

import requests, io

url = "https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/%E7%9F%B3%E5%86%B0%E5%B7%9D%E6%95%B0%E6%8D%AE-%E9%81%A5%E6%84%9F%2B%E6%97%A0%E4%BA%BA%E6%9C%BA/%E9%B2%81%E6%9C%97%E7%9F%B3%E5%86%B0%E5%B7%9D_1107/001/MSS/SJY01_MSS_20241107_121209_002811_062_001_L1A.jpg?Expires=1756728492&AccessKeyId=HPUAJG9A804YAQ6QIVSS&Signature=9or72F8v5SM0/a3daRS7wMwzCxY%3D"
resp = requests.get(url, timeout=10)
# resp.raise_for_status()               # 确保请求成功
st.image(io.BytesIO(resp.content), caption="the image")

import pandas as pd
import pydeck as pdk
import streamlit as st
from numpy.random import default_rng as rng

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


# @st.cache_data
# def from_data_file(filename):
#     url = (
#         "http://raw.githubusercontent.com/streamlit/"
#         "example-data/master/hello/v1/%s" % filename
#     )
#     return pd.read_json(url)


# try:
#     ALL_LAYERS = {
#         "Bike Rentals": pdk.Layer(
#             "HexagonLayer",
#             data=from_data_file("bike_rental_stats.json"),
#             get_position=["lon", "lat"],
#             radius=200,
#             elevation_scale=4,
#             elevation_range=[0, 1000],
#             extruded=True,
#         ),
#         "Bart Stop Exits": pdk.Layer(
#             "ScatterplotLayer",
#             data=from_data_file("bart_stop_stats.json"),
#             get_position=["lon", "lat"],
#             get_color=[200, 30, 0, 160],
#             get_radius="[exits]",
#             radius_scale=0.05,
#         ),
#         "Bart Stop Names": pdk.Layer(
#             "TextLayer",
#             data=from_data_file("bart_stop_stats.json"),
#             get_position=["lon", "lat"],
#             get_text="name",
#             get_color=[0, 0, 0, 200],
#             get_size=15,
#             get_alignment_baseline="'bottom'",
#         ),
#         "Outbound Flow": pdk.Layer(
#             "ArcLayer",
#             data=from_data_file("bart_path_stats.json"),
#             get_source_position=["lon", "lat"],
#             get_target_position=["lon2", "lat2"],
#             get_source_color=[200, 30, 0, 160],
#             get_target_color=[200, 30, 0, 160],
#             auto_highlight=True,
#             width_scale=0.0001,
#             get_width="outbound",
#             width_min_pixels=3,
#             width_max_pixels=30,
#         ),
#     }
#     st.sidebar.markdown("### Map Layers")
#     selected_layers = [
#         layer
#         for layer_name, layer in ALL_LAYERS.items()
#         if st.sidebar.checkbox(layer_name, True)
#     ]
#     if selected_layers:
#         st.pydeck_chart(
#             pdk.Deck(
#                 map_style="mapbox://styles/mapbox/light-v9",
#                 initial_view_state={
#                     "latitude": 37.76,
#                     "longitude": -122.4,
#                     "zoom": 11,
#                     "pitch": 50,
#                 },
#                 layers=selected_layers,
#             )
#         )
#     else:
#         st.error("Please choose at least one layer above.")
# except URLError as e:
#     st.error(
#         """
#         **This demo requires internet access.**
#         Connection error: %s
#     """
#         % e.reason
#     )
