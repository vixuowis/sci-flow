import streamlit as st

from pages.b_realtime_wrap_pages.scene1_earth import run_scene1
from pages.b_realtime_wrap_pages.scene2_sea import run_scene2

st.title("实时封装")

st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

match st.session_state.data_scene:
    case "地质灾害":
        run_scene1()
    case "海洋牧场":
        run_scene2()
    case _: # 默认情况，类似于 switch 的 default
        run_scene1()
