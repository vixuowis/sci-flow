import streamlit as st

from pages.c_safe_release_pages.scene1_earth import run_scene1
from pages.c_safe_release_pages.scene2_sea import run_scene2


st.title("可控发布")

st.markdown("# DataFrame Demo")
st.sidebar.header("DataFrame Demo")
st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
)

match st.session_state.data_scene:
    case "地质灾害":
        run_scene1()
    case "海洋牧场":
        run_scene2()
    case _: # 默认情况，类似于 switch 的 default
        run_scene1()
