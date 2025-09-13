from session_fix import page_setup

from integration_helper import check_authentication, add_auth_sidebar
user_info = check_authentication()
add_auth_sidebar()

import streamlit as st

from pages.c_safe_release_pages import run_scene1, run_scene2, run_scene3, run_scene4, run_scene5
if 'data_scene' not in st.session_state:
    st.session_state.data_scene = None


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
    case "高原脆弱":
        run_scene3()
    case "农业智慧":
        run_scene4()
    case "森林生态":
        run_scene5()
    case _: # 默认情况，类似于 switch 的 default
        run_scene1()
