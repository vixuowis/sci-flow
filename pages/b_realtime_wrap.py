from session_fix import page_setup

import streamlit as st

from integration_helper import check_authentication, add_auth_sidebar
user_info = check_authentication()
add_auth_sidebar()

from pages.b_realtime_wrap_pages import run_scene1, run_scene2, run_scene3, run_scene4, run_scene5

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
    case "高原脆弱":
        run_scene3()
    case "农业智慧":
        run_scene4()
    case "森林生态":
        run_scene5()
    case _: # 默认情况，类似于 switch 的 default
        run_scene1()
