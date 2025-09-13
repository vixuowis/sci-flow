from session_fix import page_setup

from integration_helper import check_authentication, add_auth_sidebar
user_info = check_authentication()
add_auth_sidebar()

from pages.a_union_wrap_pages import run_scene1, run_scene2, run_scene3, run_scene4, run_scene5
from utils import *
if 'data_scene' not in st.session_state:
    st.session_state.data_scene = None
st.title("联合封装")

st.markdown("# Mapping Demo")
st.sidebar.header("Mapping Demo")
st.write(
    """This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
to display geospatial data."""
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
