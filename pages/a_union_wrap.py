from pages.a_union_wrap_pages.scene1_earth import run_scene1
from pages.a_union_wrap_pages.scene2_sea import run_scene2
from pages.a_union_wrap_pages.scene_4_agriculture import run_scene4
from utils import *

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
    case "农业智慧": # 默认情况，类似于 switch 的 default
        run_scene4()
