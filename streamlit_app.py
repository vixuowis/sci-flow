import streamlit as st

pg = st.navigation([
    st.Page("pages/0_home.py", title="首页", icon="🏠"),
    st.Page("pages/a_union_wrap.py", title="联合封装", icon="🔗"),
    st.Page("pages/b_realtime_wrap.py", title="实时封装", icon="📈"),
    st.Page("pages/c_safe_release.py", title="可控发布", icon="✅"),
    st.Page("pages/4_test.py", title="test", icon="✅"),
])

st.sidebar.selectbox("## 场景选择", ["地质灾害", "海洋牧场", "高原脆弱", "农业智慧", "森林生态"], key="data_scene")

pg.run()