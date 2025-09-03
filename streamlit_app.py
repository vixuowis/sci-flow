import streamlit as st

pg = st.navigation([
    st.Page("pages/0_home.py", title="首页", icon="🏠"),
    st.Page("pages/1_union_wrap.py", title="联合封装", icon="🔗"),
    st.Page("pages/2_realtime_wrap.py", title="实时封装", icon="📈"),
    st.Page("pages/3_safe_release.py", title="可控发布", icon="✅"),
    st.Page("pages/4_test.py", title="test", icon="✅"),
])

pg.run()