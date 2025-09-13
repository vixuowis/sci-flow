"""
通用 Session State 修复工具
在任何出错的页面开头导入并使用
"""

import streamlit as st

def fix_all_session_errors():
    """
    修复所有常见的 session_state 错误
    在每个页面开头调用一次
    """
    
    # 根据你的错误信息修复 data_scene
    if 'data_scene' not in st.session_state:
        st.session_state.data_scene = None
    
    # 修复其他常见错误
    common_vars = {
        'current_scene': 'default',
        'page_data': {},
        'user_inputs': {},
        'processing_state': 'idle',
        'results': None,
        'uploaded_files': [],
        'analysis_data': {},
        'plot_config': {},
        'form_state': {},
        'step_counter': 0,
        'cache_data': {},
        'temp_data': {},
        'ui_state': 'normal'
    }
    
    for key, default_value in common_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def add_page_header(page_name="当前页面"):
    """
    添加页面头部，显示认证状态
    """
    st.markdown(f"### {page_name}")
    
    # 如果已登录，显示用户信息
    if st.session_state.get('authenticated', False):
        user_info = st.session_state.get('user_info', {})
        user_name = user_info.get('name', '用户')
        st.success(f"✅ 已登录：{user_name}")
    else:
        # 未登录提示
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 建议登录以获得完整功能和数据保存")
        with col2:
            if st.button("去登录", type="secondary"):
                st.code("streamlit run main_simple_fixed.py")

# 一键修复函数
def page_setup(page_name="页面"):
    """
    一键设置页面：修复错误 + 添加头部
    """
    fix_all_session_errors()
    add_page_header(page_name)

# 使用示例
if __name__ == "__main__":
    page_setup("修复工具测试页面")
    
    st.write("### 🔧 Session State 状态")
    st.write("以下变量已安全初始化：")
    
    for key, value in st.session_state.items():
        st.write(f"- `{key}`: {type(value).__name__}")