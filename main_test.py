"""
Sci-Flow 测试版本 - 使用明文密码比较
仅用于测试登录功能
"""

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="Sci-Flow",
    page_icon="🧪",
    layout="wide"
)

# 简单的用户数据（明文密码，仅用于测试）
USERS = {
    'admin': {
        'password': 'admin123',
        'name': '管理员',
        'email': 'admin@sciflow.com'
    },
    'researcher1': {
        'password': 'research123',  
        'name': '研究员1',
        'email': 'researcher1@sciflow.com'
    },
    'user1': {
        'password': 'user123',
        'name': '用户1',
        'email': 'user1@sciflow.com'
    }
}

def authenticate_user(username, password):
    """简单的用户验证函数"""
    if username in USERS and USERS[username]['password'] == password:
        return True, USERS[username]
    return False, None

def show_login_page():
    """显示登录页面"""
    st.title("🔐 Sci-Flow 登录系统")
    st.markdown("---")
    
    # 创建登录表单
    with st.form("login_form"):
        st.markdown("### 请输入登录信息")
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        login_button = st.form_submit_button("登录")
        
        if login_button:
            if username and password:
                is_valid, user_info = authenticate_user(username, password)
                
                if is_valid:
                    # 登录成功，保存到session state
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['user_info'] = user_info
                    st.success("✅ 登录成功！")
                    st.rerun()
                else:
                    st.error("❌ 用户名或密码错误")
            else:
                st.warning("⚠️ 请输入用户名和密码")
    
    # 显示测试账号
    with st.expander("🔍 点击查看测试账号"):
        st.markdown("""
        | 角色 | 用户名 | 密码 |
        |------|--------|------|
        | 管理员 | `admin` | `admin123` |
        | 研究员 | `researcher1` | `research123` |
        | 用户 | `user1` | `user123` |
        """)

def show_main_app():
    """显示主应用"""
    user_info = st.session_state.get('user_info', {})
    
    # 顶部信息栏
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown(f"👋 **欢迎回来，{user_info.get('name', '用户')}！**")
    
    with col2:
        if st.button("登出", type="secondary"):
            # 清除登录状态
            for key in ['authenticated', 'username', 'user_info']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    st.markdown("---")
    
    # 主应用内容
    st.title("🧪 Sci-Flow 科研工作流平台")
    
    # 显示用户信息
    with st.sidebar:
        st.markdown("### 👤 用户信息")
        st.markdown(f"**姓名：** {user_info.get('name', 'N/A')}")
        st.markdown(f"**用户名：** {st.session_state.get('username', 'N/A')}")
        st.markdown(f"**邮箱：** {user_info.get('email', 'N/A')}")
        st.markdown("---")
        st.success("🎉 登录系统工作正常！")
    
    # 主要功能区域
    tab1, tab2, tab3 = st.tabs(["📊 数据分析", "📝 实验记录", "📈 结果可视化"])
    
    with tab1:
        st.header("📊 数据分析模块")
        st.success("🎉 登录系统集成成功！")
        st.write("这里将是你的数据分析功能...")
        
        # 示例内容
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("数据集", "12", "2")
        with col2:
            st.metric("分析任务", "8", "-1")
        with col3:
            st.metric("已完成", "6", "1")
    
    with tab2:
        st.header("📝 实验记录模块")
        st.write("这里将是你的实验记录功能...")
        st.info("💡 你可以在这里添加实验记录相关的功能")
    
    with tab3:
        st.header("📈 结果可视化模块")
        st.write("这里将是你的结果可视化功能...")
        
        # 示例图表
        import numpy as np
        import pandas as pd
        
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['实验A', '实验B', '实验C']
        )
        st.line_chart(chart_data)

def main():
    """主函数"""
    # 检查登录状态
    if st.session_state.get('authenticated', False):
        show_main_app()
    else:
        show_login_page()

if __name__ == "__main__":
    main()