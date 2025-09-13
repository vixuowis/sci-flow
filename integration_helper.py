"""
现有项目集成助手
在你的现有Streamlit页面中添加认证保护
"""

import streamlit as st
import yaml
import hashlib
import hmac
import time
from yaml.loader import SafeLoader

def check_authentication():
    """
    检查用户是否已认证
    在每个需要保护的页面开头调用这个函数
    
    Returns:
        dict: 用户信息，如果未认证则返回None
    """
    
    # 检查是否已登录
    if not st.session_state.get('authenticated', False):
        show_login_required_page()
        st.stop()  # 停止页面执行
    
    # 检查会话是否过期
    if 'login_time' in st.session_state:
        current_time = time.time()
        login_time = st.session_state['login_time']
        
        # 1小时超时
        if current_time - login_time > 3600:
            logout_user()
            st.warning("⏰ 会话已超时，请重新登录")
            st.rerun()
    
    return st.session_state.get('user_info', {})

def show_login_required_page():
    """显示需要登录的页面"""
    st.title("🔐 需要登录")
    st.warning("此页面需要登录后才能访问")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 前往登录页面", type="primary", use_container_width=True):
            # 重定向到主登录页面
            st.switch_page("main_production.py")  # 或者你的主页面文件名
    
    with col2:
        st.markdown("### 📞 联系管理员")
        st.write("如需账号，请联系系统管理员")

def logout_user():
    """登出用户"""
    keys_to_remove = [
        'authenticated', 'username', 'user_info', 
        'login_time', 'user_role'
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

def add_auth_sidebar():
    """
    在侧边栏添加用户信息和登出按钮
    在每个已认证的页面调用这个函数
    """
    user_info = st.session_state.get('user_info', {})
    
    with st.sidebar:
        st.markdown("### 👤 用户信息")
        st.markdown(f"**姓名：** {user_info.get('name', 'N/A')}")
        st.markdown(f"**用户名：** {st.session_state.get('username', 'N/A')}")
        st.markdown(f"**角色：** {st.session_state.get('user_role', 'user')}")
        
        if 'login_time' in st.session_state:
            login_time = time.strftime('%H:%M:%S', 
                                    time.localtime(st.session_state['login_time']))
            st.caption(f"登录时间: {login_time}")
        
        st.markdown("---")
        
        if st.button("🚪 登出", type="secondary", use_container_width=True):
            logout_user()
            st.success("👋 已安全登出！")
            st.rerun()

def require_role(required_role):
    """
    角色权限检查装饰器
    
    Args:
        required_role (str): 需要的角色 ('admin', 'researcher', 'user')
    """
    current_role = st.session_state.get('user_role', 'user')
    
    # 角色层级：admin > researcher > user
    role_hierarchy = {'admin': 3, 'researcher': 2, 'user': 1}
    
    current_level = role_hierarchy.get(current_role, 0)
    required_level = role_hierarchy.get(required_role, 1)
    
    if current_level < required_level:
        st.error(f"❌ 权限不足！此功能需要 {required_role} 权限")
        st.stop()

# 使用示例代码
def example_protected_page():
    """
    示例：一个受保护的页面
    """
    
    # 1. 检查认证（必须）
    user_info = check_authentication()
    
    # 2. 检查角色权限（可选）
    # require_role('researcher')  # 只允许研究员和管理员访问
    
    # 3. 添加侧边栏用户信息（可选）
    add_auth_sidebar()
    
    # 4. 你的页面内容
    st.title("📊 受保护的数据分析页面")
    st.write(f"欢迎，{user_info['name']}！")
    
    # ... 你的具体功能代码 ...

if __name__ == "__main__":
    # 演示如何使用
    example_protected_page()