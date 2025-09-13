"""
Sci-Flow 简化版认证模块
这个版本更加简单和兼容
"""

import yaml
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.load(file, Loader=SafeLoader)

def create_authenticator(config):
    """创建认证器"""
    return stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

def show_login_page():
    """显示登录页面"""
    st.title("🔐 Sci-Flow 登录系统")
    st.markdown("---")
    
    # 加载配置和创建认证器
    config = load_config()
    authenticator = create_authenticator(config)
    
    # 显示登录表单
    authenticator.login()
    
    # 检查登录状态
    if st.session_state.get('authentication_status') is False:
        st.error('用户名或密码错误')
    elif st.session_state.get('authentication_status') is None:
        st.warning('请输入您的用户名和密码')
        
        # 显示测试账号信息
        with st.expander("🔍 测试账号信息"):
            st.markdown("""
            **管理员账号：**
            - 用户名：`admin`
            - 密码：`admin123`
            
            **研究员账号：**
            - 用户名：`researcher1`
            - 密码：`research123`
            
            **普通用户：**
            - 用户名：`user1`
            - 密码：`user123`
            """)

def show_main_app():
    """显示主应用内容"""
    # 加载配置和创建认证器
    config = load_config()
    authenticator = create_authenticator(config)
    
    # 显示用户信息和登出按钮
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.get('name'):
            st.markdown(f"👋 欢迎回来，**{st.session_state['name']}**！")
    
    with col2:
        authenticator.logout(location='main')
    
    st.markdown("---")
    
    # 主应用内容
    st.title("🧪 Sci-Flow 科研工作流平台")
    
    # 示例功能模块
    tab1, tab2, tab3 = st.tabs(["📊 数据分析", "📝 实验记录", "📈 结果可视化"])
    
    with tab1:
        st.header("数据分析模块")
        st.write("这里是数据分析功能...")
        st.info("✅ 登录系统正常工作！")
        
    with tab2:
        st.header("实验记录模块")
        st.write("这里是实验记录功能...")
        
    with tab3:
        st.header("结果可视化模块")
        st.write("这里是结果可视化功能...")

def main():
    """主函数"""
    # 页面配置
    st.set_page_config(
        page_title="Sci-Flow",
        page_icon="🧪",
        layout="wide"
    )
    
    # 检查登录状态
    if st.session_state.get('authentication_status'):
        show_main_app()
    else:
        show_login_page()
        
        # 如果刚刚登录成功，刷新页面
        if st.session_state.get('authentication_status'):
            st.rerun()

if __name__ == "__main__":
    main()