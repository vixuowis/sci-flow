"""
Sci-Flow 简化修复版
使用最基础的哈希方法，确保兼容性
"""

import streamlit as st
import yaml
import hashlib
from yaml.loader import SafeLoader

# 页面配置
st.set_page_config(
    page_title="Sci-Flow",
    page_icon="🧪",
    layout="wide"
)

# 配置文件路径
CONFIG_FILE = "users_simple.yaml"

def simple_hash(password):
    """简单的密码哈希函数"""
    # 添加盐值后进行SHA256哈希
    salt = "sci_flow_salt_2024"
    combined = password + salt
    return hashlib.sha256(combined.encode()).hexdigest()

def create_default_config():
    """创建默认的用户配置文件"""
    default_config = {
        'users': {
            'admin': {
                'password_hash': simple_hash('admin123'),
                'name': '管理员',
                'email': 'admin@sciflow.com',
                'role': 'admin'
            },
            'researcher1': {
                'password_hash': simple_hash('research123'),
                'name': '研究员1', 
                'email': 'researcher1@sciflow.com',
                'role': 'researcher'
            },
            'user1': {
                'password_hash': simple_hash('user123'),
                'name': '用户1',
                'email': 'user1@sciflow.com', 
                'role': 'user'
            }
        },
        'settings': {
            'session_timeout': 3600,  # 1小时
            'app_name': 'Sci-Flow'
        }
    }
    
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        st.success(f"✅ 已创建默认配置文件：{CONFIG_FILE}")
        return default_config
        
    except Exception as e:
        st.error(f"❌ 创建配置文件时出错：{str(e)}")
        return None

def load_config():
    """加载用户配置"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=SafeLoader)
    except FileNotFoundError:
        st.info("⚠️ 配置文件不存在，创建默认配置...")
        return create_default_config()
    except Exception as e:
        st.error(f"❌ 加载配置文件时出错：{str(e)}")
        return None

def verify_password(password, password_hash):
    """验证密码"""
    return simple_hash(password) == password_hash

def authenticate_user(username, password, config):
    """用户认证函数"""
    if not config or 'users' not in config:
        return False, None
        
    users = config['users']
    
    if username in users:
        stored_hash = users[username]['password_hash']
        if verify_password(password, stored_hash):
            return True, users[username]
    
    return False, None

def check_session_timeout():
    """检查会话是否超时"""
    if 'login_time' in st.session_state:
        import time
        current_time = time.time()
        login_time = st.session_state['login_time']
        
        # 1小时超时
        if current_time - login_time > 3600:
            logout_user()
            return False
    return True

def logout_user():
    """登出用户"""
    keys_to_remove = [
        'authenticated', 'username', 'user_info', 
        'login_time', 'user_role'
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

def show_login_page(config):
    """显示登录页面"""
    st.title("🔐 Sci-Flow 登录系统")
    
    # 如果配置文件刚创建，显示提示
    if config is None:
        st.error("❌ 无法加载配置文件，请检查文件权限")
        return
    
    st.markdown("---")
    
    # 创建登录表单
    with st.form("login_form"):
        st.markdown("### 🚀 请输入登录信息")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            username = st.text_input("👤 用户名", placeholder="请输入用户名")
            password = st.text_input("🔑 密码", type="password", placeholder="请输入密码")
            
            login_button = st.form_submit_button("🚀 登录", type="primary", use_container_width=True)
        
        with col2:
            st.markdown("#### 🔍 测试账号")
            st.markdown("""
            | 角色 | 用户名 | 密码 |
            |------|--------|------|
            | 👑 管理员 | `admin` | `admin123` |
            | 🔬 研究员 | `researcher1` | `research123` |
            | 👤 用户 | `user1` | `user123` |
            """)
        
        if login_button:
            if username and password:
                is_valid, user_info = authenticate_user(username, password, config)
                
                if is_valid:
                    # 登录成功
                    import time
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['user_info'] = user_info
                    st.session_state['login_time'] = time.time()
                    st.session_state['user_role'] = user_info.get('role', 'user')
                    
                    st.success(f"✅ 欢迎回来，{user_info['name']}！")
                    st.balloons()  # 庆祝动画
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ 用户名或密码错误")
            else:
                st.warning("⚠️ 请输入用户名和密码")

def show_main_app(config):
    """显示主应用"""
    # 检查会话超时
    if not check_session_timeout():
        st.warning("⏰ 会话已超时，请重新登录")
        st.rerun()
        return
        
    user_info = st.session_state.get('user_info', {})
    user_role = st.session_state.get('user_role', 'user')
    
    # 顶部导航栏
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        st.markdown(f"### 👋 欢迎回来，{user_info.get('name', '用户')}")
        role_emoji = {'admin': '👑', 'researcher': '🔬', 'user': '👤'}
        st.caption(f"{role_emoji.get(user_role, '👤')} {user_role} | 用户名: {st.session_state.get('username')}")
    
    with col2:
        # 显示登录时间
        if 'login_time' in st.session_state:
            import time
            login_time = time.strftime('%H:%M:%S', 
                                    time.localtime(st.session_state['login_time']))
            st.caption(f"🕐 登录时间: {login_time}")
    
    with col3:
        if st.button("🚪 登出", type="secondary", use_container_width=True):
            logout_user()
            st.success("👋 已安全登出！")
            st.rerun()
    
    st.markdown("---")
    
    # 主应用内容
    show_dashboard(user_role, user_info)

def show_dashboard(user_role, user_info):
    """显示仪表盘"""
    st.title("🧪 Sci-Flow 科研工作流平台")
    
    # 用户侧边栏信息
    with st.sidebar:
        st.markdown("### 👤 用户信息")
        st.markdown(f"**姓名：** {user_info.get('name', 'N/A')}")
        st.markdown(f"**邮箱：** {user_info.get('email', 'N/A')}")
        st.markdown(f"**角色：** {user_role}")
        st.markdown("---")
        
        # 根据角色显示不同的侧边栏内容
        if user_role == 'admin':
            st.success("🔑 管理员权限")
            st.markdown("- 用户管理\n- 系统设置\n- 所有功能")
        elif user_role == 'researcher': 
            st.info("🔬 研究员权限")
            st.markdown("- 实验管理\n- 数据分析\n- 结果导出")
        else:
            st.warning("👤 基础权限")
            st.markdown("- 查看数据\n- 基础功能")
    
    # 主内容区域
    if user_role == 'admin':
        show_admin_dashboard()
    elif user_role == 'researcher': 
        show_researcher_dashboard()
    else:
        show_user_dashboard()

def show_admin_dashboard():
    """管理员仪表盘"""
    st.success("🔑 管理员控制台")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 系统总览", "👥 用户管理", "📝 实验管理", "⚙️ 设置"])
    
    with tab1:
        st.header("📊 系统总览")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总用户数", "3", "0")
        with col2:
            st.metric("活跃实验", "8", "+2")
        with col3:
            st.metric("数据集", "156", "+12")
        with col4:
            st.metric("系统状态", "正常", "")
        
        # 示例图表
        import numpy as np
        import pandas as pd
        
        chart_data = pd.DataFrame(
            np.random.randn(30, 3),
            columns=['用户活跃度', '实验数量', '数据处理量']
        )
        st.line_chart(chart_data)
    
    with tab2:
        st.header("👥 用户管理")
        st.success("🎉 认证系统集成成功！")
        st.write("在这里你可以添加用户管理功能...")
        
        if st.button("📝 添加新用户"):
            st.info("用户添加功能开发中...")
    
    with tab3:
        st.header("📝 实验管理")
        st.write("实验管理功能开发中...")
        
    with tab4:
        st.header("⚙️ 系统设置")
        st.write("系统设置功能开发中...")

def show_researcher_dashboard():
    """研究员仪表盘"""
    st.info("🔬 研究员工作台")
    
    tab1, tab2, tab3 = st.tabs(["📊 我的实验", "📈 数据分析", "📝 实验记录"])
    
    with tab1:
        st.header("📊 我的实验")
        st.success("🎉 认证系统工作正常！")
        
        # 示例实验列表
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("进行中", "3", "+1")
        with col2:
            st.metric("已完成", "12", "+2")
        with col3:
            st.metric("待审核", "1", "0")
        
    with tab2:
        st.header("📈 数据分析")
        st.write("数据分析功能开发中...")
        
    with tab3:
        st.header("📝 实验记录")
        st.write("实验记录功能开发中...")

def show_user_dashboard():
    """普通用户仪表盘"""
    st.warning("👤 用户工作台")
    
    tab1, tab2 = st.tabs(["📊 我的数据", "📈 查看结果"])
    
    with tab1:
        st.header("📊 我的数据")
        st.success("🎉 登录系统集成成功！")
        st.write("个人数据功能开发中...")
        
    with tab2:
        st.header("📈 查看结果")
        st.write("结果查看功能开发中...")

def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    if config is None:
        st.error("❌ 系统初始化失败")
        return
    
    # 检查登录状态
    if st.session_state.get('authenticated', False):
        show_main_app(config)
    else:
        show_login_page(config)

if __name__ == "__main__":
    main()