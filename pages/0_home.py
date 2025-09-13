# pages/0_home.py
"""
科学数据流管理平台首页
提供项目介绍和功能导航
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="科学数据流管理平台",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主标题和描述
st.title("🌐 科学数据流管理平台")
st.markdown("### 基于流批一体架构的海量动态科学数据实时封装与可控发布系统")

# 添加分隔线
st.divider()

# 项目简介
with st.container():
    st.markdown("## 📋 项目概述")
    st.markdown("""
    本平台是一个综合性科学数据管理系统，专门针对多种自然环境监测场景设计，提出了一种混合式高性能流批一体化编程模型，实现对多源异构科学数据与知识要素的统一封装与高效管理。
    系统具备联合封装、实时封装与可控发布三大核心功能，支持从外部数据处理、动态筛选、多模态可视化到权限管理、数据表格分析及导出的全流程操作。
                
    **核心特性：**
    - 🔄 **实时数据流处理** - 支持流式数据的实时监控和分析
    - 🗂️ **多模态数据融合** - 整合时序、图像、文本等多种数据类型
    - 📊 **智能可视化** - 提供丰富的交互式图表和地图展示
    - ☁️ **云原生架构** - 基于华为云 OBS 实现弹性存储和高效访问
    """)

# 功能模块介绍
st.divider()
st.markdown("## 🚀 功能模块")

# 使用三列布局展示三大功能模块
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("### 🔗 联合封装")
        st.markdown("""
        **多源数据的统一展示**
        
        不同来源和格式的数据标准化处理和展示：
        - 地震波形 3D 可视化
        - 海洋观测数据地图展示
        - 遥感影像轨迹绘制
        - 农业环境数据分析
        - 森林生态监测集成
        """)
        if st.button("进入联合封装 →", key="btn_union", use_container_width=True):
            st.switch_page("pages/a_union_wrap.py")

with col2:
    with st.container():
        st.markdown("### 📈 实时封装")
        st.markdown("""
        **动态数据流监控**
        
        实时监控和分析各类传感器数据流：
        - 地震波形实时展示
        - 海洋数据流量监控
        - 无人机直播流传输
        - 农业数据实时更新
        - 森林监测数据流
        """)
        if st.button("进入实时封装 →", key="btn_realtime", use_container_width=True):
            st.switch_page("pages/b_realtime_wrap.py")

with col3:
    with st.container():
        st.markdown("### ✅ 可控发布")
        st.markdown("""
        **数据的安全发布管理**
        
        提供数据的受控发布和展示功能：
        - OBS 文件批次统计
        - 海洋数据可控展示
        - 高原环境数据发布
        - 农业产值数据分析
        - 森林生态数据管理
        """)
        if st.button("进入可控发布 →", key="btn_release", use_container_width=True):
            st.switch_page("pages/c_safe_release.py")

# 应用场景展示
st.divider()
st.markdown("## 🌍 应用场景")

# 创建场景选项卡
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏔️ 地质灾害", "🌊 海洋牧场", "🏔️ 高原脆弱", "🌾 农业智慧", "🌲 森林生态"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **地质灾害监测与预警**
        
        - **数据源**：IRIS 地震台网 mseed 格式波形数据
        - **存储位置**：华为云 OBS `地质灾害-文本+遥感+时序/` 路径
        - **核心功能**：
            - 地震波形的实时采集和存储
            - 多台站数据的 3D 可视化展示
            - 振幅变化的动态监控
            - 历史数据的批次管理和统计
        - **技术特点**：使用 ObsPy 库处理地震数据，PyDeck 实现 3D 地图展示
        """)
    with col2:
        st.metric("监测台站数", "50+", "↑ 5")
        st.metric("数据更新频率", "30 FPS", "实时")
        st.metric("存储容量", "2.5 TB", "↑ 120 GB")

with tab2:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **海洋牧场智能监控**
        
        - **数据源**：INSITU 海洋观测 parquet 格式数据
        - **存储位置**：华为云 OBS `海洋牧场-文本+时序/` 路径
        - **核心功能**：
            - 海水温度、盐度、生物量实时监测
            - 多参数时间序列分析
            - 站点分布的地理可视化
            - 数据质量控制和筛选
        - **技术特点**：支持大规模 parquet 数据的高效读取和可视化
        """)
    with col2:
        st.metric("观测变量", "15+", "→")
        st.metric("采样间隔", "1 小时", "→")
        st.metric("覆盖海域", "5 个", "↑ 1")

with tab3:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **高原脆弱生态监测**
        
        - **数据源**：无人机遥感影像、地面传感器数据
        - **存储位置**：华为云 OBS `石冰川数据-遥感+无人机/` 路径
        - **核心功能**：
            - 无人机航拍轨迹实时展示
            - 高原植被指数(NDVI)监测
            - 气温和降水量变化分析
            - 生态退化区域识别
        - **技术特点**：集成高德地图瓦片服务，支持 GeoJSON 数据处理
        """)
    with col2:
        st.metric("监测区域", "5 个高原", "→")
        st.metric("NDVI 范围", "0.1-0.8", "↓ 0.05")
        st.metric("图像分辨率", "4K", "→")

with tab4:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **农业智慧管理系统**
        
        - **数据源**：气象站数据、农业产值统计
        - **存储位置**：华为云 OBS `农业智慧-文本+图像+时序/` 路径
        - **核心功能**：
            - 温湿度等环境参数监测
            - 农业产值趋势分析
            - 多国数据对比展示
            - 作物生长模型预测
        - **技术特点**：支持 UN 数据接入，Altair 图表库可视化
        """)
    with col2:
        st.metric("监测站点", "100+", "↑ 12")
        st.metric("数据维度", "4 个", "→")
        st.metric("更新周期", "每小时", "→")

with tab5:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **森林生态系统监测**
        
        - **数据源**：激光雷达数据、RGB 影像、传感器网络
        - **存储位置**：华为云 OBS `森林数据-rgb+激光雷达/` 路径
        - **核心功能**：
            - 森林冠层结构分析
            - 生物多样性评估
            - 碳储量动态监测
            - 病虫害预警系统
        - **技术特点**：融合 LiDAR 点云和光学影像数据
        """)
    with col2:
        st.metric("覆盖面积", "5000 km²", "↑ 500")
        st.metric("树种识别", "20+", "↑ 3")
        st.metric("精度", "95%", "↑ 2%")

# 系统状态（模拟）
st.divider()
st.markdown("## 📊 系统状态")

col1, col2, col3, col4 = st.columns(4)

# 获取当前时间
now = datetime.now()

with col1:
    st.metric(
        label="🟢 系统状态",
        value="正常运行",
        delta="99.9% 可用性"
    )

with col2:
    st.metric(
        label="💾 存储使用",
        value="8.5 TB",
        delta="↑ 256 GB 本周"
    )

with col3:
    st.metric(
        label="📡 活跃连接",
        value="127",
        delta="↑ 15 较昨日"
    )

with col4:
    st.metric(
        label="⚡ 处理速度",
        value="1.2 GB/s",
        delta="↑ 5% 性能提升"
    )

# 快速开始指南
st.divider()
with st.expander("🚀 快速开始指南", expanded=False):
    st.markdown("""
    ### 1. 进入功能模块
    根据您的需求，点击上方三个功能模块按钮之一：
    - **联合封装**：查看多源数据的综合展示
    - **实时封装**：监控实时数据流
    - **可控发布**：管理数据的发布和共享
    
    ### 2. 选择场景
    在左侧边栏的"场景选择"下拉菜单中，选择您要监测的环境类型。
    
    ### 3. 数据交互
    - 使用滑块调整时间范围
    - 点击地图标记查看详细信息
    - 下载 CSV 格式的分析结果
    - 使用筛选器精确定位数据
    
    ### 4. 配置说明
    - **OBS配置**：在各模块中可自定义 OBS endpoint、bucket 和 prefix
    - **刷新频率**：实时模块支持自定义数据更新间隔
    - **数据缓存**：系统自动缓存常用数据以提升性能
    """)

# 页脚信息
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>© 2025 科学数据流管理平台 | 基于 Streamlit 构建 | 数据存储于华为云 OBS</p>
    <p>最后更新时间：{}</p>
</div>
""".format(now.strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# 添加侧边栏信息
with st.sidebar:
    st.markdown("### ℹ️ 平台简介")
    st.info("""
    实现对多源异构科学数据与知识要素的统一封装与高效管理，具备联合封装、实时封装与可控发布三大核心功能，支持从外部数据处理、动态筛选、多模态可视化到权限管理、数据表格分析及导出的全流程操作。
    
    **当前版本**: v1.0.0  
    **更新日期**: 2025-09
    """)
    
    st.markdown("### 📚 相关资源")
    st.markdown("""
    - [GitHub项目仓库](https://github.com/vixuowis/sci-flow)
    """)