import streamlit as st
from pages import travel_agent_show_page,image_contetn_recognition_show_page,readme_show_page,stock_prediction_agent_show_page,semiconductor_yield_show_page  
from config import config_manager

def setup_page_config():
    """设置页面配置"""
    app_config = config_manager.get_app_config()
    
    st.set_page_config(
        page_title=app_config.get("title", "AI智能助手平台"),
        page_icon=app_config.get("icon", "🤖"),
        layout=app_config.get("layout", "wide"),
        initial_sidebar_state="expanded"
    )

def load_custom_css():
    """加载自定义CSS样式"""
    ui_config = config_manager.get_ui_config()
    
    css = f"""
    <style>
        .main-header {{
            text-align: center;
            padding: 1rem 0;
            background: {ui_config.get("gradients", {}).get("main_header", "linear-gradient(90deg, #667eea 0%, #764ba2 100%)")};
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }}
        
        .nav-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: {ui_config.get("shadows", {}).get("card", "0 2px 4px rgba(0,0,0,0.1)")};
            margin: 1rem 0;
            border-left: 4px solid {ui_config.get("primary_color", "#667eea")};
        }}
        
        .nav-card:hover {{
            box-shadow: {ui_config.get("shadows", {}).get("card_hover", "0 4px 8px rgba(0,0,0,0.15)")};
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }}
        
        .feature-badge {{
            background: #f0f2f6;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            margin: 0.25rem;
            display: inline-block;
        }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

def main():
    """主应用函数"""
    # 验证配置文件
    config_errors = config_manager.validate_config()
    if config_errors:
        st.error("配置文件验证失败:")
        for error in config_errors:
            st.error(f"- {error}")
        return
    
    # 设置页面配置
    setup_page_config()
    
    # 加载自定义样式
    load_custom_css()
    
    # 获取当前页面
    try:
        query_params = st.query_params
        page = query_params.get("page", "readme")
        if isinstance(page, list):
            page = page[0]
    except:
        page = "readme"
    
    # 侧边栏导航
    with st.sidebar:
        app_config = config_manager.get_app_config()
        
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2>{app_config.get("icon", "🤖")} {app_config.get("title", "AI智能助手")}</h2>
            <p style='color: #666; font-size: 0.9rem;'>选择您需要的服务</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 导航菜单 - 添加readme选项和股票预测选项
        page_options = ["readme", "travel_agent", "image_recognition", "stock_prediction","semiconductor_yield"]
        travel_config = config_manager.get_page_config("travel_agent")
        image_config = config_manager.get_page_config("image_recognition")
        readme_config = config_manager.get_page_config("readme")
        stock_config = config_manager.get_page_config("stock_prediction")
        yield_config = config_manager.get_page_config("semiconductor_yield")
        
        selected_page = st.radio(
            "选择功能模块",
            options=page_options,
            format_func=lambda x: {
                "travel_agent": f"{travel_config.get('icon', '✈️')} {travel_config.get('title', '旅行规划师')}",
                "image_recognition": f"{image_config.get('icon', '🖼️')} {image_config.get('title', '图像识别')}",
                "stock_prediction": f"{stock_config.get('icon', '📈')} {stock_config.get('title', '股票预测')}",
                "readme": f"{readme_config.get('icon', '📖')} {readme_config.get('title', '项目文档')}",
                "semiconductor_yield": f"{yield_config.get('icon', '🔬')} {yield_config.get('title', '半导体良率分析')}"
            }[x],
            index=page_options.index(page) if page in page_options else 0,
            key="page_selector"
        )
        
        # 更新URL参数
        if selected_page != page:
            st.query_params["page"] = selected_page
            st.rerun()
        
        st.divider()
        
        # 功能介绍
        current_config = config_manager.get_page_config(selected_page)
        features = current_config.get("features", [])
        
        st.markdown(f"### {current_config.get('icon', '')} {current_config.get('title', '')}")
        st.markdown("**功能特点：**")
        for feature in features:
            st.markdown(f"- {feature}")
        
        st.divider()
        
        # 版本信息
        version = app_config.get("version", "1.0.0")
        author = app_config.get("author", "AI Team")
        st.markdown(f"""
        ---
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>📱 版本 {version}</p>
        <p>👨‍💻 {author}</p>
        <p>🚀 基于配置化架构</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 主内容区域 - 添加readme页面路由
    if selected_page == "travel_agent":
        travel_agent_show_page()
    elif selected_page == "image_recognition":
        image_contetn_recognition_show_page()
    elif selected_page == "stock_prediction":
        stock_prediction_agent_show_page()
    elif selected_page == "semiconductor_yield":
        semiconductor_yield_show_page()
    elif selected_page == "readme":
        readme_show_page()
    else:
        show_homepage()

def show_homepage():
    """显示首页"""
    app_config = config_manager.get_app_config()
    
    # 主标题
    st.markdown(f"""
    <div class="main-header">
        <h1>{app_config.get("icon", "🤖")} {app_config.get("title", "AI智能助手平台")}</h1>
        <p>为您提供智能旅行规划和图像识别服务</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 功能模块展示 - 添加readme卡片和股票预测卡片
    col1, col2, col3, col4 = st.columns(4)
    
    travel_config = config_manager.get_page_config("travel_agent")
    image_config = config_manager.get_page_config("image_recognition")
    readme_config = config_manager.get_page_config("readme")
    stock_config = config_manager.get_page_config("stock_prediction")
    
    travel_config = config_manager.get_page_config("travel_agent")
    image_config = config_manager.get_page_config("image_recognition")
    readme_config = config_manager.get_page_config("readme")

    with col1:
        features_html = "".join([f'<span class="feature-badge">{f}</span>' for f in readme_config.get("features", [])])
        st.markdown(f"""
        <div class="nav-card">
            <h3>{readme_config.get("icon", "📖")} {readme_config.get("title", "项目说明文档")}</h3>
            <p>{readme_config.get("description", "")}</p>
            <div>{features_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("查看文档 →", key="readme_btn", use_container_width=True):
            st.query_params["page"] = "readme"
            st.rerun()
    
    with col2:
        features_html = "".join([f'<span class="feature-badge">{f}</span>' for f in travel_config.get("features", [])])
        st.markdown(f"""
        <div class="nav-card">
            <h3>{travel_config.get("icon", "✈️")} {travel_config.get("title", "智能旅行规划师")}</h3>
            <p>{travel_config.get("description", "")}</p>
            <div>{features_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("开始规划旅行 →", key="travel_btn", use_container_width=True):
            st.query_params["page"] = "travel_agent"
            st.rerun()
    
    with col3:
        features_html = "".join([f'<span class="feature-badge">{f}</span>' for f in image_config.get("features", [])])
        st.markdown(f"""
        <div class="nav-card">
            <h3>{image_config.get("icon", "🖼️")} {image_config.get("title", "智能图像识别")}</h3>
            <p>{image_config.get("description", "")}</p>
            <div>{features_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("开始分析图像 →", key="image_btn", use_container_width=True):
            st.query_params["page"] = "image_recognition"
            st.rerun()
    
    with col4:
        features_html = "".join([f'<span class="feature-badge">{f}</span>' for f in stock_config.get("features", [])])
        st.markdown(f"""
        <div class="nav-card">
            <h3>{stock_config.get("icon", "📈")} {stock_config.get("title", "股票预测")}</h3>
            <p>{stock_config.get("description", "")}</p>
            <div>{features_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("开始股票预测 →", key="stock_btn", use_container_width=True):
            st.query_params["page"] = "stock_prediction"
            st.rerun()
        features_html = "".join([f'<span class="feature-badge">{f}</span>' for f in image_config.get("features", [])])
        st.markdown(f"""
        <div class="nav-card">
            <h3>{image_config.get("icon", "🖼️")} {image_config.get("title", "智能图像识别")}</h3>
            <p>{image_config.get("description", "")}</p>
            <div>{features_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("开始分析图像 →", key="image_btn", use_container_width=True):
            st.query_params["page"] = "image_recognition"
            st.rerun()

if __name__ == "__main__":
    main()
