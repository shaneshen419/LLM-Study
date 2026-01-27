import streamlit as st
import time
from utils.llm_client import TravelPlannerLLM
from utils.common import generate_ics_content, format_model_description
from config import config_manager

def travel_agent_show_page():
    """显示旅行规划页面"""
    page_config = config_manager.get_page_config("travel_agent")
    
    st.title(f"{page_config.get('icon', '✈️')} {page_config.get('title', '旅行规划师')}")
    st.caption(page_config.get('description', '使用AI智能规划您的下一次冒险之旅'))
    
    # 初始化session state
    init_session_state()
    
    # 从配置获取API设置
    api_key = config_manager.get_api_key("travel_agent")
    base_url = config_manager.get_base_url("travel_agent")
    
    # 侧边栏配置
    model, enable_streaming, chunk_delay = render_sidebar(api_key)
    
    # 检查API密钥
    if not api_key:
        render_no_api_key_warning()
        return
    
    # 旅行参数输入
    destination, num_days = render_travel_inputs()
    
    # 操作按钮
    generate_button, clear_button = render_control_buttons()
    
    # 生成行程
    if generate_button and destination:
        perform_travel_planning(
            api_key, base_url, model, destination, num_days,
            enable_streaming, chunk_delay
        )
    
    # 显示已生成的行程
    display_existing_itinerary()
    
    # 页脚
    render_footer(model)


def init_session_state():
    """初始化session state"""
    if 'travel_itinerary' not in st.session_state:
        st.session_state.travel_itinerary = None
    if 'travel_destination' not in st.session_state:
        st.session_state.travel_destination = ""
    if 'travel_num_days' not in st.session_state:
        st.session_state.travel_num_days = 4
    if 'travel_generating' not in st.session_state:
        st.session_state.travel_generating = False


def render_sidebar(api_key):
    """渲染侧边栏配置"""
    with st.sidebar:
        st.header("🔧 配置设置")
        
        # API状态显示
        render_api_status(api_key)
        
        st.divider()
        
        # 模型选择
        model = render_model_selection()
        
        st.divider()
        
        # 生成设置
        enable_streaming, chunk_delay = render_generation_settings()
        
        st.divider()
        
        # 使用说明
        render_help_section()
    
    return model, enable_streaming, chunk_delay


def render_api_status(api_key):
    """渲染API状态"""
    st.subheader("🔑 API状态")
    
    if api_key:
        # 显示遮蔽的API密钥
        masked_key = config_manager.mask_api_key(api_key)
        st.success(f"✅ API密钥已配置")
        st.caption(f"🔐 密钥: `{masked_key}`")
        
        # 显示base URL
        base_url = config_manager.get_base_url("travel_agent")
        if base_url:
            st.caption(f"🌐 服务地址: `{base_url}`")
        
        # 可选：允许临时覆盖API密钥
        if config_manager.should_allow_api_key_override():
            with st.expander("🛠️ 临时覆盖设置"):
                override_key = st.text_input(
                    "临时API密钥",
                    type="password",
                    help="如需临时使用其他API密钥",
                    key="temp_api_key"
                )
                if override_key:
                    st.session_state.temp_api_key = override_key
                    st.info("✅ 已设置临时API密钥")
    else:
        st.error("❌ 未配置API密钥")
        st.markdown("""
        请在 `config/config.yaml` 中配置：
        ```yaml
        api:
          default_api_key: "your-api-key-here"
        ```
        或设置环境变量：
        ```bash
        export OPENAI_API_KEY="your-api-key-here"
        ```
        """)


def render_model_selection():
    """渲染模型选择"""
    models = config_manager.get_all_models("general")
    default_model = config_manager.get_page_config("travel_agent").get("default_model", "qwen-turbo")
    default_index = models.index(default_model) if default_model in models else 0
    
    model = st.selectbox(
        "模型选择",
        options=models,
        index=default_index,
        help="选择要使用的模型",
        key="travel_model"
    )
    
    # 显示模型信息
    model_info = config_manager.get_model_info(model)
    if model_info:
        st.info(f"ℹ️ {model_info.get('description', format_model_description(model))}")
    else:
        st.info(f"ℹ️ {format_model_description(model)}")
    
    return model


def render_generation_settings():
    """渲染生成设置"""
    streaming_config = config_manager.get_streaming_config()
    
    st.subheader("⚡ 生成设置")
    enable_streaming = st.checkbox(
        "启用流式生成",
        value=streaming_config.get("enabled", True),
        help="实时显示生成过程，提升用户体验",
        key="travel_streaming"
    )
    
    chunk_delay = streaming_config.get("default_delay", 30)
    if enable_streaming:
        chunk_delay = st.slider(
            "显示延迟 (毫秒)",
            min_value=streaming_config.get("min_delay", 10),
            max_value=streaming_config.get("max_delay", 100),
            value=streaming_config.get("default_delay", 30),
            help="调整文本显示速度",
            key="travel_delay"
        )
    
    return enable_streaming, chunk_delay


def render_help_section():
    """渲染帮助说明"""
    with st.expander("📖 使用说明"):
        st.markdown("""
        **步骤：**
        1. 确认API密钥已配置 ✅
        2. 选择合适的模型
        3. 输入目的地和天数
        4. 点击生成行程
        5. 下载日历文件(.ics)
        
        **配置方式：**
        - **配置文件**: 在 `config/config.yaml` 中设置
        - **环境变量**: 设置 `OPENAI_API_KEY`
        
        **推荐模型：**
        - **qwen-turbo**: 快速响应，日常使用
        - **qwen-plus**: 平衡性能，推荐使用
        - **qwen3系列**: 最新最强，复杂任务
        """)


def render_no_api_key_warning():
    """渲染无API密钥警告"""
    st.error("❌ 未配置API密钥")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔧 配置文件方式
        
        在 `config/config.yaml` 中添加：
        ```yaml
        api:
          default_api_key: "sk-your-key-here"
          default_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ```
        """)
    
    with col2:
        st.markdown("""
        ### 🌍 环境变量方式
        
        设置环境变量：
        ```bash
        export OPENAI_API_KEY="sk-your-key-here"
        export OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
        ```
        """)
    
    st.info("💡 配置完成后请重新启动应用")


def render_travel_inputs():
    """渲染旅行参数输入"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        destination = st.text_input(
            "🎯 您想去哪里？",
            value=st.session_state.travel_destination,
            placeholder="例如：北京、合肥、浙江...",
            disabled=st.session_state.travel_generating,
            key="travel_dest_input"
        )
        st.session_state.travel_destination = destination
    
    with col2:
        num_days = st.number_input(
            "📅 旅行天数",
            min_value=1,
            max_value=30,
            value=st.session_state.travel_num_days,
            disabled=st.session_state.travel_generating,
            key="travel_days_input"
        )
        st.session_state.travel_num_days = num_days
    
    return destination, num_days


def render_control_buttons():
    """渲染控制按钮"""
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        generate_button = st.button(
            "🚀 生成行程" if not st.session_state.travel_generating else "⏳ 生成中...",
            type="primary",
            disabled=not st.session_state.travel_destination or st.session_state.travel_generating,
            use_container_width=True,
            key="travel_generate_btn"
        )
    
    clear_button = False
    with col2:
        if st.session_state.travel_itinerary and not st.session_state.travel_generating:
            clear_button = st.button(
                "🗑️ 清除",
                use_container_width=True,
                key="travel_clear_btn"
            )
            if clear_button:
                st.session_state.travel_itinerary = None
                st.rerun()
    
    return generate_button, clear_button


def perform_travel_planning(api_key, base_url, model, destination, num_days, enable_streaming, chunk_delay):
    """执行旅行规划"""
    st.session_state.travel_generating = True
    st.session_state.travel_itinerary = ""
    
    # 检查是否有临时覆盖的API密钥
    if hasattr(st.session_state, 'temp_api_key') and st.session_state.temp_api_key:
        api_key = st.session_state.temp_api_key
    
    try:
        # 初始化LLM客户端
        llm_client = TravelPlannerLLM(
            api_key=api_key,
            base_url=base_url.strip() if base_url.strip() else None,
            model=model
        )
        
        # 创建容器用于流式显示
        st.divider()
        progress_container = st.container()
        content_container = st.container()
        
        with progress_container:
            st.info(f"🤖 正在为您规划{destination}的{num_days}天行程...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with content_container:
            st.subheader(f"📋 {destination} {num_days}天行程")
            content_placeholder = st.empty()
        
        # 流式生成
        if enable_streaming:
            perform_streaming_generation(
                llm_client, destination, num_days, chunk_delay,
                progress_bar, status_text, content_placeholder
            )
        else:
            perform_batch_generation(
                llm_client, destination, num_days,
                progress_bar, status_text, content_placeholder
            )
        
        st.session_state.travel_generating = False
        st.success("🎉 行程规划完成！您可以下载日历文件或继续编辑。")
        
    except Exception as e:
        st.session_state.travel_generating = False
        handle_api_error(str(e))


def perform_streaming_generation(llm_client, destination, num_days, chunk_delay, progress_bar, status_text, content_placeholder):
    """执行流式生成"""
    accumulated_text = ""
    chunk_count = 0
    streaming_config = config_manager.get_streaming_config()
    cursor_symbol = streaming_config.get("cursor_symbol", "▊")
    
    for chunk in llm_client.generate_itinerary_stream(destination, num_days):
        accumulated_text += chunk
        chunk_count += 1
        
        # 更新显示内容
        with content_placeholder.container():
            if streaming_config.get("show_cursor", True):
                st.markdown(accumulated_text + cursor_symbol)
            else:
                st.markdown(accumulated_text)
        
        # 更新进度（估算）
        estimated_progress = min(chunk_count * 0.01, 0.95)
        progress_bar.progress(estimated_progress)
        status_text.text(f"已生成 {len(accumulated_text)} 字符...")
        
        # 添加延迟以控制显示速度
        time.sleep(chunk_delay / 1000.0)
    
    # 完成生成
    st.session_state.travel_itinerary = accumulated_text
    progress_bar.progress(1.0)
    status_text.text("✅ 生成完成！")
    
    # 移除光标效果
    with content_placeholder.container():
        st.markdown(accumulated_text)


def perform_batch_generation(llm_client, destination, num_days, progress_bar, status_text, content_placeholder):
    """执行批量生成"""
    status_text.text("生成中，请稍候...")
    itinerary = llm_client.generate_itinerary(destination, num_days)
    st.session_state.travel_itinerary = itinerary
    progress_bar.progress(1.0)
    status_text.text("✅ 生成完成！")
    
    with content_placeholder.container():
        st.markdown(itinerary)


def handle_api_error(error_message):
    """处理API错误"""
    if "401" in error_message or "invalid_api_key" in error_message:
        st.error("❌ API密钥验证失败")
        st.markdown("""
        **可能的原因：**
        - API密钥格式不正确
        - API密钥已过期或被禁用
        - 账户余额不足
        
        **解决方案：**
        1. 检查 `config/config.yaml` 中的API密钥
        2. 登录服务商控制台检查密钥状态
        3. 检查账户余额是否充足
        4. 尝试使用临时覆盖功能测试新密钥
        """)
    elif "403" in error_message:
        st.error("❌ API访问被拒绝")
        st.info("💡 请检查API密钥权限或账户状态")
    elif "429" in error_message:
        st.error("❌ API调用频率超限")
        st.info("💡 请稍后重试，或升级API套餐")
    elif "timeout" in error_message.lower():
        st.error("❌ 网络连接超时")
        st.info("💡 请检查网络连接或稍后重试")
    elif "500" in error_message:
        st.error("❌ 服务器内部错误")
        st.info("💡 API服务暂时不可用，请稍后重试")
    else:
        st.error(f"❌ 生成行程时发生错误：{error_message}")
        st.info("💡 请检查配置和网络连接，或联系技术支持")


def display_existing_itinerary():
    """显示已生成的行程"""
    if st.session_state.travel_itinerary and not st.session_state.travel_generating:
        st.divider()
        
        # 行程标题和下载按钮
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"📋 {st.session_state.travel_destination} {st.session_state.travel_num_days}天行程")
        
        with col2:
            try:
                ics_content = generate_ics_content(st.session_state.travel_itinerary)
                st.download_button(
                    label="📅 下载日历",
                    data=ics_content,
                    file_name=f"{st.session_state.travel_destination}_itinerary.ics",
                    mime="text/calendar",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"生成日历文件时出错：{str(e)}")
        
        # 显示行程内容
        with st.container():
            st.markdown(st.session_state.travel_itinerary)


def render_footer(model):
    """渲染页脚"""
    if not st.session_state.travel_generating:
        st.divider()
        app_config = config_manager.get_app_config()
        st.markdown(
            f"""
            <div style='text-align: center; color: #666;'>
                <small>🤖 Powered by {model} | 🎨 Built with Streamlit | ⚡ Streaming Enabled | 📱 v{app_config.get('version', '1.0.0')}</small>
            </div>
            """,
            unsafe_allow_html=True
        )
