import streamlit as st
import time
import os
from PIL import Image
from utils.llm_client import VisionLLMClient
from utils.common import (
    format_model_description, 
    process_uploaded_image, 
    create_analysis_report,
    validate_image_file
)
from config import config_manager

def image_contetn_recognition_show_page():
    """显示图像识别页面"""
    page_config = config_manager.get_page_config("image_recognition")
    
    st.title(f"{page_config.get('icon', '🖼️')} {page_config.get('title', '图像识别')}")
    st.caption(page_config.get('description', '使用先进的视觉大模型智能识别和分析图片内容'))
    
    # 加载自定义CSS
    load_custom_css()
    
    # 初始化session state
    init_session_state()
    
    # 从配置获取API设置
    api_key = config_manager.get_api_key("image_recognition")
    base_url = config_manager.get_base_url("image_recognition")
    
    # 侧边栏配置
    model, analysis_type, enable_streaming, chunk_delay = render_sidebar(api_key)
    
    # 检查API密钥
    if not api_key:
        render_no_api_key_warning()
        return
    
    # 图片上传区域
    uploaded_file = render_upload_area()
    
    # 处理上传的图片
    if uploaded_file:
        handle_uploaded_image(uploaded_file)
    
    # 分析控制区域
    if st.session_state.image_uploaded_image:
        handle_analysis_controls(
            api_key, base_url, model, analysis_type, 
            enable_streaming, chunk_delay, uploaded_file
        )
    
    # 显示已有的分析结果
    display_existing_results(uploaded_file)
    
    # 页脚
    render_footer(model)


def load_custom_css():
    """加载自定义CSS样式"""
    ui_config = config_manager.get_ui_config()
    
    st.markdown(f"""
    <style>
    .upload-area {{
        border: 2px dashed {ui_config.get('primary_color', '#667eea')};
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        background-color: #f8f9fa;
    }}
    .analysis-container {{
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: {ui_config.get('shadows', {}).get('card', '0 2px 4px rgba(0,0,0,0.1)')};
    }}
    .image-preview {{
        border-radius: 10px;
        box-shadow: {ui_config.get('shadows', {}).get('card', '0 2px 4px rgba(0,0,0,0.1)')};
    }}
    .metric-card {{
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: {ui_config.get('shadows', {}).get('card', '0 2px 4px rgba(0,0,0,0.1)')};
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    """初始化session state"""
    if 'image_analysis_result' not in st.session_state:
        st.session_state.image_analysis_result = None
    if 'image_uploaded_image' not in st.session_state:
        st.session_state.image_uploaded_image = None
    if 'image_analyzing' not in st.session_state:
        st.session_state.image_analyzing = False
    if 'image_file_name' not in st.session_state:
        st.session_state.image_file_name = ""


def render_sidebar(api_key):
    """渲染侧边栏配置"""
    with st.sidebar:
        st.header("🔧 识别配置")
        
        # API状态显示
        render_api_status(api_key)
        
        st.divider()
        
        # 模型选择
        model = render_model_selection()
        
        st.divider()
        
        # 分析类型选择
        analysis_type = render_analysis_type_selection()
        
        st.divider()
        
        # 生成设置
        enable_streaming, chunk_delay = render_generation_settings()
        
        st.divider()
        
        # 使用说明
        render_help_section()
    
    return model, analysis_type, enable_streaming, chunk_delay


def render_api_status(api_key):
    """渲染API状态"""
    st.subheader("🔑 API状态")
    
    if api_key:
        # 显示遮蔽的API密钥
        masked_key = config_manager.mask_api_key(api_key)
        st.success(f"✅ API密钥已配置")
        st.caption(f"🔐 密钥: `{masked_key}`")
        
        # 显示base URL
        base_url = config_manager.get_base_url("image_recognition")
        if base_url:
            st.caption(f"🌐 服务地址: `{base_url}`")
        
        # 可选：允许临时覆盖API密钥
        if config_manager.should_allow_api_key_override():
            with st.expander("🛠️ 临时覆盖设置"):
                override_key = st.text_input(
                    "临时API密钥",
                    type="password",
                    help="如需临时使用其他API密钥",
                    key="temp_vision_api_key"
                )
                if override_key:
                    st.session_state.temp_vision_api_key = override_key
                    st.info("✅ 已设置临时API密钥")
    else:
        st.error("❌ 未配置API密钥")
        st.markdown("""
        请在 `config/config.yaml` 中配置：
        ```yaml
        api:
          default_api_key: "your-api-key-here"
        ```
        """)


def render_model_selection():
    """渲染模型选择"""
    vision_models = config_manager.get_all_models("vision")
    default_model = config_manager.get_page_config("image_recognition").get("default_model", "qwen-vl-plus")
    default_index = vision_models.index(default_model) if default_model in vision_models else 0
    
    model = st.selectbox(
        "模型选择",
        options=vision_models,
        index=default_index,
        help="选择视觉识别模型",
        key="image_model"
    )
    
    # 显示模型信息
    model_info = config_manager.get_model_info(model)
    if model_info:
        st.info(f"ℹ️ {model_info.get('description', format_model_description(model))}")
    else:
        st.info(f"ℹ️ {format_model_description(model)}")
    
    return model


def render_analysis_type_selection():
    """渲染分析类型选择"""
    st.subheader("📊 分析设置")
    analysis_modes = config_manager.get_analysis_modes()
    mode_options = list(analysis_modes.keys())
    
    analysis_type = st.radio(
        "分析模式",
        options=mode_options,
        format_func=lambda x: analysis_modes.get(x, {}).get('name', x),
        help="选择分析的详细程度和风格",
        key="image_analysis_type"
    )
    
    # 显示分析模式说明
    mode_info = analysis_modes.get(analysis_type, {})
    if mode_info.get('description'):
        st.caption(f"💡 {mode_info['description']}")
    
    return analysis_type


def render_generation_settings():
    """渲染生成设置"""
    streaming_config = config_manager.get_streaming_config()
    
    enable_streaming = st.checkbox(
        "启用流式生成",
        value=streaming_config.get("enabled", True),
        help="实时显示分析过程",
        key="image_streaming"
    )
    
    chunk_delay = streaming_config.get("default_delay", 20)
    if enable_streaming:
        chunk_delay = st.slider(
            "显示延迟 (毫秒)",
            min_value=streaming_config.get("min_delay", 10),
            max_value=streaming_config.get("max_delay", 100),
            value=streaming_config.get("default_delay", 20),
            help="调整文本显示速度",
            key="image_delay"
        )
    
    return enable_streaming, chunk_delay


def render_help_section():
    """渲染帮助说明"""
    with st.expander("📖 使用说明"):
        upload_config = config_manager.get_upload_config()
        supported_formats = upload_config.get("supported_image_formats", ["jpg", "jpeg", "png"])
        max_size = upload_config.get("max_file_size", 10)
        
        st.markdown(f"""
        **步骤：**
        1. 确认API密钥已配置 ✅
        2. 选择分析模式
        3. 上传图片文件
        4. 点击开始分析
        5. 查看结果并下载报告
        
        **支持格式：**
        {' • '.join(supported_formats).upper()}
        
        **文件限制：**
        - 最大文件大小: {max_size}MB
        - 建议分辨率: 1920x1920以下
        
        **分析模式：**
        """)
        
        analysis_modes = config_manager.get_analysis_modes()
        for mode_key, mode_info in analysis_modes.items():
            st.markdown(f"- {mode_info.get('name', mode_key)}: {mode_info.get('description', '')}")


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


def render_upload_area():
    """渲染图片上传区域"""
    st.subheader("📤 上传图片")
    
    upload_config = config_manager.get_upload_config()
    supported_formats = upload_config.get("supported_image_formats", ["jpg", "jpeg", "png"])
    max_size_mb = upload_config.get("max_file_size", 10)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "选择图片文件",
            type=supported_formats,
            disabled=st.session_state.image_analyzing,
            help=f"支持格式: {', '.join(supported_formats).upper()}，最大{max_size_mb}MB",
            key="image_file_uploader"
        )
    
    with col2:
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(f"""
            <div class="metric-card">
                <h4>📁 文件信息</h4>
                <p><strong>名称:</strong> {uploaded_file.name}</p>
                <p><strong>大小:</strong> {file_size_mb:.2f} MB</p>
                <p><strong>类型:</strong> {uploaded_file.type}</p>
            </div>
            """, unsafe_allow_html=True)
    
    return uploaded_file


def handle_uploaded_image(uploaded_file):
    """处理上传的图片"""
    upload_config = config_manager.get_upload_config()
    max_size_mb = upload_config.get("max_file_size", 10)
    
    # 验证文件
    is_valid, error_msg = validate_image_file(uploaded_file, max_size_mb)
    if not is_valid:
        st.error(f"❌ {error_msg}")
        return
    
    try:
        # 处理图片
        max_size = tuple(upload_config.get("image_max_size", [1920, 1920]))
        image = process_uploaded_image(uploaded_file, max_size)
        st.session_state.image_uploaded_image = image
        st.session_state.image_file_name = uploaded_file.name
        
        # 显示图片预览
        render_image_preview(image, uploaded_file.name)
        
    except Exception as e:
        st.error(f"❌ 图片处理失败: {str(e)}")
        st.session_state.image_uploaded_image = None


def render_image_preview(image, filename):
    """渲染图片预览"""
    st.subheader("🖼️ 图片预览")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(
            image, 
            caption=f"预览: {filename}", 
            use_column_width=True,
            clamp=True
        )
    
    # 显示图片信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔍 宽度", f"{image.size[0]}px")
    with col2:
        st.metric("📏 高度", f"{image.size[1]}px")
    with col3:
        st.metric("🎨 模式", image.mode)


def handle_analysis_controls(api_key, base_url, model, analysis_type, enable_streaming, chunk_delay, uploaded_file):
    """处理分析控制"""
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button(
            "🔍 开始分析" if not st.session_state.image_analyzing else "⏳ 分析中...",
            type="primary",
            disabled=st.session_state.image_analyzing,
            use_container_width=True,
            key="image_analyze_btn"
        )
    
    with col2:
        if st.session_state.image_analysis_result and not st.session_state.image_analyzing:
            clear_button = st.button(
                "🗑️ 清除结果",
                use_container_width=True,
                key="image_clear_btn"
            )
            if clear_button:
                st.session_state.image_analysis_result = None
                st.rerun()
    
    # 开始分析
    if analyze_button:
        perform_image_analysis(
            api_key, base_url, model, analysis_type,
            enable_streaming, chunk_delay, uploaded_file
        )


def perform_image_analysis(api_key, base_url, model, analysis_type, enable_streaming, chunk_delay, uploaded_file):
    """执行图像分析"""
    st.session_state.image_analyzing = True
    st.session_state.image_analysis_result = ""
    
    # 检查是否有临时覆盖的API密钥
    if hasattr(st.session_state, 'temp_vision_api_key') and st.session_state.temp_vision_api_key:
        api_key = st.session_state.temp_vision_api_key
    
    try:
        # 初始化LLM客户端
        llm_client = VisionLLMClient(
            api_key=api_key,
            base_url=base_url.strip() if base_url.strip() else None,
            model=model
        )
        
        # 创建分析容器
        st.divider()
        progress_container = st.container()
        content_container = st.container()
        
        with progress_container:
            st.info(f"🤖 正在使用 {model} 分析图片...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with content_container:
            st.subheader("📊 分析结果")
            content_placeholder = st.empty()
        
        # 执行分析
        if enable_streaming:
            perform_streaming_analysis(
                llm_client, analysis_type, chunk_delay,
                progress_bar, status_text, content_placeholder
            )
        else:
            perform_batch_analysis(
                llm_client, analysis_type,
                progress_bar, status_text, content_placeholder
            )
        
        st.session_state.image_analyzing = False
        
        # 添加下载按钮
        if st.session_state.image_analysis_result:
            render_download_button(uploaded_file, model, analysis_type)
        
        st.success("🎉 图像分析完成！")
        
    except Exception as e:
        st.session_state.image_analyzing = False
        handle_analysis_error(str(e))


def perform_streaming_analysis(llm_client, analysis_type, chunk_delay, progress_bar, status_text, content_placeholder):
    """执行流式分析"""
    accumulated_text = ""
    chunk_count = 0
    streaming_config = config_manager.get_streaming_config()
    cursor_symbol = streaming_config.get("cursor_symbol", "▊")
    
    for chunk in llm_client.analyze_image_stream(st.session_state.image_uploaded_image, analysis_type):
        accumulated_text += chunk
        chunk_count += 1
        
        # 更新显示内容
        with content_placeholder.container():
            if streaming_config.get("show_cursor", True):
                st.markdown(accumulated_text + cursor_symbol)
            else:
                st.markdown(accumulated_text)
        
        # 更新进度
        estimated_progress = min(chunk_count * 0.02, 0.95)
        progress_bar.progress(estimated_progress)
        status_text.text(f"已生成 {len(accumulated_text)} 字符...")
        
        # 控制显示速度
        time.sleep(chunk_delay / 1000.0)
    
    # 完成分析
    st.session_state.image_analysis_result = accumulated_text
    progress_bar.progress(1.0)
    status_text.text("✅ 分析完成！")
    
    # 移除光标效果
    with content_placeholder.container():
        st.markdown(accumulated_text)


def perform_batch_analysis(llm_client, analysis_type, progress_bar, status_text, content_placeholder):
    """执行批量分析"""
    status_text.text("分析中，请稍候...")
    result = llm_client.analyze_image(st.session_state.image_uploaded_image, analysis_type)
    st.session_state.image_analysis_result = result
    progress_bar.progress(1.0)
    status_text.text("✅ 分析完成！")
    
    with content_placeholder.container():
        st.markdown(result)


def handle_analysis_error(error_message):
    """处理分析错误"""
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
        st.error(f"❌ 分析过程中发生错误：{error_message}")
        st.info("💡 请检查配置和网络连接，或联系技术支持")


def render_download_button(uploaded_file, model, analysis_type):
    """渲染下载按钮"""
    if uploaded_file and st.session_state.image_analysis_result:
        report_content = create_analysis_report(
            st.session_state.image_analysis_result,
            uploaded_file.name,
            model,
            analysis_type
        )
        
        filename_base = uploaded_file.name.split('.')[0]
        download_filename = f"image_analysis_{filename_base}_{analysis_type}.md"
        
        st.download_button(
            label="📄 下载分析报告",
            data=report_content,
            file_name=download_filename,
            mime="text/markdown",
            use_container_width=False
        )


def display_existing_results(uploaded_file):
    """显示已有的分析结果"""
    if st.session_state.image_analysis_result and not st.session_state.image_analyzing and not st.session_state.image_uploaded_image:
        st.divider()
        st.subheader("📊 分析结果")
        st.markdown(st.session_state.image_analysis_result)
        
        if uploaded_file:
            render_download_button(uploaded_file, "unknown", "comprehensive")


def render_footer(model):
    """渲染页脚"""
    if not st.session_state.image_analyzing:
        st.divider()
        app_config = config_manager.get_app_config()
        st.markdown(
            f"""
            <div style='text-align: center; color: #666;'>
                <small>🤖 Powered by {model} | 🎨 Built with Streamlit | 📊 Smart Image Analysis | 📱 v{app_config.get('version', '1.0.0')}</small>
            </div>
            """,
            unsafe_allow_html=True
        )
