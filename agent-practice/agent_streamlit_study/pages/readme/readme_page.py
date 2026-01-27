import streamlit as st
from pathlib import Path
from config import config_manager

def readme_show_page():
    """显示README页面"""
    # 自定义CSS样式
    st.markdown("""
    <style>
        .readme-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .author-info {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        
        .update-info {
            background: #e8f5e8;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
        }
        
        .feature-section {
            background: #fff3cd;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }
        
        .code-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 页面头部
    st.markdown("""
    <div class="readme-header">
        <h1>📖 AI Agent学习记录</h1>
        <p>项目开发文档与使用指南</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 作者信息
    st.markdown("""
    <div class="author-info">
        <strong>👨‍💻 @shane shen</strong><br>
        <strong>联系我：shaneshen@futurefab.ai</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # 更新状态
    st.markdown("## 💡 更新状态")
    st.markdown("""
    <div class="update-info">
        <strong>📅 20250911</strong>：搭建框架，实现了旅行规划、图像内容识别模块，目前只是最基本的内容，没有涉及到其他MCP、langchain等框架内容，只是实现了简单的LLM调用，prompts设计。
        <br><strong>📅 20250912</strong>：加入了README显示模块。
        <br><strong>📅 20250915</strong>：完善了股票预测模块，实现了yfinance数据获取、LSTM模型预测、技术指标分析等功能。
        <br><strong>📅 20250916</strong>：加入了半导体良率分析模块，初步尝试固定步骤执行方式，没有做完，没有输入。
    </div>
    """, unsafe_allow_html=True)
    
    # 快速开始
    st.markdown("## 🧪 快速开始")
    
    st.markdown("**环境要求:**")
    st.markdown("""
    - Python 3.8+
    - Streamlit
    - 其他依赖见 requirements.txt
    """)
    
    st.markdown("**安装依赖:**")
    st.code("pip install -r requirements.txt", language="bash")
    
    st.markdown("**开始执行:**")
    st.code("streamlit run main.py --server.port 8501 --server.address 0.0.0.0", language="bash")
    
    st.markdown("**或者一键运行:**")
    st.code("bash run.sh", language="bash")
    
    # 实现的功能模块
    st.markdown("## 🎉 实现的功能模块")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-section">
            <h4>1. ✈️ 智能旅行规划师</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        - 🗺️ 智能行程规划
        - 📅 日历文件导出
        - 🏨 住宿餐饮推荐
        - 💰 预算估算建议
        - 🚀 流式实时生成
        """)
    
    with col2:
        st.markdown("""
        <div class="feature-section">
            <h4>2. 🖼️ 智能图像识别</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        - 🔍 智能内容识别
        - 🎨 多维度分析
        - 📊 结构化输出
        - 📄 分析报告导出
        - ⚡ 流式实时分析
        """)
    
    with col3:
        st.markdown("""
        <div class="feature-section">
            <h4>3. 📈 股票预测</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        - 📊 历史数据分析
        - 📈 价格预测图表
        - 📐 技术指标分析
        - 📋 预测报告生成
        - ⚡ 实时预测更新
        """)
    
    # 添加半导体良率分析模块介绍
    with col1:
        st.markdown("""
        <div class="feature-section">
            <h4>4. 🔬 半导体良率分析</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        - 🔬 交互式工作流
        - 📝 可观测的步骤
        - ✍️ 对话式修改
        - 📊 结构化输出
        - 🚀 逐步执行控制
        """)
    
    # 项目结构
    st.markdown("## 📁 项目结构")
    st.code("""
project/
├── main.py                    # 主应用入口
├── config/
│   ├── __init__.py
│   ├── config.yaml           # 主配置文件
│   ├── models.yaml           # 模型配置
│   └── config_manager.py     # 配置管理器
├── pages/
│   ├── __init__.py
│   ├── image_content_recognition_agent  # 图像识别模块
│   │   ├── __init__.py
│   │   └── image_content_recognition.py
│   ├── travel_agent          # 旅行规划模块
│   │   ├── __init__.py
│   │   └── travel_agent_shane.py
│   ├── readme/               # README页面模块
│   │   ├── __init__.py
│   │   └── readme_page.py
│   ├── stock_prediction_agent  # 股票预测模块
│   │   ├── __init__.py
│   │   └── stock_prediction.py
│   └── semiconductor_yield_agent  # 半导体良率分析模块
│       ├── __init__.py
│       └── semiconductor_yield.py
├── utils/
│   ├── __init__.py
│   ├── llm_client.py         # LLM客户端
│   └── common.py             # 通用工具
├── .streamlit/
│   └── config.toml           # Streamlit配置
└── requirements.txt          # 依赖包
    """, language="")
    
    # 新增模块流程
    st.markdown("## 🚀 新增模块流程")
    
    # 步骤1
    st.markdown("### 1. 在 pages/ 下创建新文件")
    st.code("""
pages/
├── __init__.py
├── image_content_recognition_agent  # 图像内容识别
├── travel_agent                     # 旅行规划
├── new_agent                        # 新增的模块
    """, language="")
    
    # 步骤2
    st.markdown("### 2. 在config/config.yaml中配置")
    st.code("""
# 页面配置
pages:
  travel_agent:
    # ... 现有配置
    
  image_recognition:
    # ... 现有配置
    
  your_new_agent:  # 🆕 新增配置
    title: "你的功能名称"
    icon: "🤖"  # 选择合适的图标
    description: "功能描述"
    default_model: "qwen-plus"  # 默认模型
    # API配置（可选，覆盖全局配置）
    api_key: ""  # 留空则使用全局配置
    base_url: ""  # 留空则使用全局配置
    features:
      - "🎯 功能特点1"
      - "⚡ 功能特点2"
      - "📊 功能特点3"
    """, language="yaml")
    

    
    # 步骤3
    st.markdown("### 3. 在utils/llm_client.py中添加功能类")
    
    with st.expander("查看代码模板"):
        st.code("""
# 在文件末尾添加新的功能类
class YourNewAgentLLM(BaseLLMClient):
    \"\"\"你的新功能LLM客户端\"\"\"
    
    def __init__(self, api_key: str, base_url: str = None, model: str = "qwen-plus"):
        \"\"\"初始化客户端\"\"\"
        super().__init__(api_key, base_url, model)
    
    def your_main_function(self, input_param: str) -> str:
        \"\"\"主要功能方法\"\"\"
        prompt = self._build_prompt(input_param)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"功能执行失败: {str(e)}")
    
    def your_main_function_stream(self, input_param: str):
        \"\"\"流式生成版本\"\"\"
        prompt = self._build_prompt(input_param)
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"流式功能执行失败: {str(e)}")
    
    def _build_prompt(self, input_param: str) -> str:
        \"\"\"构建提示词\"\"\"
        return f\"\"\"
        你是一个专业的AI助手。

        用户输入：{input_param}

        请提供详细的回答...
        \"\"\"
        """, language="python")
    
    # 步骤4
    st.markdown("### 4. 创建页面文件模板")
    
    with st.expander("查看页面模板"):
        st.code("""
import streamlit as st
import time
from utils.llm_client import YourNewAgentLLM  # 导入你的功能类
from utils.common import format_model_description
from config import config_manager

def show_page():
    \"\"\"显示你的功能页面\"\"\"
    page_config = config_manager.get_page_config("your_new_agent")
    
    st.title(f"{page_config.get('icon', '🤖')} {page_config.get('title', '新功能')}")
    st.caption(page_config.get('description', '新功能描述'))
    
    # 初始化session state
    init_session_state()
    
    # 从配置获取API设置
    api_key = config_manager.get_api_key("your_new_agent")
    base_url = config_manager.get_base_url("your_new_agent")
    
    # 侧边栏配置
    model, enable_streaming, chunk_delay = render_sidebar(api_key)
    
    # 检查API密钥
    if not api_key:
        render_no_api_key_warning()
        return
    
    # 功能主界面
    render_main_interface()
    
    # 处理用户操作
    handle_user_actions(api_key, base_url, model, enable_streaming, chunk_delay)
    
    # 显示结果
    display_results()
    
    # 页脚
    render_footer(model)

# ... 其他方法按需实现
        """, language="python")
    
    # 步骤5
    st.markdown("### 5. 更新主应用导航")
    st.code("""
import pages.your_new_agent as your_new_agent

# 在页面字典中添加
pages = {
    "🏠 首页": home,
    "✈️ 旅行规划": travel_agent, 
    "🖼️ 图像识别": image_recognition,
    "🤖 新功能": your_new_agent,  # 🆕 添加这行
}
    """, language="python")
    
    # 开发流程总结
    st.markdown("## 🔄 开发流程总结")
    
    st.markdown("""
    ```mermaid
    graph TD
        A[创建页面文件] --> B[配置config.yaml]
        B --> C[添加LLM功能类]
        C --> D[实现页面逻辑]
        D --> E[更新主应用导航]
        E --> F[测试功能]
        F --> G[完成开发]
    ```
    """)
    
    # 流程步骤
    st.markdown("""
    **开发步骤：**
    1. 📁 创建页面文件
    2. ⚙️ 配置config.yaml
    3. 🔧 添加LLM功能类
    4. 💻 实现页面逻辑
    5. 🔗 更新主应用导航
    6. 🧪 测试功能
    7. ✅ 完成开发
    """)
    
    # 访问地址
    st.markdown("## 🌐 访问地址")
    st.markdown("默认打开时会显示项目说明文档页面。")
    
    access_info = {
        "🏠 主页": "http://112.124.102.254:8501",
        "📖 README": "http://112.124.102.254:8501/?page=readme",
        "✈️ 旅行规划": "http://112.124.102.254:8501/?page=travel_agent",
        "🖼️ 图像识别": "http://112.124.102.254:8501/?page=image_recognition",
        "📈 股票预测": "http://112.124.102.254:8501/?page=stock_prediction",
        "🔬 半导体良率分析": "http://112.124.102.254:8501/?page=semiconductor_yield"
    }
    
    for name, url in access_info.items():
        st.markdown(f"- **{name}**: `{url}`")
    
    # 模型配置
    st.markdown("## 🤖 模型配置")
    
    st.markdown("支持多种大语言模型，包括：")
    st.markdown("""
    - 🔥 **Qwen3 系列**（最新）
    - 💻 **代码专用模型**
    - 🧮 **数学专用模型**
    - 🤔 **推理专用模型**
    - 🎯 **多模态模型**
    """)
    
    # 安全特性
    st.markdown("## 🔒 安全特性")
    
    security_features = [
        "🔐 API密钥加密存储",
        "👁️ 界面密钥遮蔽显示", 
        "📤 文件上传安全检查",
        "✅ 配置文件验证"
    ]
    
    for feature in security_features:
        st.markdown(f"- {feature}")
    
    # 技术架构
    st.markdown("## 📈 技术架构")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **核心技术：**
        - **前端框架**: Streamlit
        - **配置管理**: YAML配置文件
        - **模型支持**: 阿里云DashScope、OpenAI等
        """)
    
    with col2:
        st.markdown("""
        **技术特性：**
        - **流式处理**: 实时响应生成
        - **多模态**: 支持文本和图像处理
        - **模块化**: 易于扩展的架构设计
        """)

if __name__ == "__main__":
    readme_show_page()
