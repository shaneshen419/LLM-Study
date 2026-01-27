# AI Agent学习记录

**@shane shen**

## 💡 更新状态

- 20250911：搭建框架，实现了旅行规划、图像内容识别模块，目前只是最基本的内容，没有涉及到其他MCP、langchain等框架内容，只是实现了简单的LLM调用，prompts设计。



## **🧪**快速开始

```python
# 环境要求
- python 3.8+
- streamlit
- 其他依赖见 requirement.txt

# 安装依赖
pip install -r requirements.txt

# 开始执行
streamlit run main.py --server.port 8501 --server.address 0.0.0.0

# 或者一键运行
bash run.sh
```

## 🎉 实现的功能模块

##### 1.  ✈️ 智能旅行规划师**

```python
- 🗺️ 智能行程规划
- 📅 日历文件导出
- 🏨 住宿餐饮推荐
- 💰 预算估算建议
- 🚀 流式实时生成
```

##### 2. **🖼️ 智能图像识别**

```python
- 🔍 智能内容识别
- 🎨 多维度分析
- 📊 结构化输出
- 📄 分析报告导出
- ⚡ 流式实时分析
```

## 项目结构

```
project/
├── main.py                    # 主应用入口
├── config/
│   ├── __init__.py
│   ├── config.yaml           # 主配置文件
│   ├── models.yaml           # 模型配置
│   └── config_manager.py     # 配置管理器
├── pages/
│   ├── __init__.py
|	├── image_contetn_recognition_agent	# 图像识别模块
|	|	├── __init__py
|	|	├──image_content_recognition.py
|	├── traval_agent			# 旅行规划模块
|	|	├── __init__.py
|	|	├── travel_agent_shane.py
├── utils/
│   ├── __init__.py
│   ├── llm_client.py         # LLM客户端
│   └── common.py             # 通用工具
├── .streamlit/
│   └── config.toml           # Streamlit配置
└── requirements.txt          # 依赖包

```




## 🚀新增模块流程
1. ##### 在 pages/ 下创建新文件

```python
pages/
├── __init__.py
├── image_content_recognition_agent	# 图像内容识别
├── traval_agent 					# 旅行规划
├── new_agent 						# 新增的模块
```

2. ##### 在config/config.yaml中配置

```python
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
```

3. ##### 在utils/llm_client.py中添加功能类

```python
# 在文件末尾添加新的功能类
class YourNewAgentLLM(BaseLLMClient):
    """你的新功能LLM客户端"""
    
    def __init__(self, api_key: str, base_url: str = None, model: str = "qwen-plus"):
        """初始化客户端"""
        super().__init__(api_key, base_url, model)
    
    def your_main_function(self, input_param: str) -> str:
        """主要功能方法"""
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
        """流式生成版本"""
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
        """构建提示词"""
        return f"""
        你是一个专业的AI助手。

        用户输入：{input_param}

        请提供详细的回答...
        """
```

4. ##### 创建页面文件模板

```python
import streamlit as st
import time
from utils.llm_client import YourNewAgentLLM  # 导入你的功能类
from utils.common import format_model_description
from config import config_manager

def show_page():
    """显示你的功能页面"""
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

def init_session_state():
    """初始化session state"""
    if 'new_agent_result' not in st.session_state:
        st.session_state.new_agent_result = None
    if 'new_agent_processing' not in st.session_state:
        st.session_state.new_agent_processing = False

def render_sidebar(api_key):
    """渲染侧边栏"""
    with st.sidebar:
        st.header("🔧 配置设置")
        
        # API状态
        render_api_status(api_key)
        
        st.divider()
        
        # 模型选择
        model = render_model_selection()
        
        st.divider()
        
        # 生成设置
        enable_streaming, chunk_delay = render_generation_settings()
        
        st.divider()
        
        # 帮助说明
        render_help_section()
    
    return model, enable_streaming, chunk_delay

def render_api_status(api_key):
    """渲染API状态"""
    st.subheader("🔑 API状态")
    
    if api_key:
        masked_key = config_manager.mask_api_key(api_key)
        st.success(f"✅ API密钥已配置")
        st.caption(f"🔐 密钥: `{masked_key}`")
        
        base_url = config_manager.get_base_url("your_new_agent")
        if base_url:
            st.caption(f"🌐 服务地址: `{base_url}`")
    else:
        st.error("❌ 未配置API密钥")

# ... 其他方法按需实现

```

5. ##### 更新主应用导航

```python
import pages.your_new_agent as your_new_agent

# 在页面字典中添加
pages = {
    "🏠 首页": home,
    "✈️ 旅行规划": travel_agent, 
    "🖼️ 图像识别": image_recognition,
    "🤖 新功能": your_new_agent,  # 🆕 添加这行
}
```

## 🔄 开发流程总结

```python
graph TD
    A[创建页面文件] --> B[配置config.yaml]
    B --> C[添加LLM功能类]
    C --> D[实现页面逻辑]
    D --> E[更新主应用导航]
    E --> F[测试功能]
    F --> G[完成开发]
```

### 访问地址

```python
访问地址
主页: http://localhost:8501
旅行规划: http://localhost:8501/?page=travel_agent
图像识别: http://localhost:8501/?page=image_recognition
README: http://localhost:8501/readme
```

### 模型配置

支持多种大语言模型，包括：

- 🔥 Qwen3 系列（最新）
- 💻 代码专用模型
- 🧮 数学专用模型
- 🤔 推理专用模型
- 🎯 多模态模型

## 🔒 安全特性

- API密钥加密存储
- 界面密钥遮蔽显示
- 文件上传安全检查
- 配置文件验证

## 📈 技术架构

- **前端框架**: Streamlit
- **配置管理**: YAML配置文件
- **模型支持**: 阿里云DashScope、OpenAI等
- **流式处理**: 实时响应生成
- **多模态**: 支持文本和图像处理
