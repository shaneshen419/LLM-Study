import re
import time
from datetime import datetime, timedelta
from icalendar import Calendar, Event
from PIL import Image
from typing import Tuple

def generate_ics_content(plan_text: str, start_date: datetime = None) -> bytes:
    """
    从旅行行程文本生成ICS日历文件
    
    Args:
        plan_text: 旅行行程文本
        start_date: 可选的开始日期（默认为今天）
    
    Returns:
        bytes: ICS文件内容
    """
    cal = Calendar()
    cal.add('prodid', '-//AI Travel Planner//github.com//')
    cal.add('version', '2.0')
    
    if start_date is None:
        start_date = datetime.today()
    
    # 将行程按天分割
    day_pattern = re.compile(r'Day (\d+)[:\s]+(.*?)(?=Day \d+|$)', re.DOTALL)
    days = day_pattern.findall(plan_text)
    
    if not days:  # 如果没有找到日期模式，创建单个全天事件
        event = Event()
        event.add('summary', "旅行行程")
        event.add('description', plan_text)
        event.add('dtstart', start_date.date())
        event.add('dtend', start_date.date())
        event.add("dtstamp", datetime.now())
        cal.add_component(event)
    else:
        # 处理每一天
        for day_num, day_content in days:
            day_num = int(day_num)
            current_date = start_date + timedelta(days=day_num - 1)
            
            # 为整天创建一个事件
            event = Event()
            event.add('summary', f"第{day_num}天行程")
            event.add('description', day_content.strip())
            
            # 设置为全天事件
            event.add('dtstart', current_date.date())
            event.add('dtend', current_date.date())
            event.add("dtstamp", datetime.now())
            cal.add_component(event)
    
    return cal.to_ical()


def format_model_description(model_name: str) -> str:
    """格式化模型描述信息"""
    model_descriptions = {
        # Qwen3系列
        "qwen3-235b-a22b-instruct-2507": "🔥 Qwen3最大模型，2350亿参数，性能最强",
        "qwen3-coder-480b-a35b-instruct": "💻 Qwen3代码专用，4800亿参数，编程能力极强",
        "qwen3-math-72b-instruct": "🧮 Qwen3数学专用，720亿参数，数学推理能力强",
        "qwq-32b-preview": "🤔 QwQ推理专用，320亿参数，复杂推理能力强",
        
        # 经典模型
        "qwen-turbo": "⚡ 快速响应，适合日常对话",
        "qwen-plus": "⚖️ 平衡性能，适合大多数任务", 
        "qwen-max": "💪 最强性能，适合复杂任务",
        
        # 视觉模型
        "qwen-vl-plus": "🎯 视觉理解专用，平衡性能和速度",
        "qwen-vl-max": "🔍 最强视觉模型，细节识别能力强",
        "qwen2-vl-72b-instruct": "📊 大参数视觉模型，理解能力强",
        "gpt-4o": "🌟 OpenAI最强多模态模型",
        "gpt-4o-mini": "⚡ OpenAI快速多模态模型"
    }
    
    return model_descriptions.get(model_name, "📝 通用模型")


def process_uploaded_image(uploaded_file, max_size: Tuple[int, int] = (1920, 1920)) -> Image.Image:
    """
    处理上传的图片文件
    
    Args:
        uploaded_file: Streamlit上传的文件对象
        max_size: 最大尺寸限制
        
    Returns:
        Image.Image: 处理后的PIL图像对象
    """
    try:
        image = Image.open(uploaded_file)
        
        # 如果图片太大，进行压缩
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        raise Exception(f"图片处理失败: {str(e)}")


def create_analysis_report(analysis_result: str, image_name: str, model_name: str, analysis_type: str) -> str:
    """
    创建可下载的分析报告
    
    Args:
        analysis_result: 分析结果
        image_name: 图片名称
        model_name: 使用的模型名称
        analysis_type: 分析类型
        
    Returns:
        str: 格式化的报告内容
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    analysis_type_names = {
        "comprehensive": "🎯 综合分析",
        "simple": "⚡ 简洁描述",
        "detailed": "🔍 详细报告", 
        "creative": "🎨 创意描述"
    }
    
    type_display = analysis_type_names.get(analysis_type, analysis_type)
    
    report = f"""# 🖼️ 图像分析报告

## 📋 基本信息
- **图片名称**: {image_name}
- **分析时间**: {timestamp}
- **使用模型**: {model_name}
- **分析模式**: {type_display}
- **生成工具**: AI图像识别Agent

---

## 📊 分析结果

{analysis_result}

---

## 📝 说明
- 本报告由AI自动生成，仅供参考
- 分析结果基于视觉信息，可能存在主观性
- 建议结合专业知识进行综合判断

*© AI智能助手平台 - 图像识别模块*
"""
    return report


def validate_image_file(uploaded_file, max_size_mb: int = 10) -> Tuple[bool, str]:
    """
    验证上传的图片文件
    
    Args:
        uploaded_file: 上传的文件
        max_size_mb: 最大文件大小(MB)
        
    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    if uploaded_file is None:
        return False, "未选择文件"
    
    # 检查文件大小
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"文件过大，请选择小于{max_size_mb}MB的图片"
    
    # 检查文件类型
    allowed_types = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in allowed_types:
        return False, f"不支持的文件格式，请选择: {', '.join(allowed_types)}"
    
    return True, ""
