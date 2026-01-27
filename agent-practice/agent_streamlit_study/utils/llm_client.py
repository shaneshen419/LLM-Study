from typing import Dict, Any, List, Optional, Tuple, Generator
from openai import OpenAI
import re
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta
from icalendar import Calendar, Event

class LLMClient:
    """通用LLM客户端基类"""
    
    def __init__(self, api_key: str, base_url: str = None, model: str = "qwen-turbo"):
        """初始化LLM客户端"""
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
        # 初始化OpenAI客户端
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": None
        }
        
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = OpenAI(**client_kwargs)


class TravelPlannerLLM(LLMClient):
    """旅行规划专用LLM客户端"""
    
    def generate_itinerary_stream(self, destination: str, num_days: int) -> Generator[str, None, None]:
        """流式生成旅行行程"""
        system_prompt = """你是一个专业的旅行规划师，具有丰富的全球旅行知识。
请为用户创建详细的、实用的旅行行程。

要求：
1. 为每一天创建详细的活动安排
2. 包括景点推荐、餐厅建议、住宿选择
3. 提供交通方式、最佳游览时间、预算估计
4. 确保行程节奏合理，不过于紧张
5. 使用清晰的"Day 1:", "Day 2:"等格式标题
6. 考虑当地文化和实际情况"""

        user_prompt = f"""请为{destination}创建一个{num_days}天的详细旅行行程。

请包括：
- 每日具体的景点和活动安排
- 餐厅和美食推荐
- 住宿建议
- 交通提示
- 预算估计
- 当地文化洞察

请确保每天都有明确的"Day X:"标题，方便转换为日历事件。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                top_p=0.9,
                max_tokens=4096,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"生成行程时发生错误: {str(e)}")
    
    def generate_itinerary(self, destination: str, num_days: int) -> str:
        """非流式生成旅行行程"""
        full_text = ""
        for chunk in self.generate_itinerary_stream(destination, num_days):
            full_text += chunk
        return full_text


class VisionLLMClient(LLMClient):
    """视觉识别专用LLM客户端"""
    
    def __init__(self, api_key: str, base_url: str = None, model: str = "qwen-vl-plus"):
        super().__init__(api_key, base_url, model)
    
    def encode_image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """将PIL图像转换为base64编码"""
        buffered = BytesIO()
        
        # 如果是PNG且有透明度，保持PNG格式
        if format.upper() == "PNG" or (hasattr(image, 'mode') and image.mode == 'RGBA'):
            format = "PNG"
        else:
            format = "JPEG"
            # 如果图像有透明度，转换为RGB
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
        
        image.save(buffered, format=format, quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    
    def get_system_prompt(self, analysis_type: str) -> str:
        """根据分析类型获取系统提示"""
        system_prompts = {
            "comprehensive": """你是一个专业的图像分析专家，具有敏锐的视觉洞察力和丰富的知识背景。
请对图像进行全面深入的分析，包括但不限于：

📋 **基本信息识别**
- 主要对象和元素
- 场景和环境描述
- 颜色、光线、构图分析

🎨 **艺术和设计角度**
- 视觉风格和美学特点
- 构图技巧和视觉平衡
- 色彩搭配和氛围营造

🔍 **细节观察**
- 重要的细节和特征
- 背景信息和上下文
- 可能的象征意义或隐含信息

📊 **结构化输出**
请使用Markdown格式，包含适当的emoji和分段，让内容易读且美观。""",

            "simple": """你是一个图像识别助手，请简洁准确地描述图像的主要内容。

包括：
- 主要对象
- 场景描述  
- 关键特征

请用简洁的语言，重点突出最重要的信息。""",

            "detailed": """你是一个详细的图像分析专家，请提供深入全面的图像分析报告。

请从以下维度进行分析：
1. 视觉元素识别
2. 场景和环境分析
3. 技术参数评估（如可见）
4. 艺术和美学评价
5. 可能的用途和意义
6. 改进建议（如适用）

请使用专业的术语和结构化的格式。""",

            "creative": """你是一个富有创意的视觉故事家，请用生动有趣的方式描述这张图片。

可以包括：
- 富有想象力的场景描述
- 可能的故事背景
- 情感和氛围感受
- 创意联想和类比

让描述既准确又富有艺术感染力。"""
        }
        
        return system_prompts.get(analysis_type, system_prompts["comprehensive"])
    
    def analyze_image_stream(self, image: Image.Image, analysis_type: str = "comprehensive") -> Generator[str, None, None]:
        """流式分析图片内容"""
        
        system_prompt = self.get_system_prompt(analysis_type)
        
        user_prompt = f"""请对这张图片进行{analysis_type}分析。

请确保：
1. 准确识别图片中的所有重要元素
2. 使用清晰的结构和格式
3. 包含适当的emoji和标题来美化排版
4. 提供有价值的洞察和信息

请开始你的分析："""
        
        # 编码图像
        base64_image = self.encode_image_to_base64(image)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image,
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                top_p=0.9,
                max_tokens=4096,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"图像分析时发生错误: {str(e)}")
    
    def analyze_image(self, image: Image.Image, analysis_type: str = "comprehensive") -> str:
        """非流式分析图片内容"""
        full_text = ""
        for chunk in self.analyze_image_stream(image, analysis_type):
            full_text += chunk
        return full_text
