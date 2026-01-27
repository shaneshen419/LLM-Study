import yaml
import os
import re
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
from pathlib import Path
import copy

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        """初始化配置管理器"""
        self.config_dir = Path(config_dir)
        self._main_config = None
        self._models_config = None
        self._workflows_config = None
        
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        file_path = self.config_dir / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            st.error(f"配置文件未找到: {file_path}")
            return {}
        except yaml.YAMLError as e:
            st.error(f"配置文件格式错误: {e}")
            return {}
    
    @property
    def main_config(self) -> Dict[str, Any]:
        """获取主配置"""
        if self._main_config is None:
            self._main_config = self._load_yaml("config.yaml")
        return self._main_config
    
    @property
    def models_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        if self._models_config is None:
            self._models_config = self._load_yaml("models.yaml")
        return self._models_config
    
    @property
    def workflows_config(self) -> Dict[str, Any]:
        """获取工作流配置"""
        if self._workflows_config is None:
            self._workflows_config = self._load_yaml("workflows.yaml")
        return self._workflows_config

    def get_workflow(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        获取指定智能体的工作流定义。

        Args:
            agent_name (str): 智能体的名称 (与workflows.yaml中的键对应).

        Returns:
            List[Dict[str, Any]]: 工作流步骤的列表。返回深拷贝以防止会话间状态污染。
        """
        # 通过我们刚刚创建的属性来获取配置
        workflow_config = self.workflows_config.get(agent_name, [])
        # 使用深拷贝确保每个会话的工作流状态是独立的
        return copy.deepcopy(workflow_config)
    
    def get_app_config(self) -> Dict[str, Any]:
        """获取应用配置"""
        return self.main_config.get("app", {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """获取API配置"""
        return self.main_config.get("api", {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """获取安全配置"""
        return self.main_config.get("security", {})
    
    def get_api_key(self, page_name: str = "", base_url: str = "") -> str:
        """
        获取API密钥，按优先级：环境变量 > 页面配置 > 服务商配置 > 全局配置
        
        Args:
            page_name: 页面名称
            base_url: API服务地址
            
        Returns:
            str: API密钥
        """
        api_config = self.get_api_config()
        
        # 1. 优先使用环境变量
        if api_config.get("use_environment_variables", True):
            env_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            if env_key:
                return env_key
        
        # 2. 使用页面特定配置
        if page_name:
            page_config = self.get_page_config(page_name)
            page_api_key = page_config.get("api_key", "")
            if page_api_key:
                return page_api_key
        
        # 3. 使用特定服务商配置
        if base_url:
            base_urls = self.get_base_urls()
            for url_config in base_urls:
                if base_url == url_config.get("url", ""):
                    service_key = url_config.get("api_key", "")
                    if service_key:
                        return service_key
        
        # 4. 使用全局默认配置
        return api_config.get("default_api_key", "")
    
    def get_base_url(self, page_name: str = "") -> str:
        """
        获取基础URL
        
        Args:
            page_name: 页面名称
            
        Returns:
            str: 基础URL
        """
        # 1. 优先使用环境变量
        api_config = self.get_api_config()
        if api_config.get("use_environment_variables", True):
            env_base_url = os.getenv("OPENAI_API_BASE") or os.getenv("API_BASE_URL")
            if env_base_url:
                return env_base_url
        
        # 2. 使用页面特定配置
        if page_name:
            page_config = self.get_page_config(page_name)
            page_base_url = page_config.get("base_url", "")
            if page_base_url:
                return page_base_url
        
        # 3. 使用全局默认配置
        return api_config.get("default_base_url", "")
    
    def get_page_config(self, page_name: str) -> Dict[str, Any]:
        """获取页面配置"""
        pages_config = self.main_config.get("pages", {})
        return pages_config.get(page_name, {})
    
    def get_streaming_config(self) -> Dict[str, Any]:
        """获取流式生成配置"""
        return self.main_config.get("streaming", {})
    
    def get_upload_config(self) -> Dict[str, Any]:
        """获取上传配置"""
        return self.main_config.get("upload", {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """获取UI配置"""
        return self.main_config.get("ui", {})
    
    def get_analysis_modes(self) -> Dict[str, Any]:
        """获取分析模式配置"""
        return self.main_config.get("analysis_modes", {})
    
    def get_model_groups(self) -> Dict[str, Any]:
        """获取模型分组配置"""
        return self.models_config.get("model_groups", {})
    
    def get_vision_models(self) -> Dict[str, Any]:
        """获取视觉模型配置"""
        return self.models_config.get("vision_models", {})
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        model_info = self.models_config.get("model_info", {})
        return model_info.get(model_name, {})
    
    def get_base_urls(self) -> List[Dict[str, str]]:
        """获取可选的base URL列表"""
        api_config = self.get_api_config()
        return api_config.get("alternative_base_urls", [])
    
    def get_all_models(self, model_type: str = "general") -> List[str]:
        """获取所有模型列表"""
        if model_type == "vision":
            model_groups = self.get_vision_models()
        else:
            model_groups = self.get_model_groups()
        
        all_models = []
        for group_name, group_data in model_groups.items():
            models = group_data.get("models", [])
            all_models.extend(models)
        
        # 去重并保持顺序
        seen = set()
        unique_models = []
        for model in all_models:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)
        
        return unique_models
    
    def should_hide_api_key_input(self) -> bool:
        """是否应该隐藏API密钥输入框"""
        security_config = self.get_security_config()
        return security_config.get("hide_api_key_in_ui", False)
    
    def should_allow_api_key_override(self) -> bool:
        """是否允许在界面中覆盖API密钥"""
        security_config = self.get_security_config()
        return security_config.get("allow_api_key_override", True)
    
    def mask_api_key(self, api_key: str) -> str:
        """遮蔽显示API密钥"""
        if not api_key:
            return ""
        
        security_config = self.get_security_config()
        if not security_config.get("mask_api_key_display", True):
            return api_key
        
        if len(api_key) <= 8:
            return "*" * len(api_key)
        
        return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
    
    def validate_config(self) -> List[str]:
        """验证配置文件完整性"""
        errors = []
        
        # 检查必需的配置项
        required_configs = [
            ("app", "title"),
            ("api", "default_base_url"),
            ("pages", "travel_agent"),
            ("pages", "image_recognition")
        ]
        
        for config_path in required_configs:
            current = self.main_config
            for key in config_path:
                if key not in current:
                    errors.append(f"缺少配置项: {'.'.join(config_path)}")
                    break
                current = current[key]
        
        # 检查API密钥
        api_key = self.get_api_key()
        if not api_key:
            errors.append("未配置API密钥，请在配置文件中设置或通过环境变量提供")
        
        return errors

# 全局配置管理器实例
config_manager = ConfigManager()
