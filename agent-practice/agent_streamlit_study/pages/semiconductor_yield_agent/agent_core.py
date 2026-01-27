import time
import streamlit as st
from copy import deepcopy

# 在真实项目中，这些应该从 utils 导入
# from utils.llm_client import LLMClient
# from utils.db_client import DBClient

# --- 为了Demo独立运行，我们先在这里模拟 ---
def run_llm_step(prompt):
    st.toast(f"🤖 正在调用LLM分析...")
    time.sleep(2)
    return {"analysis": "这是一个基于prompt的模拟分析结果。", "confidence": 0.95}

def run_db_step(query):
    st.toast(f"🔍 正在查询数据库...")
    time.sleep(1.5)
    return [{"wafer_id": "W001", "yield": 88.5, "defect_code": "D-45"}, {"wafer_id": "W002", "yield": 89.1, "defect_code": "D-45"}]

def run_modifier_agent(original_params, user_instruction):
    st.toast(f"🧠 修改Agent正在理解您的指令...")
    time.sleep(2)
    if "G-2046" in user_instruction:
        new_params = deepcopy(original_params)
        new_params["query_template"] = "SELECT * FROM wafer_lots WHERE lot_id = 'G-2046'"
        st.success("参数已成功修改！")
        return new_params
    st.warning("未能完全理解修改指令，参数未变。")
    return original_params
# --- 模拟结束 ---


class YieldAnalysisAgent:
    def __init__(self, workflow_definition):
        self.steps = workflow_definition
        self.current_step_index = 0

    def reset(self, user_input=""):
        self.current_step_index = 0
        for step in self.steps:
            step['status'] = 'pending'
            step['result'] = None
        # 将用户输入注入到第一个步骤的prompt中
        if user_input and self.steps:
            self.steps[0]['params']['prompt_template'] = self.steps[0]['params']['prompt_template'].format(user_input=user_input)

    def get_current_step(self):
        if self.is_finished():
            return None
        return self.steps[self.current_step_index]

    def run_next_step(self):
        if self.is_finished():
            return

        step_info = self.get_current_step()
        step_info['status'] = 'running'

        # 动态替换参数
        params = self._replace_params(step_info['params'])

        result = None
        if step_info['type'] == 'llm_call':
            result = run_llm_step(params['prompt_template'])
        elif step_info['type'] == 'db_query':
            result = run_db_step(params['query_template'])
        
        step_info['result'] = result
        step_info['status'] = 'completed'
        self.current_step_index += 1

    def modify_and_reset_from_step(self, step_index, user_instruction):
        step_to_modify = self.steps[step_index]
        new_params = run_modifier_agent(step_to_modify['params'], user_instruction)
        self.steps[step_index]['params'] = new_params

        # 重置工作流到当前步骤
        self.current_step_index = step_index
        # 清理当前及后续步骤的状态和结果
        for i in range(step_index, len(self.steps)):
            self.steps[i]['status'] = 'pending'
            self.steps[i]['result'] = None

    def is_finished(self):
        return self.current_step_index >= len(self.steps)

    def _replace_params(self, params):
        """用前面步骤的结果替换prompt/query中的占位符"""
        # 这是一个简化实现，真实场景需要更复杂的模板引擎
        filled_params = deepcopy(params)
        for key, value in filled_params.items():
            if isinstance(value, str):
                if "{step_2_result}" in value and self.steps[1]['result']:
                    filled_params[key] = value.replace("{step_2_result}", str(self.steps[1]['result']))
        return filled_params

