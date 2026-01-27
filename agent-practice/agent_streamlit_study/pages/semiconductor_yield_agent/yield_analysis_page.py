import streamlit as st
from .agent_core import YieldAnalysisAgent
from config.config_manager import config_manager
import time

def show_page():
    # 从配置中获取页面标题和图标
    page_config = config_manager.get_page_config('semiconductor_yield')
    st.title(f"{page_config.get('icon', '🔬')} {page_config.get('title', '半导体良率分析')}")

    # 在session_state中初始化或获取Agent实例
    if 'yield_agent' not in st.session_state:
        workflow_def = config_manager.get_workflow('semiconductor_yield')
        st.session_state.yield_agent = YieldAnalysisAgent(workflow_def)
    
    agent = st.session_state.yield_agent

    # --- 侧边栏控制器 ---
    with st.sidebar:
        st.header("流程控制")
        user_initial_input = st.text_area("请输入您要分析的良率问题", 
                                          "产线A-2区，过去24小时内批次G-2045的良率从95%下降到88%", 
                                          height=150, key="yield_user_input")

        if st.button("🚀 开始/重置工作流"):
            agent.reset(user_initial_input)
            st.rerun()

        run_next_disabled = agent.is_finished()
        if st.button("➡️ 执行下一步", disabled=run_next_disabled):
            with st.spinner(f"执行中..."):
                agent.run_next_step()
            st.rerun()

        if agent.is_finished():
            st.success("🎉 工作流已全部执行完毕！")

    # --- 主界面展示工作流 ---
    st.subheader("工作流步骤")
    for i, step in enumerate(agent.steps):
        status_icon = {"pending": "⏳", "running": "⚙️", "completed": "✅"}[step['status']]
        
        with st.expander(f"{status_icon} 步骤 {step['id']}: {step['name']}", expanded=True):
            st.markdown("**参数:**")
            st.json(step['params'])

            if step['result']:
                st.markdown("**结果:**")
                if isinstance(step['result'], (dict, list)):
                    st.json(step['result'])
                else:
                    st.write(step['result'])

            st.markdown("---")
            if step['status'] != 'running':
                user_modification_input = st.text_area(
                    "对此步骤进行修改(自然语言)", 
                    key=f"mod_input_{step['id']}"
                )
                if st.button("应用修改并从这里开始", key=f"mod_btn_{step['id']}"):
                    if user_modification_input:
                        agent.modify_and_reset_from_step(i, user_modification_input)
                        st.success(f"工作流已重置到步骤 {step['id']}，请点击'执行下一步'继续。")
                        time.sleep(1)
                        st.rerun()

