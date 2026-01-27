import streamlit as st
import time
import numpy as np
import pandas as pd
from utils.stock_predictor import stock_predictor
from utils.common import format_model_description
from config import config_manager

def stock_prediction_show_page():
    """显示股票预测页面"""
    page_config = config_manager.get_page_config("stock_prediction")
    
    st.title(f"{page_config.get('icon', '📈')} {page_config.get('title', '股票预测')}")
    st.caption(page_config.get('description', '使用LSTM模型进行股票价格预测和分析'))
    
    # 初始化session state
    init_session_state()
    
    # 从配置获取API设置
    api_key = config_manager.get_api_key("stock_prediction")
    base_url = config_manager.get_base_url("stock_prediction")
    
    # 侧边栏配置
    model, enable_streaming, chunk_delay = render_sidebar(api_key)
    
    # 股票参数输入
    ticker, period, test_size, epochs, batch_size, future_days = render_stock_inputs()
    
    # 操作按钮
    generate_button, clear_button = render_control_buttons()
    
    # 生成预测
    if generate_button and ticker:
        perform_stock_prediction(
            ticker, period, test_size, epochs, batch_size, future_days
        )
    
    # 显示已生成的预测结果
    display_existing_predictions()
    
    # 页脚
    render_footer(model)


def init_session_state():
    """初始化session state"""
    if 'stock_ticker' not in st.session_state:
        st.session_state.stock_ticker = "AAPL"
    if 'stock_period' not in st.session_state:
        st.session_state.stock_period = "5y"
    if 'stock_test_size' not in st.session_state:
        st.session_state.stock_test_size = 0.2
    if 'stock_epochs' not in st.session_state:
        st.session_state.stock_epochs = 50
    if 'stock_batch_size' not in st.session_state:
        st.session_state.stock_batch_size = 32
    if 'stock_future_days' not in st.session_state:
        st.session_state.stock_future_days = 30
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'stock_predicted_prices' not in st.session_state:
        st.session_state.stock_predicted_prices = None
    if 'stock_actual_prices' not in st.session_state:
        st.session_state.stock_actual_prices = None
    if 'stock_future_predictions' not in st.session_state:
        st.session_state.stock_future_predictions = None
    if 'stock_technical_data' not in st.session_state:
        st.session_state.stock_technical_data = None
    if 'stock_predicting' not in st.session_state:
        st.session_state.stock_predicting = False
    if 'stock_ma_fig' not in st.session_state:
        st.session_state.stock_ma_fig = None
    if 'stock_rsi_fig' not in st.session_state:
        st.session_state.stock_rsi_fig = None
    if 'stock_macd_fig' not in st.session_state:
        st.session_state.stock_macd_fig = None
    if 'stock_bb_fig' not in st.session_state:
        st.session_state.stock_bb_fig = None
    if 'stock_history_plot' not in st.session_state:
        st.session_state.stock_history_plot = None


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
        
        # 使用说明
        render_help_section()
    
    return model, True, 30  # 股票预测功能不使用流式生成，但保持接口一致


def render_api_status(api_key):
    """渲染API状态"""
    st.subheader("🔑 API状态")
    
    # 对于股票预测功能，我们不需要显示API状态，因为我们使用的是本地的LSTM模型
    st.info("📊 股票预测功能使用本地LSTM模型，无需API密钥")
    
    if api_key:
        # 显示遮蔽的API密钥（如果有）
        masked_key = config_manager.mask_api_key(api_key)
        st.caption(f"💡 系统API密钥: `{masked_key}`")


def render_model_selection():
    """渲染模型选择"""
    st.subheader("🤖 模型选择")
    
    # 对于股票预测功能，我们默认使用本地LSTM模型
    st.info("📈 当前使用本地LSTM模型进行股票预测")
    
    # 获取推荐模型
    models_config = config_manager.models_config
    recommended_models = models_config.get("model_groups", {}).get("recommended", {}).get("models", [])
    
    # 如果有推荐模型，显示选择器（虽然实际使用的是LSTM，但保持UI一致性）
    if recommended_models:
        default_model = config_manager.get_page_config("stock_prediction").get("default_model", "qwen-turbo")
        model = st.selectbox(
            "选择语言模型（用于生成分析报告）",
            options=recommended_models,
            index=recommended_models.index(default_model) if default_model in recommended_models else 0,
            format_func=format_model_description
        )
    else:
        model = "qwen-turbo"
        st.selectbox(
            "选择语言模型（用于生成分析报告）",
            options=[model],
            format_func=lambda x: "qwen-turbo (默认)"
        )
    
    return model


def render_help_section():
    """渲染帮助说明部分"""
    st.subheader("❓ 使用帮助")
    
    with st.expander("📝 如何使用股票预测功能"):
        st.markdown("""
        1. **输入股票代码**：输入您想要预测的股票代码（如AAPL、MSFT等）
        2. **选择数据周期**：选择要使用的历史数据周期
        3. **调整模型参数**：根据需要调整测试集比例、训练轮数等参数
        4. **设置预测天数**：指定要预测未来多少天的股票价格
        5. **点击预测按钮**：开始训练模型并生成预测结果
        6. **查看结果**：查看预测图表和技术指标分析
        
        **注意**：股票预测仅供参考，不构成投资建议。
        """)


def render_stock_inputs():
    """渲染股票预测参数输入"""
    with st.container(border=True):
        st.subheader("📊 股票参数设置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 股票代码输入
            ticker = st.text_input(
                "股票代码",
                value=st.session_state.stock_ticker,
                placeholder="例如: AAPL, MSFT, TSLA, 000001.SS",
                help="输入要预测的股票代码，A股请使用 .SS 或 .SZ 后缀"
            )
            
            # 数据周期选择
            period = st.selectbox(
                "历史数据周期",
                options=["1y", "2y", "5y", "10y", "max"],
                index=["1y", "2y", "5y", "10y", "max"].index(st.session_state.stock_period),
                help="选择用于训练模型的历史数据周期"
            )
        
        with col2:
            # 测试集比例
            test_size = st.slider(
                "测试集比例",
                min_value=0.1, max_value=0.3, value=st.session_state.stock_test_size,
                step=0.05,
                help="测试集占总数据的比例"
            )
            
            # 预测未来天数
            future_days = st.slider(
                "预测未来天数",
                min_value=7, max_value=90, value=st.session_state.stock_future_days,
                step=7,
                help="要预测未来多少天的股票价格"
            )
        
        st.divider()
        
        # 高级参数设置（可折叠）
        with st.expander("⚙️ 高级模型参数设置"):
            col3, col4 = st.columns(2)
            
            with col3:
                # 训练轮数
                epochs = st.slider(
                    "训练轮数 (Epochs)",
                    min_value=10, max_value=200, value=st.session_state.stock_epochs,
                    step=10,
                    help="模型训练的轮数，轮数越多训练越充分，但可能导致过拟合"
                )
            
            with col4:
                # 批次大小
                batch_size = st.selectbox(
                    "批次大小 (Batch Size)",
                    options=[16, 32, 64, 128],
                    index=[16, 32, 64, 128].index(st.session_state.stock_batch_size),
                    help="每次训练使用的数据量"
                )
    
    # 保存到session state
    st.session_state.stock_ticker = ticker
    st.session_state.stock_period = period
    st.session_state.stock_test_size = test_size
    st.session_state.stock_epochs = epochs
    st.session_state.stock_batch_size = batch_size
    st.session_state.stock_future_days = future_days
    
    return ticker, period, test_size, epochs, batch_size, future_days


def render_control_buttons():
    """渲染控制按钮"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        generate_button = st.button(
            "🔮 开始预测",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.stock_predicting
        )
    
    with col2:
        clear_button = st.button(
            "🧹 清除结果",
            use_container_width=True,
            disabled=st.session_state.stock_predicting
        )
    
    # 清除结果
    if clear_button:
        st.session_state.stock_data = None
        st.session_state.stock_predicted_prices = None
        st.session_state.stock_actual_prices = None
        st.session_state.stock_future_predictions = None
        st.session_state.stock_technical_data = None
        st.session_state.stock_ma_fig = None
        st.session_state.stock_rsi_fig = None
        st.session_state.stock_macd_fig = None
        st.session_state.stock_bb_fig = None
        st.session_state.stock_history_plot = None
        st.rerun()
    
    return generate_button, clear_button


def perform_stock_prediction(ticker, period, test_size, epochs, batch_size, future_days):
    """执行股票预测"""
    # 设置预测状态
    st.session_state.stock_predicting = True
    
    try:
        # 显示进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. 获取股票数据
        status_text.text("📊 正在获取股票历史数据...")
        progress_bar.progress(20)
        
        data = stock_predictor.get_stock_data(ticker, period)
        if data.empty:
            st.error("❌ 无法获取股票数据，请检查股票代码是否正确")
            st.session_state.stock_predicting = False
            return
        
        # 保存数据到session state
        st.session_state.stock_data = data
        
        # 2. 预处理数据
        status_text.text("🔄 正在预处理数据...")
        progress_bar.progress(40)
        
        X_train, y_train, X_test, y_test = stock_predictor.preprocess_data(data, test_size)
        
        # 3. 构建和训练模型
        status_text.text(f"🧠 正在训练LSTM模型 (Epochs: {epochs})...")
        progress_bar.progress(60)
        
        history = stock_predictor.train_model(X_train, y_train, epochs, batch_size)
        
        # 4. 进行预测
        status_text.text("🔮 正在进行预测...")
        progress_bar.progress(80)
        
        predicted_prices = stock_predictor.predict(X_test)
        
        # 获取实际价格（反归一化）
        actual_prices = stock_predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # 5. 预测未来价格
        status_text.text(f"🔮 正在预测未来{future_days}天的价格...")
        progress_bar.progress(90)
        
        future_predictions = stock_predictor.predict_future(future_days)
        
        # 6. 计算技术指标
        status_text.text("📈 正在计算技术指标...")
        
        technical_data = stock_predictor.calculate_technical_indicators(data)
        ma_fig, rsi_fig, macd_fig, bb_fig = stock_predictor.create_technical_indicators_plots(technical_data)
        
        # 7. 创建历史和预测图表
        history_plot = stock_predictor.create_plot(actual_prices.flatten(), predicted_prices.flatten(), future_predictions)
        
        # 保存结果到session state
        st.session_state.stock_predicted_prices = predicted_prices
        st.session_state.stock_actual_prices = actual_prices
        st.session_state.stock_future_predictions = future_predictions
        st.session_state.stock_technical_data = technical_data
        st.session_state.stock_ma_fig = ma_fig
        st.session_state.stock_rsi_fig = rsi_fig
        st.session_state.stock_macd_fig = macd_fig
        st.session_state.stock_bb_fig = bb_fig
        st.session_state.stock_history_plot = history_plot
        
        # 更新进度条
        progress_bar.progress(100)
        status_text.text("✅ 股票预测完成！")
        
    except Exception as e:
        st.error(f"❌ 股票预测过程中发生错误: {str(e)}")
    finally:
        # 重置预测状态
        st.session_state.stock_predicting = False
        
        # 延迟后清除状态文本
        time.sleep(1)
        status_text.empty()


def display_existing_predictions():
    """显示已生成的预测结果"""
    if st.session_state.stock_history_plot is not None:
        with st.container(border=True):
            st.subheader("📈 股票价格预测结果")
            
            # 显示预测图表
            st.plotly_chart(st.session_state.stock_history_plot, use_container_width=True)
            
            # 显示未来预测数据
            st.markdown("### 🔮 未来价格预测")
            
            # 对未来预测数据进行格式化显示
            future_data_display = st.session_state.stock_future_predictions.copy()
            future_data_display['Predicted_Close'] = future_data_display['Predicted_Close'].apply(lambda x: f"${x:.2f}")
            future_data_display.index = future_data_display.index.strftime('%Y-%m-%d')
            
            st.dataframe(future_data_display, use_container_width=True)
            
            # 添加预测提示
            st.info("💡 提示：股票预测仅供参考，不构成投资建议。投资有风险，入市需谨慎。")
        
        # 显示技术指标分析
        with st.container(border=True):
            st.subheader("📊 技术指标分析")
            
            # 使用选项卡显示不同的技术指标
            tab1, tab2, tab3, tab4 = st.tabs(["移动平均线", "RSI指标", "MACD指标", "布林带指标"])
            
            with tab1:
                st.plotly_chart(st.session_state.stock_ma_fig, use_container_width=True)
                st.markdown("""
                **移动平均线说明**:
                - MA5: 5日均线，反映短期趋势
                - MA10: 10日均线，反映短期到中期趋势
                - MA20: 20日均线，反映中期趋势
                - MA50: 50日均线，反映中长期趋势
                - 当短期均线从下向上穿越长期均线，形成"金叉"，可能是买入信号
                - 当短期均线从上向下穿越长期均线，形成"死叉"，可能是卖出信号
                """)
            
            with tab2:
                st.plotly_chart(st.session_state.stock_rsi_fig, use_container_width=True)
                st.markdown("""
                **RSI指标说明**:
                - RSI值范围为0-100
                - RSI > 70: 股票可能处于超买状态，可能回调
                - RSI < 30: 股票可能处于超卖状态，可能反弹
                - RSI在50左右: 市场处于平衡状态
                """)
            
            with tab3:
                st.plotly_chart(st.session_state.stock_macd_fig, use_container_width=True)
                st.markdown("""
                **MACD指标说明**:
                - MACD线: 快线，由12日EMA减去26日EMA计算得到
                - 信号线: 慢线，由MACD线的9日EMA计算得到
                - MACD柱状图: MACD线减去信号线的差值
                - 当MACD线从下向上穿越信号线，形成"金叉"，可能是买入信号
                - 当MACD线从上向下穿越信号线，形成"死叉"，可能是卖出信号
                - 柱状图的变化反映了多空力量的变化
                """)
            
            with tab4:
                st.plotly_chart(st.session_state.stock_bb_fig, use_container_width=True)
                st.markdown("""
                **布林带指标说明**:
                - 中轨: 20日移动平均线
                - 上轨: 中轨 + 2倍标准差
                - 下轨: 中轨 - 2倍标准差
                - 当价格接近上轨，可能超买；接近下轨，可能超卖
                - 布林带开口扩大，表明市场波动加剧；收缩表明市场波动减小
                """)
        
        # 显示模型评估
        with st.container(border=True):
            st.subheader("📉 模型评估")
            
            # 计算评估指标
            actual_prices = st.session_state.stock_actual_prices.flatten()
            predicted_prices = st.session_state.stock_predicted_prices.flatten()
            
            # 计算MSE
            mse = np.mean((actual_prices - predicted_prices) ** 2)
            
            # 计算RMSE
            rmse = np.sqrt(mse)
            
            # 计算MAE
            mae = np.mean(np.abs(actual_prices - predicted_prices))
            
            # 计算MAPE
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            
            # 显示评估指标
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("均方误差 (MSE)", f"{mse:.4f}")
            col2.metric("均方根误差 (RMSE)", f"{rmse:.4f}")
            col3.metric("平均绝对误差 (MAE)", f"{mae:.4f}")
            col4.metric("平均绝对百分比误差 (MAPE)", f"{mape:.2f}%")
            
            st.markdown("""
            **模型评估说明**:
            - **MSE**: 均方误差，衡量预测值与实际值差异的平方的平均值
            - **RMSE**: 均方根误差，MSE的平方根，与原始数据单位相同
            - **MAE**: 平均绝对误差，衡量预测值与实际值差异的绝对值的平均值
            - **MAPE**: 平均绝对百分比误差，以百分比形式表示预测误差
            """)


def render_footer(model):
    """渲染页脚"""
    st.divider()
    
    with st.container():
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.85rem;'>
            <p>📊 股票预测功能基于LSTM深度学习模型</p>
            <p>⚠️ 免责声明：本工具提供的预测仅供参考，不构成任何投资建议</p>
            <p>📈 投资有风险，入市需谨慎</p>
        </div>
        """, unsafe_allow_html=True)

# 创建页面显示函数，遵循项目命名规范
def stock_prediction_agent_show_page():
    stock_prediction_show_page()