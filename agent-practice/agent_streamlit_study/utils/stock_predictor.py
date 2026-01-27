import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from io import StringIO

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class StockPredictor:
    """股票预测器类，使用LSTM模型进行股票价格预测"""
    
    def __init__(self):
        """初始化股票预测器"""
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60  # 使用60天的历史数据来预测
        self.history_data = None
        self.train_data = None
        self.test_data = None
        self.ticker = None
    
    def get_stock_data(self, ticker: str, period: str = '5y') -> pd.DataFrame:
        """
        获取股票历史数据
        
        Args:
            ticker: 股票代码
            period: 数据周期，默认为5年
        
        Returns:
            pd.DataFrame: 包含股票数据的DataFrame
        """
        try:
            self.ticker = ticker
            # 显式设置auto_adjust参数来避免警告，并添加重试机制处理速率限制
            max_retries = 3
            retry_delay = 2  # 秒
            
            for attempt in range(max_retries):
                try:
                    # 显式设置auto_adjust=True以匹配新版本的默认行为
                    stock_data = yf.download(ticker, period=period, auto_adjust=True)
                    
                    # 检查数据是否成功获取
                    if not stock_data.empty:
                        self.history_data = stock_data
                        return stock_data
                    
                except Exception as inner_e:
                    # 捕获速率限制错误
                    if 'Too Many Requests' in str(inner_e) and attempt < max_retries - 1:
                        st.info(f"速率限制，{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    # 捕获其他错误
                    elif attempt == max_retries - 1:
                        raise inner_e
            
            # 如果所有尝试都失败，尝试使用示例数据
            st.warning(f"无法获取{self.ticker}的实时数据，将使用示例数据进行演示")
            return self._get_sample_stock_data()
            
        except Exception as e:
            st.error(f"获取股票数据时出错: {str(e)}")
            # 提供示例数据作为备用
            return self._get_sample_stock_data()
            
    def _get_sample_stock_data(self) -> pd.DataFrame:
        """\提供示例股票数据作为备用"""
        # 创建示例数据
        dates = pd.date_range(end=datetime.now(), periods=120, freq='B')
        np.random.seed(42)  # 确保结果可复现
        closing_prices = 150 + np.cumsum(np.random.randn(len(dates)) * 2)
        
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': closing_prices * (1 + np.random.randn(len(dates)) * 0.01),
            'High': closing_prices * (1 + np.random.randn(len(dates)) * 0.02),
            'Low': closing_prices * (1 - np.random.randn(len(dates)) * 0.02),
            'Close': closing_prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        })
        sample_data.set_index('Date', inplace=True)
        return sample_data
    
    def preprocess_data(self, data: pd.DataFrame, test_size: float = 0.2) -> tuple:
        """
        预处理数据以用于LSTM模型
        
        Args:
            data: 股票数据
            test_size: 测试集比例
        
        Returns:
            tuple: 包含训练数据和测试数据的元组
        """
        # 使用收盘价进行预测
        close_prices = data['Close'].values.reshape(-1, 1)
        
        # 数据归一化
        scaled_data = self.scaler.fit_transform(close_prices)
        
        # 分割训练集和测试集
        train_size = int(len(scaled_data) * (1 - test_size))
        self.train_data = scaled_data[0:train_size, :]
        self.test_data = scaled_data[train_size - self.sequence_length:, :]
        
        # 创建训练集
        X_train, y_train = [], []
        for i in range(self.sequence_length, len(self.train_data)):
            X_train.append(self.train_data[i-self.sequence_length:i, 0])
            y_train.append(self.train_data[i, 0])
        
        # 重塑数据以适应LSTM输入
        X_train = np.array(X_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = np.array(y_train)
        
        # 创建测试集
        X_test, y_test = [], []
        for i in range(self.sequence_length, len(self.test_data)):
            X_test.append(self.test_data[i-self.sequence_length:i, 0])
            y_test.append(self.test_data[i, 0])
        
        # 重塑数据以适应LSTM输入
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_test = np.array(y_test)
        
        return X_train, y_train, X_test, y_test
    
    def build_lstm_model(self, input_shape: tuple) -> Sequential:
        """
        构建LSTM模型
        
        Args:
            input_shape: 输入数据的形状
        
        Returns:
            Sequential: LSTM模型
        """
        model = Sequential()
        
        # 第一层LSTM，返回序列以便于堆叠
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))  # 添加Dropout层防止过拟合
        
        # 第二层LSTM
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        
        # 第三层LSTM
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        
        # 输出层
        model.add(Dense(units=1))
        
        # 编译模型
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.model = model
        return model
    
    def train_model(self, X_train: np.array, y_train: np.array, epochs: int = 50, batch_size: int = 32) -> dict:
        """
        训练LSTM模型
        
        Args:
            X_train: 训练输入数据
            y_train: 训练目标数据
            epochs: 训练轮数
            batch_size: 批次大小
        
        Returns:
            dict: 训练历史
        """
        if self.model is None:
            self.build_lstm_model((X_train.shape[1], 1))
        
        # 设置早停机制防止过拟合
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history.history
    
    def predict(self, X_test: np.array) -> np.array:
        """
        使用训练好的模型进行预测
        
        Args:
            X_test: 测试数据
        
        Returns:
            np.array: 预测结果
        """
        if self.model is None:
            raise Exception("模型未训练，请先训练模型")
        
        # 预测
        predictions = self.model.predict(X_test)
        
        # 反归一化预测结果
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def predict_future(self, days: int = 30) -> pd.DataFrame:
        """
        预测未来几天的股票价格
        
        Args:
            days: 预测未来天数
        
        Returns:
            pd.DataFrame: 包含预测结果的DataFrame
        """
        if self.model is None or self.history_data is None:
            raise Exception("模型未训练或无历史数据，请先训练模型")
        
        # 获取最后sequence_length天的收盘价
        last_sequence = self.history_data['Close'].values[-self.sequence_length:].reshape(-1, 1)
        
        # 归一化数据
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        # 预测未来价格
        future_predictions = []
        current_sequence = last_sequence_scaled.reshape(1, self.sequence_length, 1)
        
        for _ in range(days):
            # 预测下一天
            next_price_scaled = self.model.predict(current_sequence)
            
            # 将预测结果添加到future_predictions
            future_predictions.append(next_price_scaled[0, 0])
            
            # 更新当前序列，移除最旧的数据，添加新的预测结果
            current_sequence = np.append(current_sequence[:, 1:, :], [[next_price_scaled[0]]], axis=1)
        
        # 反归一化预测结果
        future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # 创建预测日期
        last_date = self.history_data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        # 创建结果DataFrame
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_predictions.flatten()
        })
        future_df.set_index('Date', inplace=True)
        
        return future_df
    
    def create_plot(self, actual_prices: np.array, predicted_prices: np.array, future_prices: pd.DataFrame = None) -> go.Figure:
        """
        创建股票价格预测图表
        
        Args:
            actual_prices: 实际价格
            predicted_prices: 预测价格
            future_prices: 未来预测价格
        
        Returns:
            go.Figure: Plotly图表对象
        """
        # 创建图表
        fig = go.Figure()
        
        # 添加实际价格曲线
        dates = self.history_data.index[-len(actual_prices):]
        fig.add_trace(go.Scatter(
            x=dates, 
            y=actual_prices, 
            mode='lines', 
            name='实际价格',
            line=dict(color='blue')
        ))
        
        # 添加预测价格曲线
        fig.add_trace(go.Scatter(
            x=dates, 
            y=predicted_prices, 
            mode='lines', 
            name='预测价格',
            line=dict(color='red', dash='dash')
        ))
        
        # 如果有未来预测价格，添加未来预测曲线
        if future_prices is not None:
            fig.add_trace(go.Scatter(
                x=future_prices.index, 
                y=future_prices['Predicted_Close'], 
                mode='lines', 
                name='未来预测',
                line=dict(color='green', dash='dot')
            ))
        
        # 设置图表标题和标签
        fig.update_layout(
            title=f'{self.ticker} 股票价格预测',
            xaxis_title='日期',
            yaxis_title='价格',
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算常用技术指标
        
        Args:
            data: 股票数据
        
        Returns:
            pd.DataFrame: 包含技术指标的数据
        """
        df = data.copy()
        
        # 计算移动平均线
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # 计算相对强弱指标(RSI)
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # 计算布林带
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
        
        return df
    
    def create_technical_indicators_plots(self, data: pd.DataFrame):
        """
        创建技术指标图表
        
        Args:
            data: 包含技术指标的股票数据
        """
        # 选择最近90天的数据进行展示
        recent_data = data.iloc[-90:].copy()
        
        # 创建移动平均线图表
        ma_fig = go.Figure()
        ma_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'], mode='lines', name='收盘价', line=dict(color='black')))
        ma_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MA5'], mode='lines', name='MA5', line=dict(color='blue', width=1)))
        ma_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MA10'], mode='lines', name='MA10', line=dict(color='cyan', width=1)))
        ma_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MA20'], mode='lines', name='MA20', line=dict(color='green', width=1)))
        ma_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MA50'], mode='lines', name='MA50', line=dict(color='yellow', width=1)))
        ma_fig.update_layout(title='移动平均线', xaxis_title='日期', yaxis_title='价格', template='plotly_white')
        
        # 创建RSI图表
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
        rsi_fig.add_hline(y=70, line=dict(color='red', dash='dash'), name='超买线')
        rsi_fig.add_hline(y=30, line=dict(color='green', dash='dash'), name='超卖线')
        rsi_fig.update_layout(title='相对强弱指标(RSI)', xaxis_title='日期', yaxis_title='RSI值', yaxis_range=[0, 100], template='plotly_white')
        
        # 创建MACD图表
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
        macd_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Signal_Line'], mode='lines', name='信号线', line=dict(color='orange')))
        macd_fig.add_bar(x=recent_data.index, y=recent_data['MACD_Histogram'], name='MACD柱状图', marker_color='gray', opacity=0.5)
        macd_fig.update_layout(title='MACD指标', xaxis_title='日期', yaxis_title='值', template='plotly_white')
        
        # 创建布林带图表
        bb_fig = go.Figure()
        bb_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'], mode='lines', name='收盘价', line=dict(color='black')))
        bb_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['BB_Upper'], mode='lines', name='上轨', line=dict(color='red', dash='dash')))
        bb_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['BB_Middle'], mode='lines', name='中轨', line=dict(color='blue', dash='dash')))
        bb_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['BB_Lower'], mode='lines', name='下轨', line=dict(color='green', dash='dash')))
        bb_fig.update_layout(title='布林带', xaxis_title='日期', yaxis_title='价格', template='plotly_white')
        
        return ma_fig, rsi_fig, macd_fig, bb_fig

# 创建单例实例
stock_predictor = StockPredictor()