#!/bin/bash

echo "🚀 启动AI智能助手平台..."
echo "================================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python3"
    exit 1
fi

# 检查依赖包
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "📦 正在安装依赖包..."
    pip3 install -r requirements.txt
fi

# 启动应用
echo "🌐 启动Streamlit应用..."
echo "访问地址: http://localhost:8501"
echo "================================"

streamlit run main.py --server.port 8501 --server.address 0.0.0.0
