import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# 字符串
st.write("这是一段文本。")
 
# 数字
st.write(42)
 
# 列表
st.write([1, 2, 3])
 
# 字典
st.write({"key": "value"})
 
# 数据框（DataFrame）
df = pd.DataFrame({"Column 1": [1, 2, 3], "Column 2": ["A", "B", "C"]})
st.write(df)
 
#多参数用法
st.write("这是一个字符串", 42, [1, 2, 3], {"key": "value"})
 
#自定义渲染
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
st.write(fig)