import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 初始化一个简单的数据集
def create_dataset():
    np.random.seed(42)
    data = pd.DataFrame({
        'Feature1': np.random.normal(0, 1, 100),
        'Feature2': np.random.normal(5, 2, 100),
        'Target': np.random.normal(10, 5, 100)
    })
    return data

# 简单的随机森林模型训练和预测
def train_and_predict(feature1, feature2):
    data = create_dataset()
    model = RandomForestRegressor()
    model.fit(data[['Feature1', 'Feature2']], data['Target'])
    prediction = model.predict([[feature1, feature2]])
    return prediction[0]

# 页面布局和导航
st.sidebar.title('导航')
page = st.sidebar.radio("选择页面", ["EDA", "模型训练"])

# EDA 页面
if page == "EDA":
    st.title("探索性数据分析 (EDA)")
    data = create_dataset()
    st.write("数据预览：")
    st.write(data.head())
    st.write("描述性统计：")
    st.write(data.describe())
    
    # 数据可视化
    st.write("特征分布可视化：")
    fig, ax = plt.subplots()
    sns.histplot(data, x='Feature1', kde=True, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(data, x='Feature2', kde=True, ax=ax)
    st.pyplot(fig)

# 模型训练页面
elif page == "模型训练":
    st.title("模型训练和预测")
    
    # 用户输入特征值
    st.write("请输入特征值以进行预测：")
    feature1 = st.number_input("特征 1 (Feature1)", value=0.0, format="%.2f")
    feature2 = st.number_input("特征 2 (Feature2)", value=0.0, format="%.2f")
    
    # 计算按钮
    if st.button("计算预测结果"):
        result = train_and_predict(feature1, feature2)
        st.success(f"预测的目标值是：{result:.2f}")
