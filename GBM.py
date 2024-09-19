import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# 读取训练集数据 (由于你说是复制粘贴的数据格式，所以这里我们从GitHub仓库中加载训练集)
@st.cache
def load_data():
    data = pd.DataFrame({
        'Race': [0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1],
        'WHO_classification': [5, 2, 3, 3, 4, 4, 2, 4, 5, 5, 5, 3],
        'Masaoka_Koga_Stage': [2, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 2],
        'Lung_metastasis': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })
    return data

# 加载数据
train_data = load_data()

# 分离输入特征和目标变量
X = train_data[['Race', 'WHO_classification', 'Masaoka_Koga_Stage']]
y = train_data['Lung_metastasis']

# 使用SMOTE技术处理数据不平衡
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# 特征缩放
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# 创建并训练GBM模型
gbm_model = GradientBoostingClassifier(random_state=42)
gbm_model.fit(X_resampled_scaled, y_resampled)

# 将模型保存到文件，以便在预测时使用
joblib.dump(gbm_model, 'gbm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 创建一个预测函数
def predict_lung_metastasis(Race, WHO_classification, Masaoka_Koga_Stage):
    # 加载模型和标准化器
    gbm_model = joblib.load('gbm_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # 创建输入数据
    input_data = pd.DataFrame({
        'Race': [Race],
        'WHO_classification': [WHO_classification],
        'Masaoka_Koga_Stage': [Masaoka_Koga_Stage]
    })

    # 特征缩放
    input_data_scaled = scaler.transform(input_data)

    # 预测肺转移的概率
    prediction = gbm_model.predict(input_data_scaled)
    probability = gbm_model.predict_proba(input_data_scaled)[0][1]  # 获取属于类别1的概率
    return prediction[0], probability

# 构建Streamlit Web应用程序
st.title("GBM Model Predicting Lung Metastasis of Thymoma")

st.sidebar.write("输入特征进行预测")

# 用户输入特征
Race = st.sidebar.selectbox("Race", options={0: "White", 1: "Black", 2: "Other"})
WHO_classification = st.sidebar.selectbox("WHO Classification", options={0: "A", 1: "AB", 2: "B1", 3: "B2", 4: "B3", 5: "C"})
Masaoka_Koga_Stage = st.sidebar.selectbox("Masaoka Koga Stage", options={0: "I/IIA", 1: "IIB", 2: "III/IV"})

# 预测按钮
if st.button("Predict"):
    prediction, probability = predict_lung_metastasis(Race, WHO_classification, Masaoka_Koga_Stage)
    class_label = "Yes" if prediction == 1 else "No"
    st.write(f"Predicted Lung Metastasis: {class_label}")
    st.write(f"Probability: {probability:.3f}")
