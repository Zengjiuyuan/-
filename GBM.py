import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# 加载数据
@st.cache
def load_data():
    # 读取训练集数据
    data = pd.read_csv('训练集.csv')
    return data

# 数据预处理
@st.cache
def preprocess_data(data):
    # 分离输入特征和目标变量
    X = data[['Race', 'WHO_classification', 'Masaoka_Koga_Stage']]
    y = data['Lung_metastasis']
    
    # 使用SMOTE处理数据不平衡
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # 特征缩放
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    return X_resampled_scaled, y_resampled, scaler

# 加载和训练模型
@st.cache(allow_output_mutation=True)
def train_model():
    data = load_data()
    X_resampled_scaled, y_resampled, scaler = preprocess_data(data)
    
    # 创建并训练GBM模型
    gbm_model = GradientBoostingClassifier(random_state=42)
    gbm_model.fit(X_resampled_scaled, y_resampled)
    
    # 保存模型和标准化器
    joblib.dump(gbm_model, 'gbm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return gbm_model, scaler

# 预测函数
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

# 训练模型（首次运行时会执行）
train_model()
