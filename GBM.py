import streamlit as st
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 假设有函数 load_data 来加载数据
def load_data():
    # 替换为你自己的数据加载逻辑
    data = {
        'Race': [1, 2, 1, 2],
        'WHO_classification': [1, 2, 1, 2],
        'Masaoka_Koga_Stage': [1, 2, 3, 2],
        'target': [0, 1, 0, 1]
    }
    return data

# 数据预处理函数
def preprocess_data(data):
    X = [[data['Race'][i], data['WHO_classification'][i], data['Masaoka_Koga_Stage'][i]] for i in range(len(data['Race']))]
    y = data['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# 模型训练并保存模型
def train_model():
    data = load_data()
    X_scaled, y, scaler = preprocess_data(data)
    
    gbm_model = GradientBoostingClassifier(random_state=42)
    gbm_model.fit(X_scaled, y)

    # 保存模型和标准化器
    joblib.dump(gbm_model, 'gbm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return gbm_model, scaler

# 从文件加载模型
def load_model():
    if os.path.exists('gbm_model.pkl') and os.path.exists('scaler.pkl'):
        gbm_model = joblib.load('gbm_model.pkl')
        scaler = joblib.load('scaler.pkl')
    else:
        gbm_model, scaler = train_model()
    
    return gbm_model, scaler

# 使用 Streamlit 缓存机制
@st.cache_resource
def get_model():
    return load_model()

# 预测函数
def predict_lung_metastasis(Race, WHO_classification, Masaoka_Koga_Stage):
    gbm_model, scaler = get_model()
    
    # 将输入数据标准化
    input_data = [[Race, WHO_classification, Masaoka_Koga_Stage]]
    input_data_scaled = scaler.transform(input_data)
    
    prediction = gbm_model.predict(input_data_scaled)
    probability = gbm_model.predict_proba(input_data_scaled)
    
    return prediction[0], probability[0]

# Streamlit 应用界面
st.title('Lung Metastasis Prediction')

# 输入框
Race = st.number_input('Enter Race:', min_value=1, max_value=2, step=1)
WHO_classification = st.number_input('Enter WHO Classification:', min_value=1, max_value=2, step=1)
Masaoka_Koga_Stage = st.number_input('Enter Masaoka-Koga Stage:', min_value=1, max_value=3, step=1)

# 预测按钮
if st.button('Predict'):
    prediction, probability = predict_lung_metastasis(Race, WHO_classification, Masaoka_Koga_Stage)
    
    st.write(f'Prediction: {"Metastasis" if prediction == 1 else "No Metastasis"}')
    st.write(f'Probability: {probability}')
