import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import streamlit as st
import joblib

# 加载数据时指定编码为 utf-8
data = pd.read_csv('训练集.csv', encoding='utf-8')

# 分离特征和目标变量
X = data[['Race', 'WHO_classification', 'Masaoka_Koga_Stage']]
y = data['Lung_metastasis']

# 使用SMOTE处理数据不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 数据缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# 训练GBM模型
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(X_scaled, y_resampled)

# 保存模型和缩放器
joblib.dump(gbm_model, 'gbm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 特征映射
Race_mapper = {0: 'White', 1: 'Black', 2: 'Other'}
WHO_mapper = {0: 'A', 1: 'AB', 2: 'B1', 3: 'B2', 4: 'B3', 5: 'C'}
Stage_mapper = {0: 'I/IIA', 1: 'IIB', 2: 'III/IV'}
Metastasis_mapper = {0: 'No Lung Metastasis', 1: 'Lung Metastasis'}

# 预测函数
def predict_lung_metastasis(Race, WHO_classification, Masaoka_Koga_Stage):
    model = joblib.load('gbm_model.pkl')
    scaler = joblib.load('scaler.pkl')

    input_data = pd.DataFrame({
        'Race': [Race],
        'WHO_classification': [WHO_classification],
        'Masaoka_Koga_Stage': [Masaoka_Koga_Stage]
    })

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]
    
    return Metastasis_mapper[prediction], probability

# Streamlit Web应用程序
st.title('Lung Metastasis Prediction for Thymoma Patients')
st.write('This app predicts the likelihood of lung metastasis in thymoma patients based on Race, WHO classification, and Masaoka-Koga stage.')

# 侧边栏输入
st.sidebar.header('Input Features')
Race = st.sidebar.selectbox('Race', options=list(Race_mapper.keys()), format_func=lambda x: Race_mapper[x])
WHO_classification = st.sidebar.selectbox('WHO Classification', options=list(WHO_mapper.keys()), format_func=lambda x: WHO_mapper[x])
Masaoka_Koga_Stage = st.sidebar.selectbox('Masaoka-Koga Stage', options=list(Stage_mapper.keys()), format_func=lambda x: Stage_mapper[x])

# 预测
if st.sidebar.button('Predict'):
    prediction, probability = predict_lung_metastasis(Race, WHO_classification, Masaoka_Koga_Stage)
    st.write(f'Prediction: {prediction}')
    st.write(f'Probability of Lung Metastasis: {probability:.2f}')
