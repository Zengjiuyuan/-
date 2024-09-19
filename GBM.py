import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import streamlit as st
import joblib

# 读取训练集数据
train_data = pd.read_csv('训练集.csv')

# 分离输入特征和目标变量
X = train_data[['Race', 'WHO_classification', 'Masaoka_Koga_Stage']]
y = train_data['Lung_metastasis']

# 使用SMOTE处理数据不平衡问题
sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_resample(X, y)

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_smote)

# 创建并训练GBM模型
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(X_scaled, y_smote)

# 保存模型和缩放器
joblib.dump(gbm_model, 'gbm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 加载模型和缩放器
gbm_model = joblib.load('gbm_model.pkl')
scaler = joblib.load('scaler.pkl')

# 特征映射
feature_order = ['Race', 'WHO_classification', 'Masaoka_Koga_Stage']
class_mapping = {0: "No lung metastasis", 1: "Lung metastasis"}
Race_mapper = {"White": 0, "Black": 1, "Other": 2}
WHO_classification_mapper = {"A": 0, "AB": 1, "B1": 2, "B2": 3, "B3": 4, "C": 5}
Masaoka_Koga_Stage_mapper = {"I/IIA": 0, "IIB": 1, "III/IV": 2}

# 预测函数
def predict_lung_metastasis(Race, WHO_classification, Masaoka_Koga_Stage):
    input_data = pd.DataFrame({
        'Race': [Race_mapper[Race]],
        'WHO_classification': [WHO_classification_mapper[WHO_classification]],
        'Masaoka_Koga_Stage': [Masaoka_Koga_Stage_mapper[Masaoka_Koga_Stage]],
    }, columns=feature_order)

    input_scaled = scaler.transform(input_data)
    prediction = gbm_model.predict(input_scaled)[0]
    probability = gbm_model.predict_proba(input_scaled)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability

# 创建Web应用程序
st.title("GBM Model Predicting Lung Metastasis of Thymoma")
st.sidebar.write("Patient Information")

Race = st.sidebar.selectbox("Race", options=list(Race_mapper.keys()))
WHO_classification = st.sidebar.selectbox("WHO Classification", options=list(WHO_classification_mapper.keys()))
Masaoka_Koga_Stage = st.sidebar.selectbox("Masaoka-Koga Stage", options=list(Masaoka_Koga_Stage_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_lung_metastasis(Race, WHO_classification, Masaoka_Koga_Stage)

    st.write("Class Label: ", prediction)
    st.write("Probability of lung metastasis: {:.3f}".format(probability))
