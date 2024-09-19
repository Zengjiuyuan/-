import os
import subprocess

# 安装 chardet 库
def install_chardet():
    try:
        subprocess.check_call([os.sys.executable, '-m', 'pip', 'install', 'chardet'])
        print("chardet successfully installed.")
    except Exception as e:
        print(f"An error occurred while installing chardet: {e}")

install_chardet()  # 安装 chardet

import pandas as pd
import chardet
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import streamlit as st
import joblib

# 使用chardet库检测文件编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# 读取CSV文件
def read_csv_with_fallback(file_path):
    try:
        # 检测文件的编码
        encoding = detect_encoding(file_path)
        st.write(f"Detected file encoding: {encoding}")

        # 尝试读取文件，忽略格式错误行
        data = pd.read_csv(file_path, encoding=encoding, delimiter='\t', error_bad_lines=False)
        st.write("Successfully read the CSV file.")
        return data
    except Exception as e:
        st.write(f"An error occurred while reading the file: {e}")
        return None

# 读取训练集数据
train_data = read_csv_with_fallback('训练集.csv')

# 如果读取失败，停止后续处理
if train_data is None:
    st.write("Failed to load training data.")
else:
    # 对数据进行标准化
    scaler = StandardScaler()
    X = train_data.drop('Lung_metastasis', axis=1)  # 假设数据集中 'Lung_metastasis' 是目标列
    y = train_data['Lung_metastasis']

    X_scaled = scaler.fit_transform(X)

    # 使用SMOTE进行上采样
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # 创建GBM分类器
    model = GradientBoostingClassifier()

    # 训练模型
    model.fit(X_resampled, y_resampled)

    # 保存模型
    joblib.dump(model, 'trained_model.pkl')

    st.write("Model training complete and saved as 'trained_model.pkl'")
