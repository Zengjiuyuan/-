好的，以下是一个完整的代码示例，包含了如何处理编码问题并读取 训练集.csv 文件，还包括常用的异常处理机制。如果遇到编码错误，代码会自动尝试不同的编码格式，并最终读取数据：

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import streamlit as st
import joblib
import chardet

# 尝试不同编码读取CSV文件的函数
def read_csv_with_encoding(file_path):
    encodings = ['utf-8', 'gbk', 'latin1']  # 常用编码
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    # 如果所有常用编码都无法读取，尝试使用chardet检测
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        detected_encoding = result['encoding']
    return pd.read_csv(file_path, encoding=detected_encoding), detected_encoding

# 使用read_csv_with_encoding函数读取训练集数据
train_data, encoding_used = read_csv_with_encoding('训练集.csv')

# 输出读取文件时使用的编码
st.write(f"Successfully read the CSV file using encoding: {encoding_used}")

# 对数据进行标准化
scaler = StandardScaler()
X = train_data.drop('label', axis=1)  # 假设数据集中 'label' 是目标列
y = train_data['label']

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
