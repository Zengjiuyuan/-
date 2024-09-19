import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import streamlit as st
import joblib

# 尝试不同编码读取CSV文件的函数
def read_csv_with_fallback(file_path):
    encodings = ['utf-8', 'gbk', 'latin1']  # 常见编码格式
    for encoding in encodings:
        try:
            # 尝试使用当前编码读取文件
            data = pd.read_csv(file_path, encoding=encoding)
            st.write(f"Successfully read the CSV file using encoding: {encoding}")
            return data
        except UnicodeDecodeError:
            # 如果出现解码错误，继续尝试下一个编码
            st.write(f"Failed to read with encoding: {encoding}, trying next...")
            continue
        except Exception as e:
            # 捕获并显示其他类型的错误
            st.write(f"An error occurred: {e}")
            return None
    st.write("All encoding attempts failed.")
    return None

# 读取训练集数据
train_data = read_csv_with_fallback('训练集.csv')

# 如果读取失败，停止后续处理
if train_data is None:
    st.write("Failed to load training data.")
else:
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
