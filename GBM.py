import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import streamlit as st
import joblib

# 尝试读取以制表符分隔的文件
def read_csv_with_fallback(file_path):
    try:
        # 指定 delimiter='\t' 表示制表符分隔的文件
        data = pd.read_csv(file_path, delimiter='\t', encoding='utf-8')
        st.write("Successfully read the CSV file with tab delimiter.")
        return data
    except pd.errors.ParserError as e:
        st.write(f"Parser error: {e}")
        return None
    except Exception as e:
        st.write(f"An error occurred: {e}")
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
