# data_utils.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def save_data(df, file_path):
    df.to_csv(file_path, index=False)

def preprocess_data(df, target_column, scaler='minmax', missing_handling='none'):
    # 결측치 처리
    if missing_handling == 'drop':
        df = df.dropna()
    elif missing_handling == 'mode':
        df = df.fillna(df.mode().iloc[0])

    # 스케일링
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if scaler == 'minmax':
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
    else:
        X_scaled = X.values

    return X_scaled, y.values

def get_columns(file_path):
    df = pd.read_csv(file_path)
    return df.columns.tolist()

def get_data_preview(df, n=5):
    return df.head(n).to_dict(orient='records')
