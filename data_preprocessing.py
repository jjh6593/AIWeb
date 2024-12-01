import numpy as np
import pandas as pd
import torch

# 사용자 정의 Min-Max 스케일링 클래스
class MinMaxScaling:
    def __init__(self, df, target_column):
        self.target_column = target_column
        self.data = df.copy()
        self.scaled_data = pd.DataFrame()
        self.max_vals = {}
        self.min_vals = {}
        self.ranges = {}

        for col in df.columns:
            max_val = df[col].max()
            min_val = df[col].min()
            self.max_vals[col] = max_val
            self.min_vals[col] = min_val
            self.ranges[col] = max_val - min_val
            # 만약 특징값이 모두 동일하면 스케일링하지 않고 0으로 설정
            if self.ranges[col] == 0:
                self.scaled_data[col] = 0
            else:
                self.scaled_data[col] = (df[col] - min_val) / self.ranges[col]

    def get_scaled_data(self):
        X = self.scaled_data.drop(columns=[self.target_column])
        y = self.scaled_data[self.target_column]
        return X, y

    def denormalize(self, data, columns):
        denorm_data = []
        for i, col in enumerate(columns):
            element = data[i] * self.ranges[col] + self.min_vals[col]
            denorm_data.append(element)
        return denorm_data
