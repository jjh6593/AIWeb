import numpy as np
import pandas as pd
import torch

# 사용자 정의 Min-Max 스케일링 클래스
class MinMaxScalerWithFeatureUnits:
    def __init__(self, feature_range=(0, 1), feature_units=None):
        self.feature_range = feature_range
        self.feature_units = feature_units or {}
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X)
        n_features = X.shape[1]
        self.min_ = np.zeros(n_features)
        self.scale_ = np.zeros(n_features)

        for i in range(n_features):
            unit = self.feature_units.get(i, 1)  # 기본 unit은 1
            feature_min = X[:, i].min()
            adjusted_min = feature_min - (feature_min % unit)
            self.min_[i] = adjusted_min
            self.scale_[i] = X[:, i].max() - adjusted_min

        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.min_) / self.scale_ * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# 예제 사용
data_points = [[100, 200], [20, 300], [50, 400]]
feature_units = {0: 5, 1: 10}  # 첫 번째 feature는 5 단위, 두 번째 feature는 10 단위
scaler = MinMaxScalerWithFeatureUnits(feature_units=feature_units)
scaled_data = scaler.fit_transform(data_points)

print(scaled_data)