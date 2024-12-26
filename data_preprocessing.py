import numpy as np
import pandas as pd
import torch

import numpy as np

class MinMaxScalerWithFeatureUnits:
    def __init__(self, feature_range=(0, 1), feature_units=None):
        """
        사용자 정의 MinMax 스케일러
        :param feature_range: 정규화 범위 (기본값 (0, 1))
        :param feature_units: 각 feature에 적용할 단위 (기본값 None)
        """
        self.feature_range = feature_range
        self.feature_units = feature_units or {}
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        """
        데이터를 기반으로 스케일러를 설정
        :param X: 입력 데이터 (numpy array)
        :return: self
        """
        X = np.asarray(X)
        n_features = X.shape[1]
        self.min_ = np.zeros(n_features)
        self.scale_ = np.zeros(n_features)

        for i in range(n_features):
            unit = self.feature_units.get(i, 1)  # 기본 unit은 1
            feature_min = X[:, i].min()
            adjusted_min = feature_min - (feature_min % unit)  # 단위에 맞게 최소값 조정
            self.min_[i] = adjusted_min
            self.scale_[i] = X[:, i].max() - adjusted_min

        return self

    def transform(self, X):
        """
        데이터를 정규화
        :param X: 입력 데이터 (numpy array)
        :return: 정규화된 데이터
        """
        X = np.asarray(X)
        return (X - self.min_) / self.scale_ * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X):
        """
        fit과 transform을 동시에 수행
        :param X: 입력 데이터 (numpy array)
        :return: 정규화된 데이터
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        정규화된 데이터를 원본 값으로 복원
        :param X: 정규화된 데이터 (numpy array)
        :return: 원본 데이터
        """
        X = np.asarray(X)
        return (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]) * self.scale_ + self.min_


# 예제 사용
data_points = [[100, 200], [20, 300], [50, 400]]
feature_units = {0: 5, 1: 10}  # 첫 번째 feature는 5 단위, 두 번째 feature는 10 단위
scaler = MinMaxScalerWithFeatureUnits(feature_units=feature_units)
scaled_data = scaler.fit_transform(data_points)

print(scaled_data)