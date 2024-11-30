# models.py

import torch
import torch.nn as nn
import torch.optim as optim
import xgboost
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

# models.py

import torch
import torch.nn as nn
import xgboost
from sklearn.linear_model import (LinearRegression, Ridge, Lasso)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


def create_model(model_selected, input_size=14):
    # 지원되는 모델 목록 정의
    supported_models = {
        'pytorch': ['MLP_1', 'MLP_2', 'MLP_3'],
        'scikit': [
            'LinearRegressor', 'Ridge', 'Lasso', 'ElasticNet',
            'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor',
            'SVR', 'KNeighborsRegressor', 'HuberRegressor', 'GaussianProcessRegressor', 'XGBoost'
        ]
    }

    # 모든 지원 모델 이름 통합
    all_supported_models = supported_models['pytorch'] + supported_models['scikit']

    # 모델 선택 확인
    if model_selected not in all_supported_models:
        raise ValueError(
            f"지원되지 않는 모델입니다: {model_selected}. "
            f"지원되는 모델: {', '.join(all_supported_models)}"
        )

    # PyTorch 모델 생성
    if model_selected in supported_models['pytorch']:
        if model_selected == 'MLP_1':
            return MLP(input_size=input_size)
        elif model_selected == 'MLP_2':
            return MLP(input_size=input_size, n_layers=2)
        elif model_selected == 'MLP_3':
            return MLP(input_size=input_size, n_layers=3)

    # Scikit-learn 모델 생성
    elif model_selected in supported_models['scikit']:
        if model_selected == 'LinearRegressor':
            return LinearRegression()
        elif model_selected == 'Ridge':
            return Ridge(alpha=1.0)
        elif model_selected == 'Lasso':
            return Lasso(alpha=0.1)
        elif model_selected == 'ElasticNet':
            return ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif model_selected == 'DecisionTreeRegressor':
            return DecisionTreeRegressor(max_depth=5)
        elif model_selected == 'RandomForestRegressor':
            return RandomForestRegressor()
        elif model_selected == 'GradientBoostingRegressor':
            return GradientBoostingRegressor()
        elif model_selected == 'SVR':
            return SVR(kernel='rbf')
        elif model_selected == 'KNeighborsRegressor':
            return KNeighborsRegressor(n_neighbors=5)
        elif model_selected == 'HuberRegressor':
            return HuberRegressor()
        elif model_selected == 'GaussianProcessRegressor':
            return GaussianProcessRegressor()
        elif model_selected == 'XGBoost':
            return xgboost.XGBRegressor()

    # 이외의 모델은 처리하지 않음
    raise ValueError(f"지원되지 않는 모델: {model_selected}")


# MLP 클래스 정의
class MLP(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=32, n_layers=1):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)