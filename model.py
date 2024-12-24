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
torch.set_num_threads(1)  # 스레드 수를 1로 제한
def create_model(model_selected, input_size=14, hyperparams=None):
    if hyperparams is None:
        hyperparams = {}

    # 지원되는 모델 목록 정의
    supported_models = {
        'pytorch': ['MLP_1', 'MLP_2', 'MLP_3'],
        'scikit': [
            'LinearRegressor', 'Ridge', 'Lasso', 'ElasticNet',
            'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor',
            'SVR', 'KNeighborsRegressor', 'HuberRegressor', 'GaussianProcessRegressor', 'XGBoost'
        ]
    }

    # 모델 선택 확인
    if model_selected not in (supported_models['pytorch'] + supported_models['scikit']):
        raise ValueError(f"지원되지 않는 모델: {model_selected}")

    # PyTorch 모델 생성
    if model_selected in supported_models['pytorch']:
        print(f"[DEBUG] Creating PyTorch model: {model_selected} with input size: {input_size}")
        if model_selected == 'MLP_1':
            return MLP(input_size=input_size)
        elif model_selected == 'MLP_2':
            return MLP(input_size=input_size, n_layers=2)
        elif model_selected == 'MLP_3':
            return MLP(input_size=input_size, n_layers=3)

    # Scikit-learn 모델 생성
    print(f"[DEBUG] Creating Scikit-learn model: {model_selected}")
    if model_selected == 'LinearRegressor':
        return LinearRegression(**hyperparams)
    elif model_selected == 'Ridge':
        return Ridge(**hyperparams)
    elif model_selected == 'Lasso':
        return Lasso(**hyperparams)
    elif model_selected == 'ElasticNet':
        return ElasticNet(**hyperparams)
    elif model_selected == 'DecisionTreeRegressor':
        return DecisionTreeRegressor(**hyperparams)
    elif model_selected == 'RandomForestRegressor':
        return RandomForestRegressor(**hyperparams)
    elif model_selected == 'GradientBoostingRegressor':
        return GradientBoostingRegressor(**hyperparams)
    elif model_selected == 'SVR':
        return SVR(**hyperparams)
    elif model_selected == 'KNeighborsRegressor':
        return KNeighborsRegressor(**hyperparams)
    elif model_selected == 'HuberRegressor':
        return HuberRegressor(**hyperparams)
    elif model_selected == 'GaussianProcessRegressor':
        return GaussianProcessRegressor(**hyperparams)
    elif model_selected == 'XGBoost':
        return xgboost.XGBRegressor(**hyperparams)

    raise ValueError(f"지원되지 않는 모델: {model_selected}")

class MLP(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=32, n_layers=1):
        super(MLP, self).__init__()
        print(f"[DEBUG] Initializing MLP: input_size={input_size}, hidden_size={hidden_size}, n_layers={n_layers}")
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


# 사용자 제공했던 EarlyStopping 클래스 그대로 사용
class EarlyStopping:
    def __init__(self, patience=10, delta=0.1):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = float('inf')

    def __call__(self, train_loss):
        score = -train_loss
        if self.best_score is None:
            self.best_score = score
            self.train_loss_min = train_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.train_loss_min = train_loss