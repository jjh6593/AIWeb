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
        'pytorch': ['MLP_1', 'MLP_2', 'MLP_3', 'MLP_dynamic'],
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

        # ────────────── 핵심 부분 시작 ──────────────
        # 만약 'hidden_size_1', 'hidden_size_2', ... 등을 동적으로 처리하고 싶다면:
        hidden_layers = []

        # 최대 3개(또는 필요하다면 4개, 5개...) 까지 루프
        for i in range(1, 4):
            key = f"hidden_size_{i}"
            if key in hyperparams:
                hidden_layers.append(int(hyperparams[key]))

        # hidden_size_가 하나도 안 넘어온 경우, 기본값
        if not hidden_layers:
            hidden_layers = [32]  # ex) 기본값 1개 레이어, 32차원

        # ────────────── 핵심 부분 끝 ──────────────

        # 이제 model_selected가 'MLP_1', 'MLP_2', 'MLP_3' 인 경우와,
        # 그냥 'MLP_dynamic' 등인 경우에 따라 분기
        # (기존 MLP_1, MLP_2, MLP_3는 n_layers를 고정으로 넣어둔 예시 같으니,
        #  실제로는 모두 MLPDynamic로 사용해도 되지만, 예시로 분기 처리)
        if model_selected == 'MLP_1':
            # hidden_layers가 여러개라도, 강제로 1개만 쓰고 싶다면:
            return MLPDynamic(input_size, hidden_sizes=hidden_layers[:1], output_size=1)

        elif model_selected == 'MLP_2':
            return MLPDynamic(input_size, hidden_sizes=hidden_layers[:2], output_size=1)

        elif model_selected == 'MLP_3':
            return MLPDynamic(input_size, hidden_sizes=hidden_layers[:3], output_size=1)

        elif model_selected == 'MLP_dynamic':
            # hidden_size가 몇 개든 전부 그대로 사용
            return MLPDynamic(input_size, hidden_sizes=hidden_layers, output_size=1)

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
# 가변 레이어를 가진 MLP 모델
class MLPDynamic(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(MLPDynamic, self).__init__()
        print(f"[DEBUG] Initializing MLPDynamic: input_size={input_size}, hidden_sizes={hidden_sizes}")

        layers = []
        in_dim = input_size

        # hidden_sizes = [64, 32, 16] 등의 식
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # 최종 출력은 항상 1
        layers.append(nn.Linear(in_dim, output_size))

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