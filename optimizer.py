import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from itertools import product, chain
from data_preprocessing import MinMaxScaling
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from model import MLP
import xgboost

# Set random seed for reproducibility
seed = 2024
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parameters
dtype = torch.float32
regression_epochs = 500
regression_eta = 0.01
regression_show_epochs = True 
regression_batch = 256
regression_hidden_size = 32
neural_list = ['MLP', 'Conv']

# class MinMaxScaling:
#     def __init__(self, data):
#         self.max = data.max().values
#         self.min = data.min().values
#         self.range = self.max - self.min
#         self.data = ((data - self.min) / self.range).values
#         self.data = torch.tensor(self.data, dtype=dtype)
#         self.data = self.data.reshape(-1, data.shape[1])

#     def denormalize(self, data):
#         data = data.detach().numpy() if isinstance(data, torch.Tensor) else data
#         return data * self.range + self.min

# class MLP(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size=regression_hidden_size, n_layers=1):
#         super(MLP, self).__init__()
#         layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
#         for _ in range(n_layers - 1):
#             layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
#         layers.append(nn.Linear(hidden_size, output_size))
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)

# def modeller(model_name, input_size, output_size):
#     if model_name.startswith('MLP'):
#         if 'n_layers' in model_name:
#             n_layers = int(model_name.split('=')[-1].strip(')'))
#         else:
#             n_layers = 1
#         return MLP(input_size, output_size, n_layers=n_layers)
#     else:
#         return eval(f"{model_name}")

# def train_models(models, model_list, feature, target):
#     trained_models = []
#     all_training_losses = []
#     for i, model in enumerate(models):
#         if any(m in model_list[i] for m in neural_list):
#             training_losses = train_nn(model, feature.data, target.data)
#             all_training_losses.append(training_losses)
#         else:
#             train_ml(model, feature.data, target.data)
#             all_training_losses.append(None)
#         trained_models.append(model)
#         print(f"The regressor {model_list[i]} has been trained.")
#     return trained_models, all_training_losses

# def train_nn(model, feature, target, epochs=regression_epochs):
#     optimizer = optim.Adam(model.parameters(), lr=regression_eta)
#     criterion = nn.MSELoss()
#     training_losses = []
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         outputs = model(feature)
#         loss = criterion(outputs, target)
#         loss.backward()
#         optimizer.step()
#         training_losses.append(loss.item())
#         if regression_show_epochs and epoch % 10 == 0:
#             print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")
#     return training_losses

# def train_ml(model, feature, target):
#     feature_np = feature.detach().numpy()
#     target_np = target.detach().numpy().ravel()
#     model.fit(feature_np, target_np)

class UNIVERSE:
    def __init__(self, unit, upper_bounds, lower_bounds, feature, dtype=torch.float32):
        self.unit = unit
        self.upper_bounds = [upper_bounds[i] / feature.range[i] for i in range(len(unit))]
        self.lower_bounds = [lower_bounds[i] / feature.range[i] for i in range(len(unit))]
        self.unit_by_feature = [unit[i] / feature.range[i] for i in range(len(unit))]
        self.raw_unit = unit
        self.raw_upper = upper_bounds
        self.raw_lower = lower_bounds
        self.dtype = dtype

    def predict(self, models, x, y_prime, modeling, fake_gradient=True):
        """
        Args:
            models (list): 예측에 사용할 모델들의 리스트. torch.nn.Module 객체나 scikit-learn 모델 가능.
            x (torch.Tensor): 입력 feature 벡터.
            y_prime (torch.Tensor): 목표값(Desired Value) normalized 상태.
            modeling (str): 모델링 방식. 'single', 'averaging' 중 하나.
            fake_gradient (bool, optional): Non-neural 모델에서 fake gradient를 사용할지 여부. 기본값은 True.

        Returns:
            predictions (list): 각 모델의 예측 결과 리스트.
            gradients (list): 각 모델에서 계산된 입력값에 대한 기울기(gradient) 리스트.
        """
        predictions = []
        gradients = []
        copy_x = x.clone()
        for model in models:
            x = copy_x.clone().detach().requires_grad_(True)
            if isinstance(model, torch.nn.Module):  # Neural network model
                prediction = model(x)
                loss = torch.abs(prediction - y_prime)
                loss.backward()
                gradient = x.grad.detach().numpy()
                prediction = prediction.detach().numpy()
            else:  # Non-neural model
                x_np = x.detach().numpy().reshape(1, -1)
                prediction = model.predict(x_np)
                if fake_gradient:
                    gradient = []
                    for j in range(x_np.shape[1]):
                        new_x = x_np.copy()
                        new_x[0, j] += self.unit_by_feature[j]
                        new_prediction = model.predict(new_x)
                        slope = (new_prediction - prediction) / self.unit_by_feature[j]
                        gradient.append(slope)
                    gradient = np.array(gradient).reshape(-1)
                else:
                    gradient = np.zeros(x_np.shape[1])
            x = copy_x.clone()
            predictions.append(prediction)
            gradients.append(gradient)
        if modeling == 'single':
            return predictions[0], gradients[0]
        elif modeling == 'averaging':
            avg_prediction = sum(predictions) / len(predictions)
            avg_gradient = sum(gradients) / len(gradients)
            return avg_prediction, avg_gradient
        else:
            raise ValueError(f"Unknown modeling type: {modeling}")

    def bounding(self, configuration):
        ''' Args:
        configuration (list[float]): 후보 configuration 값 리스트 (normalized 상태).
    
        Returns:
        torch.Tensor: 바운딩된 configuration 값 (범위를 초과하지 않도록 제한됨).
        '''
        new = []
        for k, element in enumerate(configuration):
            element = element - (element % self.unit_by_feature[k])
            if element >= self.upper_bounds[k]:
                element = self.upper_bounds[k]
            elif element <= self.lower_bounds[k]:
                element = self.lower_bounds[k]
            new.append(element)
        return torch.tensor(new, dtype=self.dtype)

    def truncate(self, configuration):
        """
        Args:
            configuration (list[float]): 후보 configuration 값 리스트 (raw 값 상태).
        
        Returns:
            list[float]: 단위(unit)에 맞게 자르고 상/하한값을 초과하지 않는 값으로 변경된 configuration 값.
        """
        new_configuration = []
        for i, value in enumerate(configuration):
            value = value - (value % self.raw_unit[i])
            value = max(min(value, self.raw_upper[i]), self.raw_lower[i])
            new_configuration.append(value)
        return new_configuration

class LocalMode(UNIVERSE):
    def __init__(self, desired, models, modeling, strategy, unit, upper_bounds, lower_bounds, feature, dtype=torch.float32):
        super().__init__(unit, upper_bounds, lower_bounds, feature, dtype=dtype)
        """
        Args:
            desired (float): 목표값 (Desired Value, raw 상태).
            models (list): 예측에 사용할 모델 리스트.
            modeling (str): 모델링 방식. 'single', 'averaging' 중 하나.
            strategy (str): 로컬 탐색 전략. 'exhaustive', 'manual', 'sensitivity' 중 하나.
            unit (list[float]): 각 feature의 단위 변화 크기.
            upper_bounds (list[float]): 각 feature의 상한값.
            lower_bounds (list[float]): 각 feature의 하한값.
            feature (Feature): 각 feature의 범위 (range)와 min/max 값을 포함하는 객체.
            dtype (torch.dtype, optional): Tensor 데이터 타입. 기본값은 torch.float32.
        """
        self.desired = desired
        self.y_prime = torch.tensor([desired / feature.range[-1]], dtype=dtype)
        self.models = models
        self.modeling = modeling
        self.strategy = strategy
        self.feature = feature  # feature 객체를 클래스 변수로 저장

    def exhaustive(self, starting_point, top_k=5, alternative='keep_move'):
        """
        Args:
            starting_point (list[float]): 시작점 configuration (raw 상태).
            top_k (int, optional): 중요도 높은 feature의 개수. 기본값은 5.
            alternative (str, optional): 탐색 방식. 'keep_move', 'up_down', 'keep_up_down' 중 하나. 기본값은 'keep_move'.
        
        Returns:
            configurations (list[list[float]]): 탐색된 configuration 값들.
            predictions (list[float]): 각 configuration에 대한 예측값.
            best_config (list[float]): 최적 configuration 값.
            best_pred (float): 최적 configuration에 대한 예측값.
        """
        starting_point = self.truncate(starting_point)
        starting_point_norm = np.array([starting_point[i] / self.feature.range[i] for i in range(len(self.unit))])

        self.top_k = top_k
        self.recorder = []

        self.search_space = []
        self.counter = []
        if alternative == 'keep_up_down':
            variables = [[0, 1, 2]] * len(self.unit)
        else:
            variables = [[0, 1]] * len(self.unit)
        self.all_combinations = list(product(*variables))
        self.adj = []

        prediction_km, gradient_km = self.predict(self.models, torch.tensor(starting_point_norm, dtype=self.dtype), self.y_prime, self.modeling)
        prediction_km = prediction_km[0] if isinstance(prediction_km, list) else prediction_km

        for combination in self.all_combinations:
            count = combination.count(1) if alternative == 'up_down' else combination.count(0)
            adjustment = np.zeros(len(self.unit_by_feature))
            for i, action in enumerate(combination):
                if alternative == 'up_down':
                    if action == 1:
                        adjustment[i] += self.unit_by_feature[i]
                    elif action == 0:
                        adjustment[i] -= self.unit_by_feature[i]
                elif alternative == 'keep_move':
                    if prediction_km > self.y_prime:
                        if action == 1:
                            if gradient_km[i] >= 0:
                                adjustment[i] -= self.unit_by_feature[i]
                            else:
                                adjustment[i] += self.unit_by_feature[i]
                    else:
                        if action == 1:
                            if gradient_km[i] >= 0:
                                adjustment[i] += self.unit_by_feature[i]
                            else:
                                adjustment[i] -= self.unit_by_feature[i]
                elif alternative == 'keep_up_down':
                    important_features = np.argsort(abs(gradient_km))[::-1][:self.top_k]
                    if i in important_features:
                        if action == 2:
                            adjustment[i] += self.unit_by_feature[i]
                        elif action == 1:
                            adjustment[i] -= self.unit_by_feature[i]
            candidate = starting_point_norm + adjustment
            candidate = self.bounding(candidate)
            candidate_str = str(candidate)
            if candidate_str not in self.recorder:
                self.search_space.append(candidate)
                self.counter.append(count)
                self.recorder.append(candidate_str)

        self.predictions = []
        self.configurations = []
        for candidate in self.search_space:
            prediction, _ = self.predict(self.models, candidate, self.y_prime, self.modeling)
            prediction_denorm = prediction * self.feature.range[-1] + self.feature.min[-1]
            configuration_denorm = self.feature.denormalize(candidate)
            self.predictions.append(prediction_denorm[0])
            self.configurations.append(configuration_denorm.tolist())

        self.table = pd.DataFrame({
            'configurations': self.configurations,
            'predictions': self.predictions,
            'difference': np.abs(np.array(self.predictions) - self.desired),
            'counter': self.counter
        })

        self.table = self.table.sort_values(by='difference', ascending=True).sort_values(by='counter', ascending=False)
        self.configurations = self.table['configurations'].tolist()
        self.predictions = self.table['predictions'].tolist()
        self.difference = self.table['difference'].tolist()
        self.counter = self.table['counter'].tolist()

        best_config = self.configurations[0]
        best_pred = self.predictions[0]

        return self.configurations, self.predictions, best_config, best_pred

    def manual(self, starting_point, index=0, up=True):
        """
        Args:
            starting_point (list[float]): 시작점 configuration (raw 상태).
            index (int, optional): 조정할 feature의 인덱스. 기본값은 0.
            up (bool, optional): feature 값을 올릴지 여부. False면 값을 내림. 기본값은 True.
        
        Returns:
            configurations (list[list[float]]): 조정된 configuration 값.
            predictions (list[float]): 조정된 configuration에 대한 예측값.
            best_config (list[float]): 조정된 최적 configuration 값.
            best_pred (float): 조정된 최적 configuration에 대한 예측값.
        """
        starting_point = self.truncate(starting_point)
        starting_point_norm = np.array([starting_point[i] / self.feature.range[i] for i in range(len(self.unit))])

        adjustment = np.zeros(len(self.unit_by_feature))
        if up:
            adjustment[index] += self.unit_by_feature[index]
        else:
            adjustment[index] -= self.unit_by_feature[index]
        position = starting_point_norm + adjustment
        position = self.bounding(position)
        prediction, _ = self.predict(self.models, position, self.y_prime, self.modeling)
        prediction_denorm = prediction * self.feature.range[-1] + self.feature.min[-1]
        configuration_denorm = self.feature.denormalize(position)
        return [configuration_denorm.tolist()], prediction_denorm[0], configuration_denorm.tolist(), prediction_denorm[0]

class GlobalMode(UNIVERSE):
    def __init__(self, desired, models, modeling, strategy, tolerance, steps, unit, upper_bounds, lower_bounds, feature, dtype=torch.float32):
        super().__init__(unit, upper_bounds, lower_bounds, feature, dtype=dtype)
        """
            Args:
            desired (float): 목표값 (Desired Value, raw 상태).
            models (list): 예측에 사용할 모델 리스트.
            modeling (str): 모델링 방식. 'single', 'averaging' 중 하나.
            strategy (str): 글로벌 탐색 전략. 'beam', 'stochastic', 'best_one' 중 하나.
            tolerance (float): 목표값에 대한 허용 오차.
            steps (int): 탐색 반복 횟수.
            unit (list[float]): 각 feature의 단위 변화 크기.
            upper_bounds (list[float]): 각 feature의 상한값.
            lower_bounds (list[float]): 각 feature의 하한값.
            feature (Feature): 각 feature의 범위 (range)와 min/max 값을 포함하는 객체.
            dtype (torch.dtype, optional): Tensor 데이터 타입. 기본값은 torch.float32.
        """
        self.desired = desired
        self.y_prime = torch.tensor([desired / feature.range[-1]], dtype=dtype)
        self.models = models
        self.modeling = modeling
        self.strategy = strategy
        self.tolerance = tolerance / feature.range[-1]
        self.steps = steps
        self.feature = feature  # feature 객체를 클래스 변수로 저장

    def beam(self, starting_point, beam_width=5):
        y_prime = self.y_prime
        tolerance = self.tolerance
        x_i = [starting_point[i] / self.feature.range[i] for i in range(len(self.unit))]
        x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], dtype=self.dtype)

        self.beam_positions = []
        self.beam_targets = []
        self.beam_positions_denorm = []
        self.beam_targets_denorm = []

        for step in range(self.steps):
            if not self.beam_positions:
                current_positions = [x_prime.clone().detach()]
                for _ in range(beam_width - 1):
                    random_offsets = torch.tensor(np.random.uniform(-0.5, 0.5, len(self.unit)), dtype=self.dtype)
                    current_positions.append(x_prime.clone().detach() + random_offsets)
            else:
                current_positions = self.beam_positions[-1]

            configurations = []
            candidates = []
            candidates_score = []
            beam_predictions = []

            for current_pos in current_positions:
                configuration_denorm = self.feature.denormalize(current_pos.clone().detach())
                configurations.append(configuration_denorm.tolist())

                predictions, gradients = self.predict(self.models, current_pos, y_prime, self.modeling)
                prediction_avg = sum(predictions) / len(predictions)
                gradient_avg = sum(gradients) / len(gradients)

                prediction_original = prediction_avg * self.feature.range[-1] + feature.min[-1]
                beam_predictions.append(prediction_original[0])

                order = np.argsort(abs(gradient_avg))[::-1]
                beam = order[:beam_width]

                for b in beam:
                    adjustment = np.zeros(len(self.unit_by_feature))
                    if gradient_avg[b] >= 0:
                        adjustment[b] += self.unit_by_feature[b]
                    else:
                        adjustment[b] -= self.unit_by_feature[b]

                    if prediction_avg > y_prime:
                        position = current_pos.clone().detach() - torch.tensor(adjustment, dtype=self.dtype)
                    else:
                        position = current_pos.clone().detach() + torch.tensor(adjustment, dtype=self.dtype)

                    position = self.bounding(position)
                    candidates.append(position)
                    candidates_score.append(abs(gradient_avg[b]))

            select_indices = np.argsort(candidates_score)[::-1][:beam_width]
            new_positions = [candidates[i] for i in select_indices]

            self.beam_positions.append(new_positions)
            self.beam_targets.append(beam_predictions)
            self.beam_positions_denorm.append(configurations)
            self.beam_targets_denorm.append(beam_predictions)

            if any(abs(p - self.desired) < tolerance * self.feature.range[-1] for p in beam_predictions):
                break

        flattened_positions = [pos for positions in self.beam_positions_denorm for pos in positions]
        flattened_predictions = [pred for preds in self.beam_targets_denorm for pred in preds]
        differences = [abs(pred - self.desired) for pred in flattened_predictions]
        best_index = np.argmin(differences)

        best_position = flattened_positions[best_index]
        best_prediction = flattened_predictions[best_index]

        return self.beam_positions_denorm, self.beam_targets_denorm, best_position, best_prediction

    def stochastic(self, starting_point, num_candidates=5):
        y_prime = self.y_prime
        tolerance = self.tolerance
        x_i = [starting_point[i] / self.feature.range[i] for i in range(len(self.unit))]
        x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], dtype=self.dtype)

        self.stochastic_chosen = []
        self.stochastic_predictions = []
        self.stochastic_configurations = []

        for step in range(self.steps):
            configuration_denorm = self.feature.denormalize(x_prime)
            predictions, gradients = self.predict(self.models, x_prime, y_prime, self.modeling)
            prediction_avg = sum(predictions) / len(predictions)
            gradient_avg = sum(gradients) / len(gradients)

            candidates = np.argsort(abs(gradient_avg))[::-1][:num_candidates]
            chosen = random.choice(candidates)

            adjustment = np.zeros(len(self.unit_by_feature))
            if gradient_avg[chosen] >= 0:
                adjustment[chosen] += self.unit_by_feature[chosen]
            else:
                adjustment[chosen] -= self.unit_by_feature[chosen]

            if prediction_avg > y_prime:
                x_prime -= torch.tensor(adjustment, dtype=self.dtype)
            elif prediction_avg < y_prime:
                x_prime += torch.tensor(adjustment, dtype=self.dtype)

            x_prime = self.bounding(x_prime)

            prediction_original = prediction_avg * self.feature.range[-1] + self.feature.min[-1]

            if step % 10 == 0 and step != 0:
                print(f"Step {step} Target: {self.desired}, Prediction: {prediction_original.item()}")

            self.stochastic_chosen.append(chosen)
            self.stochastic_predictions.append(prediction_original.item())
            self.stochastic_configurations.append(configuration_denorm.tolist())

            if abs(prediction_avg - y_prime) < tolerance:
                break

        best_index = np.argmin(np.abs(np.array(self.stochastic_predictions) - self.desired))

        best_position = self.stochastic_configurations[best_index]
        best_prediction = self.stochastic_predictions[best_index]

        return self.stochastic_configurations, self.stochastic_predictions, best_position, best_prediction

    def best_one(self, starting_point, escape=True):
        y_prime = self.y_prime
        tolerance = self.tolerance
        x_i = [starting_point[i] / self.feature.range[i] for i in range(len(self.unit))]
        x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], dtype=self.dtype)

        self.best_one_chosen = []
        self.best_one_predictions = []
        self.best_one_configurations = []

        avoid = []
        memory = []
        memory_size = 5
        previous = None

        for step in range(self.steps):
            configuration_denorm = self.feature.denormalize(x_prime)
            predictions, gradients = self.predict(self.models, x_prime, y_prime, self.modeling)
            prediction_avg = sum(predictions) / len(predictions)
            gradient_avg = sum(gradients) / len(gradients)

            if escape:
                candidates = [i for i in np.argsort(abs(gradient_avg))[::-1] if i not in avoid]
            else:
                candidates = np.argsort(abs(gradient_avg))[::-1]
            chosen = candidates[0]

            adjustment = np.zeros(len(self.unit_by_feature))
            if gradient_avg[chosen] >= 0:
                adjustment[chosen] += self.unit_by_feature[chosen]
            else:
                adjustment[chosen] -= self.unit_by_feature[chosen]

            if prediction_avg > y_prime:
                x_prime -= torch.tensor(adjustment, dtype=self.dtype)
            elif prediction_avg < y_prime:
                x_prime += torch.tensor(adjustment, dtype=self.dtype)

            x_prime = self.bounding(x_prime)

            prediction_original = prediction_avg * self.feature.range[-1] + self.feature.min[-1]

            if step % 10 == 0 and step != 0:
                print(f"Step {step} Target: {self.desired}, Prediction: {prediction_original.item()}")

            self.best_one_chosen.append(chosen)
            self.best_one_predictions.append(prediction_original.item())
            self.best_one_configurations.append(configuration_denorm.tolist())
            memory.append(prediction_original.item())
            if len(memory) > memory_size:
                memory = memory[-memory_size:]
            if len(memory) == memory_size and len(set(memory)) < 3 and previous == chosen:
                avoid.append(chosen)

            if abs(prediction_avg - y_prime) < tolerance:
                break
            if escape and len(avoid) == len(self.unit):
                break
            previous = chosen

        best_index = np.argmin(np.abs(np.array(self.best_one_predictions) - self.desired))

        best_position = self.best_one_configurations[best_index]
        best_prediction = self.best_one_predictions[best_index]

        return self.best_one_configurations, self.best_one_predictions, best_position, best_prediction
