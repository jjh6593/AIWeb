import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time  
import random
import xgboost
import sklearn
from itertools import product
from itertools import chain
import itertools

def parameter_prediction(data, models, desired, starting_point, mode, modeling, strategy, tolerance, beam_width,
        num_candidates, escape, top_k, index, up, alternative, unit, lower_bound, upper_bound, data_type, decimal_place):

    configuration_patience = 10
    configuration_patience_volume = 0.01
    configuration_steps = 100 # step
    configuration_eta = 10
    configuration_eta_decay = 0.001
    configuration_show_steps = True
    configuration_show_focus = False
    configuration_tolerance = tolerance
    configuration_retrial_threshold = 10
    start_from_standard = False
    start_from_random = False
    constrain_reselection = True

    if_visualize = True

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    output_size = 1
    input_size = data.values.shape[1] - output_size
    print("The number of feature  : ",input_size)
    dtype = torch.float32
    
    def create_constraints(unit, lower_bound, upper_bound, data_type, decimal_place):
        attribute_names = [
            f"ATT{i + 1}" for i in range(len(unit))
        ]

        constraints = {}
        for i, name in enumerate(attribute_names):
        # float 혹은 int로 변환
            u = float(unit[i])               # 단위
            low = float(lower_bound[i])      # 하한
            up = float(upper_bound[i])       # 상한
            dt = data_type[i]                # str 또는 type(예: int, float)
            dp = int(decimal_place[i])       # 소수점 자리수

            constraints[name] = [u, low, up, dt, dp]
        print(constraints)
        return constraints    
    if starting_point is None:
        raise ValueError("starting_point cannot be None")
    if desired is None:
        raise ValueError("desired cannot be None")


    constraints = create_constraints(unit = unit, 
                                     lower_bound = lower_bound, 
                                     upper_bound = upper_bound, 
                                     data_type = data_type, 
                                     decimal_place = decimal_place)    
    
    

    class MinMaxScaling:
        """
        A class for normalizing and denormalizing data using Min-Max scaling.
        """
        def __init__(self, data):
            """
            Initializes the MinMaxScaling object.

            Args:
                data (pd.DataFrame): Input data to be scaled.
            """
            self.max, self.min, self.range = [], [], []
            self.data = pd.DataFrame()

            # Reshape data if necessary
            data = data.values.reshape(-1, 1) if len(data.values.shape) == 1 else data.values

            epsilon = 2  # Small adjustment to avoid division by zero

            for i in range(data.shape[1]):
                max_, min_ = max(data[:, i]), min(data[:, i])
                if max_ == min_:
                    max_ *= epsilon

                self.max.append(max_)
                self.min.append(min_)
                self.range.append(max_ - min_)

                # Normalize the column and add to the DataFrame
                normalized_column = (data[:, i]) / (max_ - min_)
                self.data = pd.concat([self.data, pd.DataFrame(normalized_column)], axis=1)

            # Convert normalized data to a torch tensor
            self.data = torch.tensor(self.data.values, dtype=dtype)
            print(self.data)

        def denormalize(self, data):
            """
            Denormalizes data back to its original scale.

            Args:
                data (torch.Tensor or np.ndarray): Normalized data to be converted.

            Returns:
                list: Denormalized data.
            """
            # Convert torch tensor to numpy array if necessary
            # data = data.detach().numpy() if isinstance(data, torch.Tensor) else data

            # new_data = []
            # for i, element in enumerate(data):
            #     element = element * (self.max[i] - self.min[i])
            #     element = round(element, np.array(list(constraints.values()))[:, 4][i])
            #     new_data.append(element)
            # torch.Tensor인 경우 numpy 배열로 변환
            data = data.detach().numpy() if isinstance(data, torch.Tensor) else data

            new_data = []
            # constraints의 소수점 자리수를 미리 추출 (정수형으로 변환)
            precisions = [int(v[4]) for v in list(constraints.values())]

            for i, element in enumerate(data):
                # element가 numpy 배열인 경우 스칼라 값으로 변환
                scalar_val = float(element)
                # 원래 스케일로 복원
                scalar_val = (scalar_val) * (self.max[i] - self.min[i])   #+ self.min[i]
                # 반올림: precisions[i]는 정수여야 함
                scalar_val = round(scalar_val, precisions[i])
                new_data.append(scalar_val)
            return new_data


    class UNIVERSE :
        def __init__(self, constraints = constraints):
            self.constraints = constraints
            self.unit_by_feature = [np.array(list(constraints.values()))[:,0][i] / (feature.max[i] - feature.min[i])
                                    for i in range(input_size)]
            self.upper_bounds = [np.array(list(constraints.values()))[:,2][i] / (feature.max[i] - feature.min[i])  
                                 for i in range(input_size)]
            self.lower_bounds = [np.array(list(constraints.values()))[:,1][i] / (feature.max[i] - feature.min[i])  
                                 for i in range(input_size)]
            
            # 디버그 출력
            print("=== 각 피처에 대한 scaling factor (unit_by_feature) ===")
            for i in range(input_size):
                print(f"Feature {i+1}: unit_by_feature = {self.unit_by_feature[i]}, "
                    f"upper_bound = {self.upper_bounds[i]}, lower_bound = {self.lower_bounds[i]}")
            self.raw_unit = np.array(list(constraints.values()))[:,0].tolist()
            self.raw_lower = np.array(list(constraints.values()))[:,1].tolist()
            self.raw_upper = np.array(list(constraints.values()))[:,2].tolist()
            

        def predict(self, models, x, y_prime, modeling, fake_gradient = True):
            predictions,gradients = [], []
            copy = x.clone()
            for i, model in enumerate(models):
                x = x.clone().detach().requires_grad_(True)
                if isinstance(model, nn.Module):
                    prediction = model(x)
                    loss = abs(prediction - y_prime)
                    loss.backward()
                    gradient = x.grad.detach().numpy()
                    prediction = prediction.detach().numpy()

                else: # ML
                    x = x.detach().numpy().reshape(1,-1)
                    prediction = model.predict(x)
                    if fake_gradient :
                        gradient = []
                        for j in range(x.shape[1]):
                            new_x = x.copy()
                            new_x[0,j] += self.unit_by_feature[j]
                            new_prediction = model.predict(new_x)
                            slope = (new_prediction - prediction) / self.unit_by_feature[j]
                            gradient.append(slope)
                        gradient = np.array(gradient).reshape(-1)   
                    else : gradient = np.repeat(0, x.shape[1])
                x = copy.clone()
                predictions.append(prediction)
                gradients.append(gradient)

            if modeling.lower() == 'single': return predictions[0], gradients[0], predictions
            elif modeling == 'ensemble': return sum(predictions)/len(predictions), sum(gradients)/len(gradients), predictions  
            else: raise Exception(f"[modeling error] there is no {modeling}.")


        def bounding(self, configuration):
            new = []
            for k, element in enumerate(configuration):
                element = element - (element % self.unit_by_feature[k])
                # print(f"element % self.unit_by_feature[k] = {element % self.unit_by_feature[k]}")
                if element >= self.upper_bounds[k]:
                    # print("경계 상한 도달")
                    element = self.upper_bounds[k]
                elif element <= self.lower_bounds[k]:
                    # print("경계 하한 도달")
                    element = self.lower_bounds[k]
                else: pass
                new.append(element)        
            configuration = torch.tensor(new, dtype = dtype)        
            return configuration


        def truncate(self, configuration):
            new_configuration = []
            for i, value in enumerate(configuration) :
                value = value - (value % self.raw_unit[i])
                value = value if value >= self.raw_lower[i] else self.raw_lower[i]
                value = value if value <= self.raw_upper[i] else self.raw_upper[i]
                new_configuration.append(value)
           # configuration = torch.tensor(new_configuration, dtype = dtype)     
            return configuration
        
    class LocalMode(UNIVERSE):
        def __init__(self, desired, models, modeling, strategy):
            super().__init__()
            self.desired = desired
            self.y_prime = self.desired / (target.max[0] - target.min[0])
            self.models = []
            self.modeling = modeling
            self.strategy = strategy
            for model in models : 
                if isinstance(model, nn.Module) : model.eval()
                self.models.append(model)

        def exhaustive(self, starting_point, top_k = 5, alternative = 'keep_move'):
            self.starting_point = super().truncate(starting_point)
            self.starting_point = np.array([self.starting_point[i] / (feature.max[i] - feature.min[i]) 
                                            for i in range(input_size)])
            self.top_k = top_k
            self.recorder = []

            self.search_space, self.counter = [],[]
            if alternative == 'keep_up_down' :
                variables = [[0, 1, 2]] * input_size
            else :
                variables = [[0, 1]] * input_size
            self.all_combinations = list(product(*variables))
            self.adj = []
            print(torch.tensor(self.starting_point, dtype = dtype))
            if alternative == 'keep_move' or alternative == 'keep_up_down':
                prediction_km, gradient_km, p_all = super().predict(self.models,torch.tensor(self.starting_point, dtype = dtype), 
                                               self.y_prime, self.modeling, fake_gradient = True)
                prediction_km = prediction_km[0] if isinstance(prediction_km,list) else prediction_km
            for combination in self.all_combinations :
                count = combination.count(1) if alternative == 'up_down' else combination.count(0)
                adjustment = np.repeat(0,len(self.unit_by_feature)).tolist()
                for i, boolean in enumerate(list(combination)):
                    if alternative == 'up_down':
                        if boolean == 1 :   adjustment[i] = adjustment[i] + self.unit_by_feature[i]
                        elif boolean == 0 : adjustment[i] = adjustment[i] - self.unit_by_feature[i]
                        else: raise Exception("ERROR")

                    elif alternative == 'keep_move':
                        if prediction_km > self.y_prime :                        
                            if boolean == 1 :   
                                if gradient_km[i] >= 0 :
                                    adjustment[i] = adjustment[i] - self.unit_by_feature[i]
                                else:
                                    adjustment[i] = adjustment[i] + self.unit_by_feature[i]
                            elif boolean == 0 : pass
                            else: raise Exception("ERROR")            

                        else: 
                            if boolean == 1 :   
                                if gradient_km[i] >= 0 :
                                    adjustment[i] = adjustment[i] + self.unit_by_feature[i]
                                else:
                                    adjustment[i] = adjustment[i] - self.unit_by_feature[i]
                            elif boolean == 0 : pass
                            else: raise Exception("ERROR")    

                    elif alternative == 'keep_up_down' :
                        important_features = np.argsort(abs(gradient_km))[::-1][:self.top_k]
                        if i in important_features :
                            if boolean == 2 :   adjustment[i] = adjustment[i] + self.unit_by_feature[i]
                            elif boolean == 1 : adjustment[i] = adjustment[i] - self.unit_by_feature[i]
                            elif boolean == 0 : pass
                            else: raise Exception("ERROR")   

                self.adj.append(adjustment)
                candidate = self.starting_point + adjustment         
                candidate = super().bounding(candidate)
                if str(candidate) not in self.recorder : 
                    self.search_space.append(candidate)
                    self.counter.append(count)
                    self.recorder.append(str(candidate))
        #    print(len(self.search_space))
            self.predictions = []
            self.configurations = []
            self.pred_all = []
            for candidate in self.search_space :
                prediction, _, p_all = super().predict(self.models,candidate, self.y_prime, self.modeling, fake_gradient = False)
                prediction = target.denormalize(prediction)[0]
                configuration = feature.denormalize(candidate)
                self.predictions.append(prediction)
                self.configurations.append(configuration)
                self.pred_all.append([target.denormalize([e.item()])[0] for e in p_all])
                

            self.table = pd.DataFrame({'configurations':self.configurations,'find_dup' : self.configurations,
                                      'predictions':self.predictions,'difference' : np.array(abs(np.array(self.predictions)-self.desired)).tolist(),
                                      'counter':self.counter})
            self.table['find_dup'] = self.table['find_dup'].apply(lambda x: str(x))
            self.table = self.table[~self.table.duplicated(subset='find_dup', keep='first')]
            self.table = self.table.drop(columns=['find_dup'])

            self.table = self.table.sort_values(by='counter', ascending=False).sort_values(by='difference', ascending=True)
            self.configurations = self.table['configurations']
            self.predictions = self.table['predictions']
            self.difference = self.table['difference']
            self.counter = self.table['counter']
            
            configurations = self.configurations[:].values.tolist()
            predictions = self.predictions[:].values.tolist()
            best_config = configurations[0]
            best_pred = predictions[0]

            try: return configurations, predictions, best_config, best_pred, self.pred_all
            except : return self.configurations[:].values.tolist(), self.predictions[:].values.tolist(), best_config, best_pred, self.pred_all

        def manual(self, starting_point, index=0, up=True):
            self.starting_point = super().truncate(starting_point)
            self.starting_point = np.array([self.starting_point[i] / (feature.max[i] - feature.min[i]) 
                                            for i in range(input_size)])

            adjustment = np.repeat(0,len(self.unit_by_feature)).tolist()
            if up : adjustment[index] += self.unit_by_feature[index]
            else : adjustment[index] -= self.unit_by_feature[index]
            position = self.starting_point + adjustment         
            position = super().bounding(position)        
            prediction, _, p_all = super().predict(self.models,position, self.y_prime, self.modeling)
            prediction = target.denormalize(prediction)
            configuration = feature.denormalize(position)
            p_all = [target.denormalize([e.item()])[0] for e in p_all]
            return [configuration], prediction, configuration, prediction, p_all 
        
    class GlobalMode(UNIVERSE):
        def __init__(self, desired, models, modeling, strategy, tolerance = configuration_tolerance, steps = configuration_steps):
            super().__init__()
            self.desired = desired
            self.y_prime = self.desired / (target.max[0] - target.min[0])
            self.models = []
            self.modeling = modeling
            self.strategy = strategy
            self.tolerance = tolerance / (target.max[0] - target.min[0])
            self.original_tolerance = tolerance
            self.steps = steps
            for model in models : 
                if isinstance(model, nn.Module): model.train()
                self.models.append(model)

        def predict_global(
                self, 
                models, 
                x, 
                y_prime, 
                method = "fd", 
                alpha=1, 
                p = 0.1,
                m = 5
                ):
            
            predictions,gradients = [], []
            copy = x.clone()
            for i, model in enumerate(models):
                x = torch.tensor(x, dtype=torch.float32)
                x = x.clone().detach().requires_grad_(True)
                # 변형은 여기부터
                if isinstance(model, nn.Module):
                    prediction = model(x)
                    # loss = prediction - y_prime # 기존 방식
                    loss = abs(prediction - y_prime) # MAE
                    # loss = (prediction - y_prime) ** 2 # MSE
                    loss.backward()
                    gradient = x.grad.detach().numpy()
                    prediction = prediction.detach().numpy()
                
                else:
                    
                    x = x.detach().numpy().reshape(1,-1)  # x : ndarray
                    
                    y = model.predict(x)

                    # prediction = model(x) # 모델의 출력 계산
                    # y = model(x)
                
                    prediction = y
                    assert m >= 1  and m <= len(self.unit_by_feature)
                    sampled_indedices = np.random.choice(
                        np.arange(len(self.unit_by_feature),dtype = int),replace = False, size = m).tolist()
                    _unit_by_feature = np.array(self.unit_by_feature)[sampled_indedices].tolist()
                    
                    if method == "fd":
                    
                        best_g = [0] * len(self.unit_by_feature)
                        # best_g를 리스트 대신 텐서로 초기화합니다
                    elif method == "target":
                        best_g = abs(y_prime - y)
                        best_change = [0] * len(self.unit_by_feature)
                    else:
                        raise ValueError("Expected 'fd' or 'target', but got {}".format(method))

                    # for change in itertools.product([-1, 0, 1], repeat=len(self.unit_by_feature)):
                    
                    for change in itertools.product([-1, 0, 1], repeat=len(_unit_by_feature)):
                        '''
                        x : ndarray
                        change : Tuple
                        unit_by_feature : List
                        '''
                        
                        # xx = x + np.array([xx2 * xx3 for xx2, xx3 in zip(change, self.unit_by_feature)])
                        if not np.random.uniform() < p:
                            continue
                        # 이거 아래가 정답
                        xx = x.copy()
                        # xx = x + alpha * np.array([xx2 * xx3 for xx2, xx3 in zip(change, self.unit_by_feature)])
                        # xx[0] = xx[0] + alpha * np.array([xx2 * xx3 for xx2, xx3 in zip(change, self.unit_by_feature)])
                        
                        xx[0][sampled_indedices] = xx[0][sampled_indedices] + alpha * np.array([xx2 * xx3 \
                            for xx2, xx3 in zip(change, _unit_by_feature)])
                        xx[0] = np.clip(xx[0], a_min = self.lower_bounds, a_max = self.upper_bounds)
                        
                        # 이거는 DL 쓸 때
                        # delta_tensor = torch.tensor([xx2 * xx3 for xx2, xx3 in zip(change, self.unit_by_feature)],
                        #         dtype=x.dtype, device=x.device)
                        # xx = x + alpha * delta_tensor
                        
                        # yy = model.predict(xx)
                        # xx = torch.tensor(xx)
                        # ML 적용
                        yy = model.predict(xx)
                        # DL 적용
                        # yy = model(xx)
                        
                        # print("yy {}".format(yy))
                        if method == "fd":  # finite difference-based
                        
                            l = abs(y_prime - y)
                            ll = abs(y_prime - yy)
                            # g = (yy - y) / np.array(unit_by_feature)
                            # 아래 3개만 해제
                        
                            g = (ll - l) / np.array(self.unit_by_feature)
                            
                            g_norm = np.linalg.norm(g)
                            
                            best_g_norm = np.linalg.norm(best_g)
                            

                            # 기준
                            # unit_tensor = torch.tensor(self.unit_by_feature, dtype=ll.dtype, device=ll.device)
                            
                            # g = (ll - l) / unit_tensor
                            # g_np = g.detach().numpy()  # 여기서 detach()를 호출!
                            # g_norm = np.linalg.norm(g_np)
                            # best_g_norm = np.linalg.norm(best_g.detach().numpy())  # 여기서 detach()를 호출!
                            # DL 일 때는 위에 5개 지우고
                            
                            
                            if best_g_norm < g_norm:
                            
                                best_g = g

                        elif method == "target":  # target based
                            g = abs(y_prime - yy)  # |target - pred|
                            # print(f'g:{g}')
                            if best_g > g:
                                best_g = g
                                # best_change = change
                                if m is not None:
                                    best_change = [0] * len(self.unit_by_feature)
                                    best_change = np.array(best_change)
                                    best_change[sampled_indedices] = change
                                    best_change = best_change.tolist()
                                else:
                                    best_g = g
                            
                            
                        else:
                            raise ValueError("Expected 'fd' or 'target', but got {}".format(method))
                    # 여기까지 들여쓰기
                    # ML    
                        if method == "fd":
                            
                            gradient = np.array(best_g)
                        elif method == "target":
                            gradient = np.array(best_change)
                    # DL
                    # if method == "fd":
                    #     gradient = g.detach().numpy()  # best_g가 텐서라면 detach() 후 numpy() 호출
                    # elif method == "target":
                    #     gradient = np.array(best_change)  # best_change는 파이썬 리스트이므로 괜찮음
                    # else:
                    #     raise ValueError
            
                x = copy.clone()
                
                predictions.append(prediction)
                gradients.append(gradient)
                
                
            return predictions, gradients

        def beam(self, beam_width, starting_point) :
            self.beam_width = beam_width
            y_prime = self.desired / (target.max[0] - target.min[0])
            tolerance = self.tolerance
            final = None

            x_i = [starting_point[i] / (feature.max[i] - feature.min[i]) for i in range(input_size)]  
            x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], 
                                   dtype = dtype)

            self.beam_positions, self.beam_targets = [], []
            self.beam_positions_denorm, self.beam_targets_denorm = [], []

            self.beam_history = []
            self.previous_gradient = [[], [], [], [], []]
            self.prediction_all = []

            success = [False]
            which = []
            close = False
            close_margin = 10
            for step in range(self.steps):
                if len(self.beam_positions) == 0 :
                    current_positions = [x_prime.clone().detach()]
                    for j in range(self.beam_width - 1):
                        random_offsets = torch.tensor(np.random.uniform(-0.5, 0.5, input_size), dtype=x_prime.dtype)
                        current_positions += [x_prime.clone().detach() + random_offsets]
                else :
                    current_positions = self.beam_positions[-1]

                configurations = []
                candidates = []
                candidates_score = []
                beam_predictions = []
                beams = []
                p_all = []
                for p, current_pos in enumerate(current_positions):
                    configuration = feature.denormalize(current_pos.clone().detach())
                    configurations.append(configuration)

                    predictions, gradients = self.predict_global(self.models, x = current_pos, y_prime = y_prime)                
                    prediction_avg = sum(predictions)/len(predictions) ###
                    gradient_avg = sum(gradients)/len(gradients)  

                    prediction_original = prediction_avg * (target.max[0] - target.min[0])
                    prediction_original = prediction_original[0]
                    beam_predictions.append(prediction_original)     
                    p_all.append([target.denormalize([e.item()])[0] for e in predictions])
                    if abs(prediction_original - self.desired) < close_margin : close = True
                    else : close = False
                #    print(close)
                    if abs(prediction_avg - y_prime) < tolerance:
                        best_config = configuration
                        best_pred = prediction_original
                        success.append(True)

               #     if close :
               #         order = np.argsort(abs(gradient_avg))
               #     else :
               #         order = np.argsort(abs(gradient_avg))[::-1]
               #     order = np.random.permutation(np.argsort(abs(gradient_avg))[::-1])
                    order = np.argsort(abs(gradient_avg))[::-1]

                    beam = order[:self.beam_width]
                    
                    for b in beam:

                        adjustment = list(np.repeat(0,len(self.unit_by_feature)))
                        if gradient_avg[b] >= 0 :
                            adjustment[b] += self.unit_by_feature[b]
                        else:
                            adjustment[b] -= self.unit_by_feature[b]    


                        adjustment = np.array(adjustment)
                        if prediction_avg > y_prime : 
                            position = current_pos.clone().detach() - adjustment 
                        else :
                            position = current_pos.clone().detach() + adjustment 
                        # update_delta를 한 번만 계산해서 적용
                        # if isinstance(self.models, nn.Module) or (isinstance(self.models, list) and all(isinstance(m, nn.Module) for m in self.models)):
                        #     update_delta = -adjustment
                        # else:
                        #     if prediction_avg > y_prime:
                        #         update_delta = -adjustment
                        #     elif prediction_avg < y_prime:
                        #         update_delta = adjustment
                        #     else:
                        #         update_delta = np.zeros_like(adjustment)
                        
                        # position = current_pos.clone().detach() + update_delta
                        position = super().bounding(position)
                        candidates.append(position)
                        candidates_score.append(abs(gradient_avg[b]))

                if step % 10 == 0 and step != 0 : print(f"Step {step} Target : {self.desired}, Prediction : {beam_predictions}")
                select = np.argsort(candidates_score)[::-1][:self.beam_width]
                # 오류 메시지 제거
                # new_positions = [torch.tensor(candidates[s], dtype = dtype) for s in select]
                new_positions = [candidates[s].clone().detach().to(dtype) for s in select]

                if len(beam_predictions) == 1 : beam_predictions = list(np.repeat(beam_predictions[0],self.beam_width))
                self.beam_positions.append(new_positions)
                self.beam_targets.append(beam_predictions)
                self.beam_history.append(beam.tolist())
                self.beam_positions_denorm.append(configurations) 
                self.beam_targets_denorm.append(beam_predictions)
                self.prediction_all.append(p_all)
                if any(success): break      

            flattened_positions = list(chain.from_iterable(self.beam_positions_denorm))
            flattened_predictions = list(chain.from_iterable(self.beam_targets_denorm))
            best = int(np.argsort(abs(np.array(flattened_predictions)-self.desired))[0])

            self.best_position = flattened_positions[best]
            self.best_prediction = flattened_predictions[best]

            return self.beam_positions_denorm, self.beam_targets_denorm, self.best_position, self.best_prediction, self.prediction_all


        def stochastic(self, num_candidates = 5, starting_point = starting_point) :
            self.num_candidates = num_candidates
            y_prime = self.desired / (target.max[0] - target.min[0])
            
            tolerance = self.tolerance
            final = None
            
            x_i = [float(starting_point[i]) / (feature.max[i] - feature.min[i]) for i in range(input_size)]
            x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], dtype=dtype)
            
            self.stochastic_chosen = []
            self.stochastic_predictions = []
            self.stochastic_configurations = []
            self.stochastic_predictions_all = []
            self.prediction_all = []
            method = "fd" # 변경사항
            for step in range(self.steps):
                
                configuration = feature.denormalize(x_prime)
                predictions, gradients = self.predict_global(self.models, x = x_prime, y_prime = y_prime)
                
                if isinstance(self.models, nn.Module) or isinstance(self.models, list) and all(isinstance(m, nn.Module) for m in self.models):
                    gradient_avg = sum(gradients)/len(gradients)
                    prediction_avg = sum(predictions)/len(predictions) 
                    gradient_avg = sum(gradients)/len(gradients) # List of ndarray
                    candidates = np.argsort(abs(gradient_avg))[::-1][:self.num_candidates] #[::-1]
                    
                    chosen = random.choice(candidates)
                    
                    adjustment = list(np.repeat(0,len(self.unit_by_feature)))
                    
                    # adjustment 업데이트
                    if gradient_avg[chosen] >= 0:
                        adjustment[chosen] += self.unit_by_feature[chosen]
                    else:
                        adjustment[chosen] -= self.unit_by_feature[chosen]
                    adjustment = np.array(adjustment)
                    update_delta = -adjustment
                    x_prime += update_delta
                    
                else:
                    # 여기부터
                    
                    if not method == "target":
                        
                        # gradients : (List of ndarray)
                        # List => The number of list
                        
                        best_g = [0] * len(self.unit_by_feature)
                        for i in range(len(gradients)):
                            
                            g_norm = np.linalg.norm(gradients[i])
                            
                            best_g_norm = np.linalg.norm(best_g)
                            if best_g_norm < g_norm:
                                
                                best_g = gradients[i]
                        sum_gradients = []
                        for i in range(len(self.unit_by_feature)):
                            if best_g.ndim == 2:
                                best_g = best_g[0]
                            
                            if best_g[i] > 0:
                                sum_gradients.append(1)
                            elif best_g[i] < 0:
                                sum_gradients.append(-1)
                            else:
                                sum_gradients.append(0)
                        sum_gradients = np.array(sum_gradients)
                        
                        
                        # x_prime = x_prime - sum_gradients * self.unit_by_feature # 이거 한줄이면 될거 아래 2줄 추가됌

                        # 수정: torch tensor로 변환하면서, x_prime과 같은 dtype, device로 맞추기
                        sum_gradients_tensor = torch.tensor(sum_gradients, dtype=x_prime.dtype, device=x_prime.device)

                        # self.unit_by_feature가 리스트라면, torch tensor로 변환
                        unit_by_feature_tensor = torch.tensor(self.unit_by_feature, dtype=x_prime.dtype, device=x_prime.device)

                        # 이후 연산에서 모두 torch tensor를 사용
                        x_prime = x_prime - sum_gradients_tensor * unit_by_feature_tensor
                        print(f'x_prime : {x_prime}')
                    # gradients : (List of ndarray)
                    else:
                    # gradients : (List of ndarray)|
                    # (-1, 0, 1, .., 0), .., (1, 1, 1, .., 1) : # models
                    # 앙상블 방향 계산
                    # 각 모델별 방향 정보를 성분별로 더함 # 변수별 계산된 방향과 변수별 단위 곱하고 그걸 각 변수에 더하면 됨 # TODO : NN 모델 + DL 모델 앙상블인 경우 NN 모델에 대한 처리 필요
                        # print(f'gradients : {gradients}')
                        # print(f'unit_by_feature : {self.unit_by_feature}')
                        # print(f'x_prime : {x_prime}')
                        
                        
                        sum_gradients = sum(gradients) # List
                        # x_prime = x_prime + sum_gradients * self.unit_by_feature
                        
                        # 수정: torch tensor로 변환하면서, x_prime과 같은 dtype, device로 맞추기
                        sum_gradients_tensor = torch.tensor(sum_gradients, dtype=x_prime.dtype, device=x_prime.device)
                        
                        # self.unit_by_feature가 리스트라면, torch tensor로 변환
                        unit_by_feature_tensor = torch.tensor(self.unit_by_feature, dtype=x_prime.dtype, device=x_prime.device)
                        
                        # 이후 연산에서 모두 torch tensor를 사용
                        x_prime = x_prime + sum_gradients_tensor * unit_by_feature_tensor
                        print(f'x_prime : {x_prime}')

                    
                    


                prediction_avg = sum(predictions)/len(predictions)
                # 여기까지
                # gradient_avg = sum(gradients)/len(gradients)

                # prediction_avg = sum(predictions)/len(predictions) 
                
                # gradient_avg = sum(gradients)/len(gradients) # List of ndarray
                
                # candidates = np.argsort(abs(gradient_avg))[::-1][:self.num_candidates] #[::-1]
                
                # chosen = random.choice(candidates)
                
                # adjustment = list(np.repeat(0,len(self.unit_by_feature)))
                
                # # adjustment 업데이트
                # if gradient_avg[chosen] >= 0:
                #     adjustment[chosen] += self.unit_by_feature[chosen]
                # else:
                #     adjustment[chosen] -= self.unit_by_feature[chosen]
                # adjustment = np.array(adjustment)

                # update_delta = 0
                # # if isinstance(self.models, nn.Module) or isinstance(self.models, list) and all(isinstance(m, nn.Module) for m in self.models):
                # #     update_delta = -adjustment
                # # else:
                # #     if prediction_avg > y_prime:
                # #         update_delta = -adjustment
                # #     elif prediction_avg < y_prime:
                # #         update_delta = adjustment
                # #     else: pass
                # # x_prime += update_delta
                
                # x_prime += adjustment

                # x_prime 조정
                # if prediction_avg > y_prime:
                #     x_prime -= adjustment
                # elif prediction_avg < y_prime:
                #     x_prime += adjustment
                # else: pass
                # version
                
                # x_prime = super().bounding(x_prime)
                x_prime = np. clip(x_prime, a_min=self.lower_bounds, a_max=self.upper_bounds)
                # print(f'self.unit_by_feature type : {type(self.unit_by_feature)}')
                prediction_original = target.denormalize(prediction_avg)
                if prediction_original is None:
                    raise ValueError(f"Step {step}: prediction_original is None")
                prediction_original = prediction_original[0]
                
             #   prediction_original_all = target.denormalize(predictions_all)
             #   prediction_original_all = prediction_original_all
                
                if configuration_show_steps and step % 10 == 0 and step != 0:
                    print(f"Step {step} Target : {self.desired}, Prediction : {prediction_original}")

                # self.stochastic_chosen.append(chosen)    
                self.stochastic_predictions.append(prediction_original)
                self.stochastic_configurations.append(configuration)
                self.prediction_all.append([target.denormalize([e.item()])[0] for e in predictions])
            #    self.stochastic_predictions_all.append(prediction_original_all)
                
                # if abs(prediction_avg - y_prime) < tolerance: break
                if abs(prediction_original- self.desired) < self.original_tolerance: break
                    
            best = np.argsort(abs(np.array(self.stochastic_predictions)-self.desired))[0]

            self.stochastic_best_position = self.stochastic_configurations[best]
            self.stochastic_best_prediction = self.stochastic_predictions[best]
            for i in range(len(self.stochastic_configurations)):
                print(self.stochastic_configurations[i], self.stochastic_predictions[i])
            
            return self.stochastic_configurations, self.stochastic_predictions, self.stochastic_best_position,self.stochastic_best_prediction, self.prediction_all


        def best_one(self, starting_point, escape = True) :
            y_prime = self.desired / (target.max[0] - target.min[0])
            tolerance = self.tolerance
            if starting_point is None or not isinstance(starting_point, (list, np.ndarray)):
                raise ValueError("Invalid starting_point input")

            x_i = [starting_point[i] / (feature.max[i] - feature.min[i]) for i in range(input_size)]  
            x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], 
                                   dtype = dtype)

            self.best_one_chosen = []
            self.best_one_predictions = []
            self.best_one_configurations = []
            self.prediction_all = []

            avoid = []
            memory = []
            memory_size = 5
            previous = None
            for step in range(self.steps):
                configuration = feature.denormalize(x_prime)
                predictions, gradients = self.predict_global(self.models, x = x_prime, y_prime = y_prime)
                prediction_avg = sum(predictions)/len(predictions) 
                gradient_avg = sum(gradients)/len(gradients)

                if escape :
                    candidates = [i for i in np.argsort(abs(gradient_avg))[::-1] if i not in avoid]
                else :
                    candidates = np.argsort(abs(gradient_avg))[::-1]
                    #[::-1]
                chosen = candidates[0]

                adjustment = list(np.repeat(0,len(self.unit_by_feature)))

                if gradient_avg[chosen] >= 0: adjustment[chosen] += self.unit_by_feature[chosen]
                else: adjustment[chosen] -= self.unit_by_feature[chosen]
                # adjustment 업데이트
                
                adjustment = np.array(adjustment)
                if prediction_avg is None or y_prime is None:
                    raise ValueError("Error: 'prediction_avg' or 'y_prime' is None. Model output needs validation.")

                print(f"Debug - Step {step}: prediction_avg={prediction_avg}, y_prime={y_prime}")

                # 이전 업데이트 수식
                if prediction_avg > y_prime: x_prime -= adjustment 
                elif prediction_avg < y_prime: x_prime += adjustment
                else: pass

                # update_delta = 0
                # if isinstance(self.models, nn.Module) or isinstance(self.models, list) and all(isinstance(m, nn.Module) for m in self.models):
                #     update_delta = -adjustment
                # else:
                #     if prediction_avg > y_prime:
                #         update_delta = -adjustment
                #     elif prediction_avg < y_prime:
                #         update_delta = adjustment
                #     else: pass
                # x_prime += update_delta

                # x_prime 조정
                # x_prime = super().bounding(x_prime)

                prediction_original = target.denormalize(prediction_avg)
                prediction_original = prediction_original[0]


                if configuration_show_steps and step % 10 == 0 and step != 0:
                    print(f"Step {step} Target : {self.desired}, Prediction : {prediction_original}")

                self.best_one_chosen.append(chosen)    
                self.best_one_predictions.append(prediction_original)
                self.best_one_configurations.append(configuration)
                self.prediction_all.append([target.denormalize([e.item()])[0] for e in predictions])

                if memory is None:
                    memory = []
                memory.append(prediction_original)
                if len(memory) > memory_size : memory = memory[len(memory)-memory_size:]
                if len(memory) == 5 and len(set(memory)) < 3  and previous == chosen: avoid.append(chosen)

                if abs(prediction_avg - y_prime) < tolerance: break
                if not isinstance(input_size, int) or input_size <= 0:
                    raise ValueError("input_size must be a positive integer")

                if escape and len(avoid) == input_size : break
                previous = chosen
            best = np.argsort(abs(np.array(self.best_one_predictions)-self.desired))[0]

            self.best_one_best_position = self.best_one_configurations[best]
            self.best_one_best_prediction = self.best_one_predictions[best]

            return self.best_one_configurations, self.best_one_predictions, self.best_one_best_position, self.best_one_best_prediction, self.prediction_all

    target = MinMaxScaling(data['Target'])
    feature = MinMaxScaling(data[[column for column in data.columns if column != 'Target']])
    print('scaling done')
    configurations, predictions, best_config, best_pred, pred_all = None, None, None, None, None
    if mode == 'global' :
        G = GlobalMode(desired = desired, models = models, modeling = modeling, strategy = strategy)
        if strategy == 'beam':
            configurations, predictions, best_config, best_pred, pred_all = G.beam(starting_point = starting_point,
                                                                 beam_width = beam_width)
            print('Global beam인식')
        elif strategy == 'stochastic':
            configurations, predictions, best_config, best_pred, pred_all = G.stochastic(starting_point = starting_point,
                                                                 num_candidates = num_candidates)
            print('Global stochastic인식')
        elif strategy == 'best_one':
            configurations, predictions, best_config, best_pred, pred_all = G.best_one(starting_point = starting_point, 
                                                                              escape = escape)
            print('Global bestone인식')
    elif mode == 'local':
        L = LocalMode(desired = desired, models = models, modeling = modeling, strategy = strategy)
        if strategy == 'exhaustive':
            configurations, predictions, best_config, best_pred, pred_all = L.exhaustive(starting_point = starting_point,
                                                                                alternative = alternative, top_k = top_k)
            print('Local eec인식')
        elif strategy == 'manual' :
            configurations, predictions, best_config, best_pred, pred_all = L.manual(starting_point = starting_point, index = index, up = up)
            print('Local manual')

    # if mode == 'global' and len(predictions) > 1:
    #     configurations = configurations[1:]
    #     predictions = predictions[1:]
    #     pred_all = pred_all[1:]
        
    # configurations = [
    #     [data_type[col](value) for col, value in enumerate(configurations[row])]
    #     for row in range(len(configurations))
    # ]
    # best_config = [data_type[i](c) for i, c in enumerate(best_config)]
    print(f"configurations type: {type(configurations)}")
    print(f"predictions type: {type(predictions)}")
    print(f"best_config type: {type(best_config)}")
    print(f"pred_all type: {type(pred_all)}")
    
    return configurations, predictions, best_config, best_pred, pred_all