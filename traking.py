# import sys
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import time  
# import random
# import xgboost
# import sklearn
# from itertools import product
# from itertools import chain
# import itertools

# def parameter_prediction(data, models, desired, starting_point, mode, modeling, strategy, tolerance, beam_width,
#         num_candidates, escape, top_k, index, up, alternative, unit, lower_bound, upper_bound, data_type, decimal_place):

#     configuration_patience = 10
#     configuration_patience_volume = 0.01
#     configuration_steps = 100 # step
#     configuration_eta = 10
#     configuration_eta_decay = 0.001
#     configuration_show_steps = True
#     configuration_show_focus = False
#     configuration_tolerance = tolerance
#     configuration_retrial_threshold = 10
#     start_from_standard = False
#     start_from_random = False
#     constrain_reselection = True

#     if_visualize = True

#     seed = 2025
#     random.seed(seed)
#     np.random.seed(seed)

#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)  

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
#     output_size = 1
#     input_size = data.values.shape[1] - output_size
#     print("The number of feature  : ",input_size)
#     dtype = torch.float32
    
#     def create_constraints(unit, lower_bound, upper_bound, data_type, decimal_place):
#         attribute_names = [
#             f"ATT{i + 1}" for i in range(len(unit))
#         ]

#         constraints = {}
#         for i, name in enumerate(attribute_names):
#         # float 혹은 int로 변환
#             u = float(unit[i])               # 단위
#             low = float(lower_bound[i])      # 하한
#             up = float(upper_bound[i])       # 상한
#             dt = data_type[i]                # str 또는 type(예: int, float)
#             dp = int(decimal_place[i])       # 소수점 자리수

#             constraints[name] = [u, low, up, dt, dp]
#         print(constraints)
#         return constraints    
#     if starting_point is None:
#         raise ValueError("starting_point cannot be None")
#     if desired is None:
#         raise ValueError("desired cannot be None")


#     constraints = create_constraints(unit = unit, 
#                                      lower_bound = lower_bound, 
#                                      upper_bound = upper_bound, 
#                                      data_type = data_type, 
#                                      decimal_place = decimal_place)    
    
    

#     class MinMaxScaling:
#         """
#         A class for normalizing and denormalizing data using Min-Max scaling.
#         """
#         def __init__(self, data):
#             """
#             Initializes the MinMaxScaling object.

#             Args:
#                 data (pd.DataFrame): Input data to be scaled.
#             """
#             self.max, self.min, self.range = [], [], []
#             self.data = pd.DataFrame()

#             # Reshape data if necessary
#             data = data.values.reshape(-1, 1) if len(data.values.shape) == 1 else data.values

#             epsilon = 2  # Small adjustment to avoid division by zero

#             for i in range(data.shape[1]):
#                 max_, min_ = max(data[:, i]), min(data[:, i])
#                 if max_ == min_:
#                     max_ *= epsilon

#                 self.max.append(max_)
#                 self.min.append(min_)
#                 self.range.append(max_ - min_)

#                 # Normalize the column and add to the DataFrame
#                 normalized_column = (data[:, i]) / (max_ - min_)
#                 self.data = pd.concat([self.data, pd.DataFrame(normalized_column)], axis=1)

#             # Convert normalized data to a torch tensor
#             self.data = torch.tensor(self.data.values, dtype=dtype)
#             print(self.data)

#         def denormalize(self, data):
#             """
#             Denormalizes data back to its original scale.

#             Args:
#                 data (torch.Tensor or np.ndarray): Normalized data to be converted.

#             Returns:
#                 list: Denormalized data.
#             """
#             # Convert torch tensor to numpy array if necessary
#             # data = data.detach().numpy() if isinstance(data, torch.Tensor) else data

#             # new_data = []
#             # for i, element in enumerate(data):
#             #     element = element * (self.max[i] - self.min[i])
#             #     element = round(element, np.array(list(constraints.values()))[:, 4][i])
#             #     new_data.append(element)
#             # torch.Tensor인 경우 numpy 배열로 변환
#             data = data.detach().numpy() if isinstance(data, torch.Tensor) else data

#             new_data = []
#             # constraints의 소수점 자리수를 미리 추출 (정수형으로 변환)
#             precisions = [int(v[4]) for v in list(constraints.values())]

#             for i, element in enumerate(data):
#                 # element가 numpy 배열인 경우 스칼라 값으로 변환
#                 scalar_val = float(element)
#                 # 원래 스케일로 복원
#                 scalar_val = (scalar_val) * (self.max[i] - self.min[i])   #+ self.min[i]
#                 # 반올림: precisions[i]는 정수여야 함
#                 scalar_val = round(scalar_val, precisions[i])
#                 new_data.append(scalar_val)
#             return new_data


#     class UNIVERSE :
#         def __init__(self, constraints = constraints):
#             self.constraints = constraints
#             self.unit_by_feature = [np.array(list(constraints.values()))[:,0][i] / (feature.max[i] - feature.min[i])
#                                     for i in range(input_size)]
#             self.upper_bounds = [np.array(list(constraints.values()))[:,2][i] / (feature.max[i] - feature.min[i])  
#                                  for i in range(input_size)]
#             self.lower_bounds = [np.array(list(constraints.values()))[:,1][i] / (feature.max[i] - feature.min[i])  
#                                  for i in range(input_size)]
            
#             # 디버그 출력
#             print("=== 각 피처에 대한 scaling factor (unit_by_feature) ===")
#             for i in range(input_size):
#                 print(f"Feature {i+1}: unit_by_feature = {self.unit_by_feature[i]}, "
#                     f"upper_bound = {self.upper_bounds[i]}, lower_bound = {self.lower_bounds[i]}")
#             self.raw_unit = np.array(list(constraints.values()))[:,0].tolist()
#             self.raw_lower = np.array(list(constraints.values()))[:,1].tolist()
#             self.raw_upper = np.array(list(constraints.values()))[:,2].tolist()
            

#         def predict(self, models, x, y_prime, modeling, fake_gradient = True):
#             predictions,gradients = [], []
#             copy = x.clone()
#             for i, model in enumerate(models):
#                 x = x.clone().detach().requires_grad_(True)
#                 if isinstance(model, nn.Module):
#                     prediction = model(x)
#                     loss = abs(prediction - y_prime)
#                     loss.backward()
#                     gradient = x.grad.detach().numpy()
#                     prediction = prediction.detach().numpy()

#                 else: # ML
#                     x = x.detach().numpy().reshape(1,-1)
#                     prediction = model.predict(x)
#                     if fake_gradient :
#                         gradient = []
#                         for j in range(x.shape[1]):
#                             new_x = x.copy()
#                             new_x[0,j] += self.unit_by_feature[j]
#                             new_prediction = model.predict(new_x)
#                             slope = (new_prediction - prediction) / self.unit_by_feature[j]
#                             gradient.append(slope)
#                         gradient = np.array(gradient).reshape(-1)   
#                     else : gradient = np.repeat(0, x.shape[1])
#                 x = copy.clone()
#                 predictions.append(prediction)
#                 gradients.append(gradient)

#             if modeling.lower() == 'single': return predictions[0], gradients[0], predictions
#             elif modeling == 'ensemble': return sum(predictions)/len(predictions), sum(gradients)/len(gradients), predictions  
#             else: raise Exception(f"[modeling error] there is no {modeling}.")


#         def bounding(self, configuration):
#             new = []
#             for k, element in enumerate(configuration):
#                 element = element - (element % self.unit_by_feature[k])
#                 # print(f"element % self.unit_by_feature[k] = {element % self.unit_by_feature[k]}")
#                 if element >= self.upper_bounds[k]:
#                     # print("경계 상한 도달")
#                     element = self.upper_bounds[k]
#                 elif element <= self.lower_bounds[k]:
#                     # print("경계 하한 도달")
#                     element = self.lower_bounds[k]
#                 else: pass
#                 new.append(element)        
#             configuration = torch.tensor(new, dtype = dtype)        
#             return configuration


#         def truncate(self, configuration):
#             new_configuration = []
#             for i, value in enumerate(configuration) :
#                 value = value - (value % self.raw_unit[i])
#                 value = value if value >= self.raw_lower[i] else self.raw_lower[i]
#                 value = value if value <= self.raw_upper[i] else self.raw_upper[i]
#                 new_configuration.append(value)
#            # configuration = torch.tensor(new_configuration, dtype = dtype)     
#             return configuration
        
#     class LocalMode(UNIVERSE):
#         def __init__(self, desired, models, modeling, strategy):
#             super().__init__()
#             self.desired = desired
#             self.y_prime = self.desired / (target.max[0] - target.min[0])
#             self.models = []
#             self.modeling = modeling
#             self.strategy = strategy
#             for model in models : 
#                 if isinstance(model, nn.Module) : model.eval()
#                 self.models.append(model)

#         def exhaustive(self, starting_point, top_k = 5, alternative = 'keep_move'):
#             self.starting_point = super().truncate(starting_point)
#             self.starting_point = np.array([self.starting_point[i] / (feature.max[i] - feature.min[i]) 
#                                             for i in range(input_size)])
#             self.top_k = top_k
#             self.recorder = []

#             self.search_space, self.counter = [],[]
#             if alternative == 'keep_up_down' :
#                 variables = [[0, 1, 2]] * input_size
#             else :
#                 variables = [[0, 1]] * input_size
#             self.all_combinations = list(product(*variables))
#             self.adj = []
#             print(torch.tensor(self.starting_point, dtype = dtype))
#             if alternative == 'keep_move' or alternative == 'keep_up_down':
#                 prediction_km, gradient_km, p_all = super().predict(self.models,torch.tensor(self.starting_point, dtype = dtype), 
#                                                self.y_prime, self.modeling, fake_gradient = True)
#                 prediction_km = prediction_km[0] if isinstance(prediction_km,list) else prediction_km
#             for combination in self.all_combinations :
#                 count = combination.count(1) if alternative == 'up_down' else combination.count(0)
#                 adjustment = np.repeat(0,len(self.unit_by_feature)).tolist()
#                 for i, boolean in enumerate(list(combination)):
#                     if alternative == 'up_down':
#                         if boolean == 1 :   adjustment[i] = adjustment[i] + self.unit_by_feature[i]
#                         elif boolean == 0 : adjustment[i] = adjustment[i] - self.unit_by_feature[i]
#                         else: raise Exception("ERROR")

#                     elif alternative == 'keep_move':
#                         if prediction_km > self.y_prime :                        
#                             if boolean == 1 :   
#                                 if gradient_km[i] >= 0 :
#                                     adjustment[i] = adjustment[i] - self.unit_by_feature[i]
#                                 else:
#                                     adjustment[i] = adjustment[i] + self.unit_by_feature[i]
#                             elif boolean == 0 : pass
#                             else: raise Exception("ERROR")            

#                         else: 
#                             if boolean == 1 :   
#                                 if gradient_km[i] >= 0 :
#                                     adjustment[i] = adjustment[i] + self.unit_by_feature[i]
#                                 else:
#                                     adjustment[i] = adjustment[i] - self.unit_by_feature[i]
#                             elif boolean == 0 : pass
#                             else: raise Exception("ERROR")    

#                     elif alternative == 'keep_up_down' :
#                         important_features = np.argsort(abs(gradient_km))[::-1][:self.top_k]
#                         if i in important_features :
#                             if boolean == 2 :   adjustment[i] = adjustment[i] + self.unit_by_feature[i]
#                             elif boolean == 1 : adjustment[i] = adjustment[i] - self.unit_by_feature[i]
#                             elif boolean == 0 : pass
#                             else: raise Exception("ERROR")   

#                 self.adj.append(adjustment)
#                 candidate = self.starting_point + adjustment         
#                 candidate = super().bounding(candidate)
#                 if str(candidate) not in self.recorder : 
#                     self.search_space.append(candidate)
#                     self.counter.append(count)
#                     self.recorder.append(str(candidate))
#         #    print(len(self.search_space))
#             self.predictions = []
#             self.configurations = []
#             self.pred_all = []
#             for candidate in self.search_space :
#                 prediction, _, p_all = super().predict(self.models,candidate, self.y_prime, self.modeling, fake_gradient = False)
#                 prediction = target.denormalize(prediction)[0]
#                 configuration = feature.denormalize(candidate)
#                 self.predictions.append(prediction)
#                 self.configurations.append(configuration)
#                 self.pred_all.append([target.denormalize([e.item()])[0] for e in p_all])
                

#             self.table = pd.DataFrame({'configurations':self.configurations,'find_dup' : self.configurations,
#                                       'predictions':self.predictions,'difference' : np.array(abs(np.array(self.predictions)-self.desired)).tolist(),
#                                       'counter':self.counter})
#             self.table['find_dup'] = self.table['find_dup'].apply(lambda x: str(x))
#             self.table = self.table[~self.table.duplicated(subset='find_dup', keep='first')]
#             self.table = self.table.drop(columns=['find_dup'])

#             self.table = self.table.sort_values(by='counter', ascending=False).sort_values(by='difference', ascending=True)
#             self.configurations = self.table['configurations']
#             self.predictions = self.table['predictions']
#             self.difference = self.table['difference']
#             self.counter = self.table['counter']
            
#             configurations = self.configurations[:].values.tolist()
#             predictions = self.predictions[:].values.tolist()
#             best_config = configurations[0]
#             best_pred = predictions[0]

#             try: return configurations, predictions, best_config, best_pred, self.pred_all
#             except : return self.configurations[:].values.tolist(), self.predictions[:].values.tolist(), best_config, best_pred, self.pred_all

#         def manual(self, starting_point, index=0, up=True):
#             self.starting_point = super().truncate(starting_point)
#             self.starting_point = np.array([self.starting_point[i] / (feature.max[i] - feature.min[i]) 
#                                             for i in range(input_size)])

#             adjustment = np.repeat(0,len(self.unit_by_feature)).tolist()
#             if up : adjustment[index] += self.unit_by_feature[index]
#             else : adjustment[index] -= self.unit_by_feature[index]
#             position = self.starting_point + adjustment         
#             position = super().bounding(position)        
#             prediction, _, p_all = super().predict(self.models,position, self.y_prime, self.modeling)
#             prediction = target.denormalize(prediction)
#             configuration = feature.denormalize(position)
#             p_all = [target.denormalize([e.item()])[0] for e in p_all]
#             return [configuration], prediction, configuration, prediction, p_all 
        
#     class GlobalMode(UNIVERSE):
#         def __init__(self, desired, models, modeling, strategy, tolerance = configuration_tolerance, steps = configuration_steps):
#             super().__init__()
#             self.desired = desired
#             self.y_prime = self.desired / (target.max[0] - target.min[0])
#             self.models = []
#             self.modeling = modeling
#             self.strategy = strategy
#             self.tolerance = tolerance / (target.max[0] - target.min[0])
#             self.original_tolerance = tolerance
#             self.steps = steps
#             for model in models : 
#                 if isinstance(model, nn.Module): model.train()
#                 self.models.append(model)

#         def predict_global(
#                 self, 
#                 models, 
#                 x, 
#                 y_prime, 
#                 method = "fd", 
#                 alpha=1, 
#                 p = 0.1,
#                 m = 5
#                 ):
            
#             predictions,gradients = [], []
#             copy = x.clone()
#             for i, model in enumerate(models):
#                 x = torch.tensor(x, dtype=torch.float32)
#                 x = x.clone().detach().requires_grad_(True)
#                 # 변형은 여기부터
#                 if isinstance(model, nn.Module):
#                     prediction = model(x)
#                     # loss = prediction - y_prime # 기존 방식
#                     loss = abs(prediction - y_prime) # MAE
#                     # loss = (prediction - y_prime) ** 2 # MSE
#                     loss.backward()
#                     gradient = x.grad.detach().numpy()
#                     prediction = prediction.detach().numpy()
                
#                 else:
                    
#                     x = x.detach().numpy().reshape(1,-1)  # x : ndarray
                    
#                     y = model.predict(x)

#                     # prediction = model(x) # 모델의 출력 계산
#                     # y = model(x)
                
#                     prediction = y
#                     assert m >= 1  and m <= len(self.unit_by_feature)
#                     sampled_indedices = np.random.choice(
#                         np.arange(len(self.unit_by_feature),dtype = int),replace = False, size = m).tolist()
#                     _unit_by_feature = np.array(self.unit_by_feature)[sampled_indedices].tolist()
                    
#                     if method == "fd":
                    
#                         best_g = [0] * len(self.unit_by_feature)
#                         # best_g를 리스트 대신 텐서로 초기화합니다
#                     elif method == "target":
#                         best_g = abs(y_prime - y)
#                         best_change = [0] * len(self.unit_by_feature)
#                     else:
#                         raise ValueError("Expected 'fd' or 'target', but got {}".format(method))

#                     # for change in itertools.product([-1, 0, 1], repeat=len(self.unit_by_feature)):
                    
#                     for change in itertools.product([-1, 0, 1], repeat=len(_unit_by_feature)):
#                         '''
#                         x : ndarray
#                         change : Tuple
#                         unit_by_feature : List
#                         '''
                        
#                         # xx = x + np.array([xx2 * xx3 for xx2, xx3 in zip(change, self.unit_by_feature)])
#                         if not np.random.uniform() < p:
#                             continue
#                         # 이거 아래가 정답
#                         xx = x.copy()
#                         # xx = x + alpha * np.array([xx2 * xx3 for xx2, xx3 in zip(change, self.unit_by_feature)])
#                         # xx[0] = xx[0] + alpha * np.array([xx2 * xx3 for xx2, xx3 in zip(change, self.unit_by_feature)])
                        
#                         xx[0][sampled_indedices] = xx[0][sampled_indedices] + alpha * np.array([xx2 * xx3 \
#                             for xx2, xx3 in zip(change, _unit_by_feature)])
#                         xx[0] = np.clip(xx[0], a_min = self.lower_bounds, a_max = self.upper_bounds)
                        
#                         # 이거는 DL 쓸 때
#                         # delta_tensor = torch.tensor([xx2 * xx3 for xx2, xx3 in zip(change, self.unit_by_feature)],
#                         #         dtype=x.dtype, device=x.device)
#                         # xx = x + alpha * delta_tensor
                        
#                         # yy = model.predict(xx)
#                         # xx = torch.tensor(xx)
#                         # ML 적용
#                         yy = model.predict(xx)
#                         # DL 적용
#                         # yy = model(xx)
                        
#                         # print("yy {}".format(yy))
#                         if method == "fd":  # finite difference-based
                        
#                             l = abs(y_prime - y)
#                             ll = abs(y_prime - yy)
#                             # g = (yy - y) / np.array(unit_by_feature)
#                             # 아래 3개만 해제
                        
#                             g = (ll - l) / np.array(self.unit_by_feature)
                            
#                             g_norm = np.linalg.norm(g)
                            
#                             best_g_norm = np.linalg.norm(best_g)
                            

#                             # 기준
#                             # unit_tensor = torch.tensor(self.unit_by_feature, dtype=ll.dtype, device=ll.device)
                            
#                             # g = (ll - l) / unit_tensor
#                             # g_np = g.detach().numpy()  # 여기서 detach()를 호출!
#                             # g_norm = np.linalg.norm(g_np)
#                             # best_g_norm = np.linalg.norm(best_g.detach().numpy())  # 여기서 detach()를 호출!
#                             # DL 일 때는 위에 5개 지우고
                            
                            
#                             if best_g_norm < g_norm:
                            
#                                 best_g = g

#                         elif method == "target":  # target based
#                             g = abs(y_prime - yy)  # |target - pred|
#                             # print(f'g:{g}')
#                             if best_g > g:
#                                 best_g = g
#                                 # best_change = change
#                                 if m is not None:
#                                     best_change = [0] * len(self.unit_by_feature)
#                                     best_change = np.array(best_change)
#                                     best_change[sampled_indedices] = change
#                                     best_change = best_change.tolist()
#                                 else:
#                                     best_g = g
                            
                            
#                         else:
#                             raise ValueError("Expected 'fd' or 'target', but got {}".format(method))
#                     # 여기까지 들여쓰기
#                     # ML    
#                         if method == "fd":
                            
#                             gradient = np.array(best_g)
#                         elif method == "target":
#                             gradient = np.array(best_change)
#                     # DL
#                     # if method == "fd":
#                     #     gradient = g.detach().numpy()  # best_g가 텐서라면 detach() 후 numpy() 호출
#                     # elif method == "target":
#                     #     gradient = np.array(best_change)  # best_change는 파이썬 리스트이므로 괜찮음
#                     # else:
#                     #     raise ValueError
            
#                 x = copy.clone()
                
#                 predictions.append(prediction)
#                 gradients.append(gradient)
                
                
#             return predictions, gradients

#         def beam(self, beam_width, starting_point) :
#             self.beam_width = beam_width
#             y_prime = self.desired / (target.max[0] - target.min[0])
#             tolerance = self.tolerance
#             final = None

#             x_i = [starting_point[i] / (feature.max[i] - feature.min[i]) for i in range(input_size)]  
#             x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], 
#                                    dtype = dtype)

#             self.beam_positions, self.beam_targets = [], []
#             self.beam_positions_denorm, self.beam_targets_denorm = [], []

#             self.beam_history = []
#             self.previous_gradient = [[], [], [], [], []]
#             self.prediction_all = []

#             success = [False]
#             which = []
#             close = False
#             close_margin = 10
#             for step in range(self.steps):
#                 if len(self.beam_positions) == 0 :
#                     current_positions = [x_prime.clone().detach()]
#                     for j in range(self.beam_width - 1):
#                         random_offsets = torch.tensor(np.random.uniform(-0.5, 0.5, input_size), dtype=x_prime.dtype)
#                         current_positions += [x_prime.clone().detach() + random_offsets]
#                 else :
#                     current_positions = self.beam_positions[-1]

#                 configurations = []
#                 candidates = []
#                 candidates_score = []
#                 beam_predictions = []
#                 beams = []
#                 p_all = []
#                 for p, current_pos in enumerate(current_positions):
#                     configuration = feature.denormalize(current_pos.clone().detach())
#                     configurations.append(configuration)

#                     predictions, gradients = self.predict_global(self.models, x = current_pos, y_prime = y_prime)                
#                     prediction_avg = sum(predictions)/len(predictions) ###
#                     gradient_avg = sum(gradients)/len(gradients)  

#                     prediction_original = prediction_avg * (target.max[0] - target.min[0])
#                     prediction_original = prediction_original[0]
#                     beam_predictions.append(prediction_original)     
#                     p_all.append([target.denormalize([e.item()])[0] for e in predictions])
#                     if abs(prediction_original - self.desired) < close_margin : close = True
#                     else : close = False
#                 #    print(close)
#                     if abs(prediction_avg - y_prime) < tolerance:
#                         best_config = configuration
#                         best_pred = prediction_original
#                         success.append(True)

#                #     if close :
#                #         order = np.argsort(abs(gradient_avg))
#                #     else :
#                #         order = np.argsort(abs(gradient_avg))[::-1]
#                #     order = np.random.permutation(np.argsort(abs(gradient_avg))[::-1])
#                     order = np.argsort(abs(gradient_avg))[::-1]

#                     beam = order[:self.beam_width]
                    
#                     for b in beam:

#                         adjustment = list(np.repeat(0,len(self.unit_by_feature)))
#                         if gradient_avg[b] >= 0 :
#                             adjustment[b] += self.unit_by_feature[b]
#                         else:
#                             adjustment[b] -= self.unit_by_feature[b]    


#                         adjustment = np.array(adjustment)
#                         if prediction_avg > y_prime : 
#                             position = current_pos.clone().detach() - adjustment 
#                         else :
#                             position = current_pos.clone().detach() + adjustment 
#                         # update_delta를 한 번만 계산해서 적용
#                         # if isinstance(self.models, nn.Module) or (isinstance(self.models, list) and all(isinstance(m, nn.Module) for m in self.models)):
#                         #     update_delta = -adjustment
#                         # else:
#                         #     if prediction_avg > y_prime:
#                         #         update_delta = -adjustment
#                         #     elif prediction_avg < y_prime:
#                         #         update_delta = adjustment
#                         #     else:
#                         #         update_delta = np.zeros_like(adjustment)
                        
#                         # position = current_pos.clone().detach() + update_delta
#                         position = super().bounding(position)
#                         candidates.append(position)
#                         candidates_score.append(abs(gradient_avg[b]))

#                 if step % 10 == 0 and step != 0 : print(f"Step {step} Target : {self.desired}, Prediction : {beam_predictions}")
#                 select = np.argsort(candidates_score)[::-1][:self.beam_width]
#                 # 오류 메시지 제거
#                 # new_positions = [torch.tensor(candidates[s], dtype = dtype) for s in select]
#                 new_positions = [candidates[s].clone().detach().to(dtype) for s in select]

#                 if len(beam_predictions) == 1 : beam_predictions = list(np.repeat(beam_predictions[0],self.beam_width))
#                 self.beam_positions.append(new_positions)
#                 self.beam_targets.append(beam_predictions)
#                 self.beam_history.append(beam.tolist())
#                 self.beam_positions_denorm.append(configurations) 
#                 self.beam_targets_denorm.append(beam_predictions)
#                 self.prediction_all.append(p_all)
#                 if any(success): break      

#             flattened_positions = list(chain.from_iterable(self.beam_positions_denorm))
#             flattened_predictions = list(chain.from_iterable(self.beam_targets_denorm))
#             best = int(np.argsort(abs(np.array(flattened_predictions)-self.desired))[0])

#             self.best_position = flattened_positions[best]
#             self.best_prediction = flattened_predictions[best]

#             return self.beam_positions_denorm, self.beam_targets_denorm, self.best_position, self.best_prediction, self.prediction_all


        # def stochastic(self, num_candidates = 5, starting_point = starting_point) :
        #     self.num_candidates = num_candidates
        #     y_prime = self.desired / (target.max[0] - target.min[0])
            
        #     tolerance = self.tolerance
        #     final = None
            
        #     x_i = [float(starting_point[i]) / (feature.max[i] - feature.min[i]) for i in range(input_size)]
        #     x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], dtype=dtype)
            
        #     self.stochastic_chosen = []
        #     self.stochastic_predictions = []
        #     self.stochastic_configurations = []
        #     self.stochastic_predictions_all = []
        #     self.prediction_all = []
        #     method = "fd" # 변경사항
        #     for step in range(self.steps):
                
        #         configuration = feature.denormalize(x_prime)
        #         predictions, gradients = self.predict_global(self.models, x = x_prime, y_prime = y_prime)
                
        #         if isinstance(self.models, nn.Module) or isinstance(self.models, list) and all(isinstance(m, nn.Module) for m in self.models):
        #             gradient_avg = sum(gradients)/len(gradients)
        #             prediction_avg = sum(predictions)/len(predictions) 
        #             gradient_avg = sum(gradients)/len(gradients) # List of ndarray
        #             candidates = np.argsort(abs(gradient_avg))[::-1][:self.num_candidates] #[::-1]
                    
        #             chosen = random.choice(candidates)
                    
        #             adjustment = list(np.repeat(0,len(self.unit_by_feature)))
                    
        #             # adjustment 업데이트
        #             if gradient_avg[chosen] >= 0:
        #                 adjustment[chosen] += self.unit_by_feature[chosen]
        #             else:
        #                 adjustment[chosen] -= self.unit_by_feature[chosen]
        #             adjustment = np.array(adjustment)
        #             update_delta = -adjustment
        #             x_prime += update_delta
                    
        #         else:
        #             # 여기부터
                    
        #             if not method == "target":
                        
        #                 # gradients : (List of ndarray)
        #                 # List => The number of list
                        
        #                 best_g = [0] * len(self.unit_by_feature)
        #                 for i in range(len(gradients)):
                            
        #                     g_norm = np.linalg.norm(gradients[i])
                            
        #                     best_g_norm = np.linalg.norm(best_g)
        #                     if best_g_norm < g_norm:
                                
        #                         best_g = gradients[i]
        #                 sum_gradients = []
        #                 for i in range(len(self.unit_by_feature)):
        #                     if best_g.ndim == 2:
        #                         best_g = best_g[0]
                            
        #                     if best_g[i] > 0:
        #                         sum_gradients.append(1)
        #                     elif best_g[i] < 0:
        #                         sum_gradients.append(-1)
        #                     else:
        #                         sum_gradients.append(0)
        #                 sum_gradients = np.array(sum_gradients)
                        
                        
        #                 # x_prime = x_prime - sum_gradients * self.unit_by_feature # 이거 한줄이면 될거 아래 2줄 추가됌

        #                 # 수정: torch tensor로 변환하면서, x_prime과 같은 dtype, device로 맞추기
        #                 sum_gradients_tensor = torch.tensor(sum_gradients, dtype=x_prime.dtype, device=x_prime.device)

        #                 # self.unit_by_feature가 리스트라면, torch tensor로 변환
        #                 unit_by_feature_tensor = torch.tensor(self.unit_by_feature, dtype=x_prime.dtype, device=x_prime.device)

        #                 # 이후 연산에서 모두 torch tensor를 사용
        #                 x_prime = x_prime - sum_gradients_tensor * unit_by_feature_tensor
        #                 print(f'x_prime : {x_prime}')
        #             # gradients : (List of ndarray)
        #             else:
        #             # gradients : (List of ndarray)|
        #             # (-1, 0, 1, .., 0), .., (1, 1, 1, .., 1) : # models
        #             # 앙상블 방향 계산
        #             # 각 모델별 방향 정보를 성분별로 더함 # 변수별 계산된 방향과 변수별 단위 곱하고 그걸 각 변수에 더하면 됨 # TODO : NN 모델 + DL 모델 앙상블인 경우 NN 모델에 대한 처리 필요
        #                 # print(f'gradients : {gradients}')
        #                 # print(f'unit_by_feature : {self.unit_by_feature}')
        #                 # print(f'x_prime : {x_prime}')
                        
                        
        #                 sum_gradients = sum(gradients) # List
        #                 # x_prime = x_prime + sum_gradients * self.unit_by_feature
                        
        #                 # 수정: torch tensor로 변환하면서, x_prime과 같은 dtype, device로 맞추기
        #                 sum_gradients_tensor = torch.tensor(sum_gradients, dtype=x_prime.dtype, device=x_prime.device)
                        
        #                 # self.unit_by_feature가 리스트라면, torch tensor로 변환
        #                 unit_by_feature_tensor = torch.tensor(self.unit_by_feature, dtype=x_prime.dtype, device=x_prime.device)
                        
        #                 # 이후 연산에서 모두 torch tensor를 사용
        #                 x_prime = x_prime + sum_gradients_tensor * unit_by_feature_tensor
        #                 print(f'x_prime : {x_prime}')

                    
                    


        #         prediction_avg = sum(predictions)/len(predictions)
        #         # 여기까지
        #         # gradient_avg = sum(gradients)/len(gradients)

        #         # prediction_avg = sum(predictions)/len(predictions) 
                
        #         # gradient_avg = sum(gradients)/len(gradients) # List of ndarray
                
        #         # candidates = np.argsort(abs(gradient_avg))[::-1][:self.num_candidates] #[::-1]
                
        #         # chosen = random.choice(candidates)
                
        #         # adjustment = list(np.repeat(0,len(self.unit_by_feature)))
                
        #         # # adjustment 업데이트
        #         # if gradient_avg[chosen] >= 0:
        #         #     adjustment[chosen] += self.unit_by_feature[chosen]
        #         # else:
        #         #     adjustment[chosen] -= self.unit_by_feature[chosen]
        #         # adjustment = np.array(adjustment)

        #         # update_delta = 0
        #         # # if isinstance(self.models, nn.Module) or isinstance(self.models, list) and all(isinstance(m, nn.Module) for m in self.models):
        #         # #     update_delta = -adjustment
        #         # # else:
        #         # #     if prediction_avg > y_prime:
        #         # #         update_delta = -adjustment
        #         # #     elif prediction_avg < y_prime:
        #         # #         update_delta = adjustment
        #         # #     else: pass
        #         # # x_prime += update_delta
                
        #         # x_prime += adjustment

        #         # x_prime 조정
        #         # if prediction_avg > y_prime:
        #         #     x_prime -= adjustment
        #         # elif prediction_avg < y_prime:
        #         #     x_prime += adjustment
        #         # else: pass
        #         # version
                
        #         # x_prime = super().bounding(x_prime)
        #         x_prime = np. clip(x_prime, a_min=self.lower_bounds, a_max=self.upper_bounds)
        #         # print(f'self.unit_by_feature type : {type(self.unit_by_feature)}')
        #         prediction_original = target.denormalize(prediction_avg)
        #         if prediction_original is None:
        #             raise ValueError(f"Step {step}: prediction_original is None")
        #         prediction_original = prediction_original[0]
                
        #      #   prediction_original_all = target.denormalize(predictions_all)
        #      #   prediction_original_all = prediction_original_all
                
        #         if configuration_show_steps and step % 10 == 0 and step != 0:
        #             print(f"Step {step} Target : {self.desired}, Prediction : {prediction_original}")

        #         # self.stochastic_chosen.append(chosen)    
        #         self.stochastic_predictions.append(prediction_original)
        #         self.stochastic_configurations.append(configuration)
        #         self.prediction_all.append([target.denormalize([e.item()])[0] for e in predictions])
        #     #    self.stochastic_predictions_all.append(prediction_original_all)
                
        #         # if abs(prediction_avg - y_prime) < tolerance: break
        #         if abs(prediction_original- self.desired) < self.original_tolerance: break
                    
        #     best = np.argsort(abs(np.array(self.stochastic_predictions)-self.desired))[0]

        #     self.stochastic_best_position = self.stochastic_configurations[best]
        #     self.stochastic_best_prediction = self.stochastic_predictions[best]
        #     for i in range(len(self.stochastic_configurations)):
        #         print(self.stochastic_configurations[i], self.stochastic_predictions[i])
            
        #     return self.stochastic_configurations, self.stochastic_predictions, self.stochastic_best_position,self.stochastic_best_prediction, self.prediction_all


#         def best_one(self, starting_point, escape = True) :
#             y_prime = self.desired / (target.max[0] - target.min[0])
#             tolerance = self.tolerance
#             if starting_point is None or not isinstance(starting_point, (list, np.ndarray)):
#                 raise ValueError("Invalid starting_point input")

#             x_i = [starting_point[i] / (feature.max[i] - feature.min[i]) for i in range(input_size)]  
#             x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], 
#                                    dtype = dtype)

#             self.best_one_chosen = []
#             self.best_one_predictions = []
#             self.best_one_configurations = []
#             self.prediction_all = []

#             avoid = []
#             memory = []
#             memory_size = 5
#             previous = None
#             for step in range(self.steps):
#                 configuration = feature.denormalize(x_prime)
#                 predictions, gradients = self.predict_global(self.models, x = x_prime, y_prime = y_prime)
#                 prediction_avg = sum(predictions)/len(predictions) 
#                 gradient_avg = sum(gradients)/len(gradients)

#                 if escape :
#                     candidates = [i for i in np.argsort(abs(gradient_avg))[::-1] if i not in avoid]
#                 else :
#                     candidates = np.argsort(abs(gradient_avg))[::-1]
#                     #[::-1]
#                 chosen = candidates[0]

#                 adjustment = list(np.repeat(0,len(self.unit_by_feature)))

#                 if gradient_avg[chosen] >= 0: adjustment[chosen] += self.unit_by_feature[chosen]
#                 else: adjustment[chosen] -= self.unit_by_feature[chosen]
#                 # adjustment 업데이트
                
#                 adjustment = np.array(adjustment)
#                 if prediction_avg is None or y_prime is None:
#                     raise ValueError("Error: 'prediction_avg' or 'y_prime' is None. Model output needs validation.")

#                 print(f"Debug - Step {step}: prediction_avg={prediction_avg}, y_prime={y_prime}")

#                 # 이전 업데이트 수식
#                 if prediction_avg > y_prime: x_prime -= adjustment 
#                 elif prediction_avg < y_prime: x_prime += adjustment
#                 else: pass

#                 # update_delta = 0
#                 # if isinstance(self.models, nn.Module) or isinstance(self.models, list) and all(isinstance(m, nn.Module) for m in self.models):
#                 #     update_delta = -adjustment
#                 # else:
#                 #     if prediction_avg > y_prime:
#                 #         update_delta = -adjustment
#                 #     elif prediction_avg < y_prime:
#                 #         update_delta = adjustment
#                 #     else: pass
#                 # x_prime += update_delta

#                 # x_prime 조정
#                 # x_prime = super().bounding(x_prime)

#                 prediction_original = target.denormalize(prediction_avg)
#                 prediction_original = prediction_original[0]


#                 if configuration_show_steps and step % 10 == 0 and step != 0:
#                     print(f"Step {step} Target : {self.desired}, Prediction : {prediction_original}")

#                 self.best_one_chosen.append(chosen)    
#                 self.best_one_predictions.append(prediction_original)
#                 self.best_one_configurations.append(configuration)
#                 self.prediction_all.append([target.denormalize([e.item()])[0] for e in predictions])

#                 if memory is None:
#                     memory = []
#                 memory.append(prediction_original)
#                 if len(memory) > memory_size : memory = memory[len(memory)-memory_size:]
#                 if len(memory) == 5 and len(set(memory)) < 3  and previous == chosen: avoid.append(chosen)

#                 if abs(prediction_avg - y_prime) < tolerance: break
#                 if not isinstance(input_size, int) or input_size <= 0:
#                     raise ValueError("input_size must be a positive integer")

#                 if escape and len(avoid) == input_size : break
#                 previous = chosen
#             best = np.argsort(abs(np.array(self.best_one_predictions)-self.desired))[0]

#             self.best_one_best_position = self.best_one_configurations[best]
#             self.best_one_best_prediction = self.best_one_predictions[best]

#             return self.best_one_configurations, self.best_one_predictions, self.best_one_best_position, self.best_one_best_prediction, self.prediction_all

#     target = MinMaxScaling(data['Target'])
#     feature = MinMaxScaling(data[[column for column in data.columns if column != 'Target']])
#     print('scaling done')
#     configurations, predictions, best_config, best_pred, pred_all = None, None, None, None, None
#     if mode == 'global' :
#         G = GlobalMode(desired = desired, models = models, modeling = modeling, strategy = strategy)
#         if strategy == 'beam':
#             configurations, predictions, best_config, best_pred, pred_all = G.beam(starting_point = starting_point,
#                                                                  beam_width = beam_width)
#             print('Global beam인식')
#         elif strategy == 'stochastic':
#             configurations, predictions, best_config, best_pred, pred_all = G.stochastic(starting_point = starting_point,
#                                                                  num_candidates = num_candidates)
#             print('Global stochastic인식')
#         elif strategy == 'best_one':
#             configurations, predictions, best_config, best_pred, pred_all = G.best_one(starting_point = starting_point, 
#                                                                               escape = escape)
#             print('Global bestone인식')
#     elif mode == 'local':
#         L = LocalMode(desired = desired, models = models, modeling = modeling, strategy = strategy)
#         if strategy == 'exhaustive':
#             configurations, predictions, best_config, best_pred, pred_all = L.exhaustive(starting_point = starting_point,
#                                                                                 alternative = alternative, top_k = top_k)
#             print('Local eec인식')
#         elif strategy == 'manual' :
#             configurations, predictions, best_config, best_pred, pred_all = L.manual(starting_point = starting_point, index = index, up = up)
#             print('Local manual')

#     # if mode == 'global' and len(predictions) > 1:
#     #     configurations = configurations[1:]
#     #     predictions = predictions[1:]
#     #     pred_all = pred_all[1:]
        
#     # configurations = [
#     #     [data_type[col](value) for col, value in enumerate(configurations[row])]
#     #     for row in range(len(configurations))
#     # ]
#     # best_config = [data_type[i](c) for i, c in enumerate(best_config)]
#     print(f"configurations type: {type(configurations)}")
#     print(f"predictions type: {type(predictions)}")
#     print(f"best_config type: {type(best_config)}")
#     print(f"pred_all type: {type(pred_all)}")
    
#     return configurations, predictions, best_config, best_pred, pred_all
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
from itertools import product, chain
import itertools

# -------------------------------
# 전역 설정 및 유틸리티 함수
# -------------------------------
DEFAULT_SEED = 2025
DTYPE = torch.float32

def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_constraints(unit, lower_bound, upper_bound, data_type, decimal_place):
    """
    각 피처에 대한 단위, 하한, 상한, 데이터 타입, 소수점 자리수를 제약조건 사전으로 생성합니다.
    """
    attribute_names = [f"ATT{i+1}" for i in range(len(unit))]
    constraints = {}
    for i, name in enumerate(attribute_names):
        u = float(unit[i])
        low = float(lower_bound[i])
        up = float(upper_bound[i])
        dt = data_type[i]
        dp = int(decimal_place[i])
        constraints[name] = [u, low, up, dt, dp]
    print("Constraints:", constraints)
    return constraints

# -------------------------------
# 데이터 스케일링 클래스
# -------------------------------
class MinMaxScaling:
    """
    Min-Max scaling을 이용하여 데이터를 정규화 및 역정규화하는 클래스입니다.
    """
    def __init__(self, data, dtype=DTYPE):
        self.dtype = dtype
        self.max = []
        self.min = []
        self.range = []
        # DataFrame 또는 Series일 경우 numpy array로 변환
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        self.scaled_data = pd.DataFrame()
        epsilon = 2  # 최대, 최소가 같은 경우를 위한 보정
        for i in range(data.shape[1]):
            col = data[:, i]
            max_val = np.max(col)
            min_val = np.min(col)
            if max_val == min_val:
                max_val *= epsilon
            self.max.append(max_val)
            self.min.append(min_val)
            self.range.append(max_val - min_val)
            normalized_col = col / (max_val - min_val)
            self.scaled_data = pd.concat([self.scaled_data, pd.DataFrame(normalized_col)], axis=1)
        self.tensor_data = torch.tensor(self.scaled_data.values, dtype=self.dtype)
        print("Normalized Data Tensor:", self.tensor_data)

    def denormalize(self, data, constraints):
        """
        정규화된 데이터를 원래 스케일로 복원합니다.
        data: torch.Tensor 또는 numpy.ndarray
        constraints: 각 피처의 제약조건 사전 (소수점 자리수를 위해 사용)
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().numpy()
        new_data = []
        # constraints의 소수점 자리수를 미리 추출
        precisions = [int(v[4]) for v in list(constraints.values())]
        for i, element in enumerate(data):
            scalar_val = float(element)
            scalar_val = scalar_val * (self.max[i] - self.min[i])
            scalar_val = round(scalar_val, precisions[i])
            new_data.append(scalar_val)
        return new_data

# -------------------------------
# 탐색 공간 관련 클래스
# -------------------------------
class UNIVERSE:
    """
    피처 제약조건과 스케일러를 이용해 각 변수의 단위, 상한, 하한 등을 계산하고,
    모델 예측 및 구성(configuration) 조정을 위한 공통 기능을 제공합니다.
    """
    def __init__(self, constraints, feature_scaler, input_size, dtype=DTYPE):
        self.constraints = constraints
        self.dtype = dtype
        self.input_size = input_size
        self.feature_scaler = feature_scaler
        self.unit_by_feature = []
        self.upper_bounds = []
        self.lower_bounds = []
        cons_list = list(constraints.values())
        for i in range(input_size):
            unit_i = float(cons_list[i][0])
            low_i = float(cons_list[i][1])
            up_i = float(cons_list[i][2])
            range_i = feature_scaler.range[i]
            self.unit_by_feature.append(unit_i / range_i)
            self.upper_bounds.append(up_i / range_i)
            self.lower_bounds.append(low_i / range_i)
            print(f"Feature {i+1}: unit_by_feature = {self.unit_by_feature[-1]}, "
                  f"upper_bound = {self.upper_bounds[-1]}, lower_bound = {self.lower_bounds[-1]}")
        self.raw_unit = [float(cons_list[i][0]) for i in range(input_size)]
        self.raw_lower = [float(cons_list[i][1]) for i in range(input_size)]
        self.raw_upper = [float(cons_list[i][2]) for i in range(input_size)]

    def predict(self, models, x, y_prime, modeling, fake_gradient=True):
        """
        모델 예측 및 변수별 기울기를 계산합니다.
        """
        predictions = []
        gradients = []
        x_copy = x.clone()
        for model in models:
            x_var = x.clone().detach().requires_grad_(True)
            if isinstance(model, nn.Module):
                prediction = model(x_var)
                loss = abs(prediction - y_prime)
                loss.backward()
                gradient = x_var.grad.detach().numpy()
                prediction = prediction.detach().numpy()
            else:
                # ML 모델인 경우
                x_np = x_var.detach().numpy().reshape(1, -1)
                prediction = model.predict(x_np)
                if fake_gradient:
                    gradient = []
                    for j in range(len(self.unit_by_feature)):
                        new_x = x_np.copy()
                        new_x[0, j] += self.unit_by_feature[j]
                        new_prediction = model.predict(new_x)
                        slope = (new_prediction - prediction) / self.unit_by_feature[j]
                        gradient.append(slope)
                    gradient = np.array(gradient).reshape(-1)
                else:
                    gradient = np.repeat(0, len(self.unit_by_feature))
            x = x_copy.clone()
            predictions.append(prediction)
            gradients.append(gradient)
        if modeling.lower() == 'single':
            return predictions[0], gradients[0], predictions
        elif modeling.lower() == 'ensemble':
            pred_avg = sum(predictions) / len(predictions)
            grad_avg = sum(gradients) / len(gradients)
            return pred_avg, grad_avg, predictions
        else:
            raise Exception(f"[modeling error] there is no {modeling}.")

    def bounding(self, configuration):
        """
        구성(configuration)의 각 값을 단위 배수로 내림하고 상한/하한 범위 내로 조정합니다.
        """
        new_config = []
        for k, element in enumerate(configuration):
            element = element - (element % self.unit_by_feature[k])
            if element >= self.upper_bounds[k]:
                element = self.upper_bounds[k]
            elif element <= self.lower_bounds[k]:
                element = self.lower_bounds[k]
            new_config.append(element)
        return torch.tensor(new_config, dtype=self.dtype)

    def truncate(self, configuration):
        """
        구성(configuration)의 값을 각 피처별 단위로 내림하고 원시 하한/상한을 벗어나지 않도록 합니다.
        """
        new_configuration = []
        for i, value in enumerate(configuration):
            value = value - (value % self.raw_unit[i])
            value = value if value >= self.raw_lower[i] else self.raw_lower[i]
            value = value if value <= self.raw_upper[i] else self.raw_upper[i]
            new_configuration.append(value)
        return new_configuration

# -------------------------------
# Local Mode (국소 탐색) 클래스
# -------------------------------
class LocalMode(UNIVERSE):
    def __init__(self, desired, models, modeling, strategy, constraints, feature_scaler, target_scaler, input_size, dtype=DTYPE):
        super().__init__(constraints, feature_scaler, input_size, dtype)
        self.desired = desired
        self.target_scaler = target_scaler
        self.y_prime = desired / (target_scaler.max[0] - target_scaler.min[0])
        self.models = []
        self.modeling = modeling
        self.strategy = strategy
        for model in models:
            if isinstance(model, nn.Module):
                model.eval()
            self.models.append(model)

    def exhaustive(self, starting_point, top_k=5, alternative='keep_move'):
        """
        모든 가능한 조합을 탐색하여 최적의 구성을 찾습니다.
        """
        self.starting_point = self.truncate(starting_point)
        normalized_start = np.array([self.starting_point[i] / self.feature_scaler.range[i] 
                                     for i in range(self.input_size)])
        self.top_k = top_k
        self.recorder = []
        self.search_space = []
        self.counter = []
        if alternative == 'keep_up_down':
            variables = [[0, 1, 2]] * self.input_size
        else:
            variables = [[0, 1]] * self.input_size
        self.all_combinations = list(product(*variables))
        self.adj = []
        print("Starting point (normalized):", torch.tensor(normalized_start, dtype=self.dtype))
        # 초기 예측 및 기울기 계산 (대안에 따라 다름)
        if alternative in ['keep_move', 'keep_up_down']:
            pred_km, grad_km, _ = self.predict(self.models, torch.tensor(normalized_start, dtype=self.dtype), 
                                               self.y_prime, self.modeling, fake_gradient=True)
            if isinstance(pred_km, list):
                pred_km = pred_km[0]
        for combination in self.all_combinations:
            count = combination.count(1) if alternative == 'up_down' else combination.count(0)
            adjustment = [0] * len(self.unit_by_feature)
            for i, val in enumerate(combination):
                if alternative == 'up_down':
                    if val == 1:
                        adjustment[i] += self.unit_by_feature[i]
                    elif val == 0:
                        adjustment[i] -= self.unit_by_feature[i]
                    else:
                        raise Exception("ERROR")
                elif alternative == 'keep_move':
                    if pred_km > self.y_prime:
                        if val == 1:
                            adjustment[i] = adjustment[i] - self.unit_by_feature[i] if grad_km[i] >= 0 \
                                          else adjustment[i] + self.unit_by_feature[i]
                        elif val == 0:
                            pass
                        else:
                            raise Exception("ERROR")
                    else:
                        if val == 1:
                            adjustment[i] = adjustment[i] + self.unit_by_feature[i] if grad_km[i] >= 0 \
                                          else adjustment[i] - self.unit_by_feature[i]
                        elif val == 0:
                            pass
                        else:
                            raise Exception("ERROR")
                elif alternative == 'keep_up_down':
                    important_features = np.argsort(np.abs(grad_km))[::-1][:self.top_k]
                    if i in important_features:
                        if val == 2:
                            adjustment[i] += self.unit_by_feature[i]
                        elif val == 1:
                            adjustment[i] -= self.unit_by_feature[i]
                        elif val == 0:
                            pass
                        else:
                            raise Exception("ERROR")
            self.adj.append(adjustment)
            candidate = normalized_start + np.array(adjustment)
            candidate_tensor = self.bounding(candidate)
            if str(candidate_tensor.tolist()) not in self.recorder:
                self.search_space.append(candidate_tensor)
                self.counter.append(count)
                self.recorder.append(str(candidate_tensor.tolist()))
        self.predictions = []
        self.configurations = []
        self.pred_all = []
        for candidate in self.search_space:
            pred, _, pred_all_single = self.predict(self.models, candidate, self.y_prime, self.modeling, fake_gradient=False)
            pred_denorm = self.target_scaler.denormalize(pred, self.constraints)[0]
            config_denorm = self.feature_scaler.denormalize(candidate, self.constraints)
            self.predictions.append(pred_denorm)
            self.configurations.append(config_denorm)
            self.pred_all.append([self.target_scaler.denormalize(np.array([e]).astype(float), self.constraints)[0] 
                                   for e in pred_all_single])
        table = pd.DataFrame({
            'configurations': self.configurations,
            'predictions': self.predictions,
            'difference': np.abs(np.array(self.predictions) - self.desired).tolist(),
            'counter': self.counter
        })
        table = table.sort_values(by=['counter', 'difference'], ascending=[False, True])
        self.configurations = table['configurations'].tolist()
        self.predictions = table['predictions'].tolist()
        self.difference = table['difference'].tolist()
        self.counter = table['counter'].tolist()
        best_config = self.configurations[0]
        best_pred = self.predictions[0]
        return self.configurations, self.predictions, best_config, best_pred, self.pred_all

    def manual(self, starting_point, index=0, up=True):
        """
        사용자가 특정 인덱스의 피처를 수동으로 조정할 수 있도록 합니다.
        """
        self.starting_point = self.truncate(starting_point)
        normalized_start = np.array([self.starting_point[i] / self.feature_scaler.range[i] 
                                     for i in range(self.input_size)])
        adjustment = [0] * len(self.unit_by_feature)
        if up:
            adjustment[index] += self.unit_by_feature[index]
        else:
            adjustment[index] -= self.unit_by_feature[index]
        position = normalized_start + np.array(adjustment)
        position_tensor = self.bounding(position)
        pred, _, pred_all = self.predict(self.models, position_tensor, self.y_prime, self.modeling)
        pred_denorm = self.target_scaler.denormalize(pred, self.constraints)
        config_denorm = self.feature_scaler.denormalize(position_tensor, self.constraints)
        pred_all_denorm = [self.target_scaler.denormalize(np.array([e]).astype(float), self.constraints)[0] 
                           for e in pred_all]
        return [config_denorm], pred_denorm, config_denorm, pred_denorm, pred_all_denorm

# -------------------------------
# Global Mode (전역 탐색) 클래스
# -------------------------------
class GlobalMode(UNIVERSE):
    def __init__(self, desired, models, modeling, strategy, constraints, feature_scaler, 
                 target_scaler, input_size, tolerance, steps, dtype=DTYPE):
        super().__init__(constraints, feature_scaler, input_size, dtype)
        self.desired = desired
        self.target_scaler = target_scaler
        self.y_prime = desired / (target_scaler.max[0] - target_scaler.min[0])
        self.models = []
        self.modeling = modeling
        self.strategy = strategy
        self.tolerance = tolerance / (target_scaler.max[0] - target_scaler.min[0])
        self.original_tolerance = tolerance
        self.steps = steps
        for model in models:
            if isinstance(model, nn.Module):
                model.train()
            self.models.append(model)

    def predict_global(self, x, method="fd", alpha=1, p=0.1, m=5):
        """
        전역 탐색 시 각 모델의 예측과 기울기를 계산합니다.
        """
        predictions = []
        gradients = []
        x_copy = x.clone()
        for model in self.models:
            x_var = x.clone().detach().to(torch.float32)

            x_var = x_var.clone().detach().requires_grad_(True)
            if isinstance(model, nn.Module):
                prediction = model(x_var)
                loss = abs(prediction - self.y_prime)
                loss.backward()
                gradient = x_var.grad.detach().numpy()
                prediction = prediction.detach().numpy()
            else:
                x_np = x_var.detach().numpy().reshape(1, -1)
                y = model.predict(x_np)
                prediction = y
                sampled_indices = np.random.choice(np.arange(len(self.unit_by_feature)), 
                                                   replace=False, size=m).tolist()
                _unit = np.array(self.unit_by_feature)[sampled_indices].tolist()
                if method == "fd":
                    best_g = np.zeros(len(self.unit_by_feature))
                elif method == "target":
                    best_g = abs(self.y_prime - y)
                else:
                    raise ValueError("Method must be 'fd' or 'target'")
                for change in itertools.product([-1, 0, 1], repeat=len(_unit)):
                    if not np.random.uniform() < p:
                        continue
                    xx = x_np.copy()
                    xx[0][sampled_indices] = xx[0][sampled_indices] + alpha * np.array(
                        [c * u for c, u in zip(change, _unit)])
                    xx[0] = np.clip(xx[0], a_min=self.lower_bounds, a_max=self.upper_bounds)
                    yy = model.predict(xx)
                    if method == "fd":
                        l = abs(self.y_prime - y)
                        ll = abs(self.y_prime - yy)
                        g = (ll - l) / np.array(self.unit_by_feature)
                        if np.linalg.norm(g) > np.linalg.norm(best_g):
                            best_g = g
                    elif method == "target":
                        g = abs(self.y_prime - yy)
                        if best_g > g:
                            best_g = g
                gradient = np.array(best_g)
            x = x_copy.clone()
            predictions.append(prediction)
            gradients.append(gradient)
        return predictions, gradients

    def beam(self, beam_width, starting_point):
        """
        Beam search 기반으로 최적 구성을 탐색합니다.
        """
        y_prime = self.y_prime
        tolerance = self.tolerance
        x_i = [starting_point[i] / self.feature_scaler.range[i] for i in range(self.input_size)]
        x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) 
                                  for i in range(len(self.unit_by_feature))], dtype=self.dtype)
        self.beam_positions = []
        self.beam_targets = []
        self.beam_positions_denorm = []
        self.beam_targets_denorm = []
        self.beam_history = []
        self.prediction_all = []
        success = [False]
        for step in range(self.steps):
            if len(self.beam_positions) == 0:
                current_positions = [x_prime.clone().detach()]
                for _ in range(beam_width - 1):
                    random_offsets = torch.tensor(np.random.uniform(-0.5, 0.5, self.input_size), 
                                                    dtype=x_prime.dtype)
                    current_positions.append(x_prime.clone().detach() + random_offsets)
            else:
                current_positions = self.beam_positions[-1]
            configurations = []
            candidates = []
            candidates_score = []
            beam_predictions = []
            for current_pos in current_positions:
                configuration = self.feature_scaler.denormalize(current_pos, self.constraints)
                configurations.append(configuration)
                preds, grads = self.predict_global(current_pos)
                pred_avg = sum(preds) / len(preds)
                grad_avg = sum(grads) / len(grads)
                prediction_original = (pred_avg * (self.target_scaler.max[0] - self.target_scaler.min[0]))[0]
                beam_predictions.append(prediction_original)
                if abs(pred_avg - y_prime) < tolerance:
                    success.append(True)
                order = np.argsort(np.abs(grad_avg))[::-1]
                beam_indices = order[:beam_width]
                for b in beam_indices:
                    adjustment = [0] * len(self.unit_by_feature)
                    if grad_avg[b] >= 0:
                        adjustment[b] += self.unit_by_feature[b]
                    else:
                        adjustment[b] -= self.unit_by_feature[b]
                    adjustment = np.array(adjustment)
                    if pred_avg > y_prime:
                        position = current_pos.clone().detach() - adjustment
                    else:
                        position = current_pos.clone().detach() + adjustment
                    position = self.bounding(position)
                    candidates.append(position)
                    candidates_score.append(abs(grad_avg[b]))
            select = np.argsort(candidates_score)[::-1][:beam_width]
            new_positions = [candidates[s].clone().detach().to(self.dtype) for s in select]
            self.beam_positions.append(new_positions)
            self.beam_targets.append(beam_predictions)
            self.beam_positions_denorm.append(configurations)
            self.beam_targets_denorm.append(beam_predictions)
            if any(success):
                break
        flattened_positions = list(chain.from_iterable(self.beam_positions_denorm))
        flattened_predictions = list(chain.from_iterable(self.beam_targets_denorm))
        best = int(np.argsort(np.abs(np.array(flattened_predictions)-self.desired))[0])
        self.best_position = flattened_positions[best]
        self.best_prediction = flattened_predictions[best]
        return self.beam_positions_denorm, self.beam_targets_denorm, self.best_position, self.best_prediction, self.prediction_all

    def stochastic(self, num_candidates, starting_point):
        """
        확률적(stochastic) 탐색 방법으로 최적 구성을 찾습니다.
        """
        y_prime = self.y_prime
        tolerance = self.tolerance
        x_i = [starting_point[i] / self.feature_scaler.range[i] for i in range(self.input_size)]
        x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) 
                                  for i in range(len(self.unit_by_feature))], dtype=self.dtype)
        self.stochastic_configurations = []
        self.stochastic_predictions = []
        self.prediction_all = []
        method = "fd"
        for step in range(self.steps):
            configuration = self.feature_scaler.denormalize(x_prime, self.constraints)
            preds, grads = self.predict_global(x_prime)
            pred_avg = sum(preds) / len(preds)
            if isinstance(self.models[0], nn.Module):
                grad_avg = sum(grads) / len(grads)
                candidates = np.argsort(np.abs(grad_avg))[::-1][:num_candidates]
                chosen = random.choice(candidates)
                adjustment = [0] * len(self.unit_by_feature)
                if grad_avg[chosen] >= 0:
                    adjustment[chosen] += self.unit_by_feature[chosen]
                else:
                    adjustment[chosen] -= self.unit_by_feature[chosen]
                adjustment = np.array(adjustment)
                x_prime = x_prime - adjustment
            else:
                # if method != "target":
                #     best_g = np.zeros(len(self.unit_by_feature))
                #     for g in grads:
                #         if np.linalg.norm(g) > np.linalg.norm(best_g):
                #             best_g = g
                #     sum_gradients = [1 if best_g[i] > 0 else -1 if best_g[i] < 0 else 0 for i in range(len(self.unit_by_feature))]
                    
                #     x_prime = x_prime - torch.tensor(sum_gradients, dtype=x_prime.dtype) * \
                #               torch.tensor(self.unit_by_feature, dtype=x_prime.dtype)
                if method != "target":
                    best_g = None
                    best_g_norm = 0
                    for g in grads:
                        # 만약 g가 2차원이면 1차원으로 변환
                        if g.ndim == 2:
                            g = g[0]
                        g_norm = np.linalg.norm(g)
                        if g_norm > best_g_norm:
                            best_g = g
                            best_g_norm = g_norm
                    if best_g is None:
                        best_g = np.zeros(len(self.unit_by_feature))
                    sum_gradients = np.array([1 if best_g[i] > 0 else -1 if best_g[i] < 0 else 0 
                                            for i in range(len(self.unit_by_feature))])
                    x_prime = x_prime - torch.tensor(sum_gradients, dtype=x_prime.dtype) * \
                            torch.tensor(self.unit_by_feature, dtype=x_prime.dtype)
                else:
                    sum_gradients = sum(grads)
                    x_prime = x_prime + torch.tensor(sum_gradients, dtype=x_prime.dtype) * \
                              torch.tensor(self.unit_by_feature, dtype=x_prime.dtype)
            x_prime = torch.tensor(np.clip(x_prime, a_min=self.lower_bounds, a_max=self.upper_bounds), dtype=self.dtype)
            prediction_original = self.target_scaler.denormalize(pred_avg, self.constraints)[0]
            print(f"Step {step} Target: {self.desired}, Prediction: {prediction_original}")
            self.stochastic_predictions.append(prediction_original)
            self.stochastic_configurations.append(configuration)
            self.prediction_all.append([self.target_scaler.denormalize(np.array([e]).astype(float), self.constraints)[0] 
                                        for e in preds])
            if abs(prediction_original - self.desired) < self.original_tolerance:
                break
        best = int(np.argsort(np.abs(np.array(self.stochastic_predictions)-self.desired))[0])
        self.stochastic_best_position = self.stochastic_configurations[best]
        self.stochastic_best_prediction = self.stochastic_predictions[best]
        return (self.stochastic_configurations, self.stochastic_predictions, 
                self.stochastic_best_position, self.stochastic_best_prediction, self.prediction_all)

    def best_one(self, starting_point, escape=True):
        """
        한 변수씩 최적의 방향을 찾아 구성을 개선해 나갑니다.
        """
        y_prime = self.y_prime
        x_i = [starting_point[i] / self.feature_scaler.range[i] for i in range(self.input_size)]
        x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) 
                                  for i in range(len(self.unit_by_feature))], dtype=self.dtype)
        self.best_one_configurations = []
        self.best_one_predictions = []
        self.prediction_all = []
        avoid = []
        memory = []
        memory_size = 5
        previous = None
        for step in range(self.steps):
            configuration = self.feature_scaler.denormalize(x_prime, self.constraints)
            preds, grads = self.predict_global(x_prime)
            pred_avg = sum(preds)/len(preds)
            grad_avg = sum(grads)/len(grads)
            if escape:
                candidates = [i for i in np.argsort(np.abs(grad_avg))[::-1] if i not in avoid]
            else:
                candidates = np.argsort(np.abs(grad_avg))[::-1]
            chosen = candidates[0]
            adjustment = [0]*len(self.unit_by_feature)
            if grad_avg[chosen] >= 0:
                adjustment[chosen] += self.unit_by_feature[chosen]
            else:
                adjustment[chosen] -= self.unit_by_feature[chosen]
            adjustment = np.array(adjustment)
            if pred_avg > y_prime:
                x_prime = x_prime - torch.tensor(adjustment, dtype=self.dtype)
            elif pred_avg < y_prime:
                x_prime = x_prime + torch.tensor(adjustment, dtype=self.dtype)
            prediction_original = self.target_scaler.denormalize(pred_avg, self.constraints)[0]
            print(f"Step {step} Target: {self.desired}, Prediction: {prediction_original}")
            self.best_one_predictions.append(prediction_original)
            self.best_one_configurations.append(configuration)
            self.prediction_all.append([self.target_scaler.denormalize(np.array([e]).astype(float), self.constraints)[0] 
                                        for e in preds])
            memory.append(prediction_original)
            if len(memory) > memory_size:
                memory = memory[-memory_size:]
            if len(memory) == memory_size and len(set(memory)) < 3 and previous == chosen:
                avoid.append(chosen)
            if abs(pred_avg - y_prime) < self.tolerance:
                break
            if escape and len(avoid) == self.input_size:
                break
            previous = chosen
        best = int(np.argsort(np.abs(np.array(self.best_one_predictions)-self.desired))[0])
        self.best_one_best_position = self.best_one_configurations[best]
        self.best_one_best_prediction = self.best_one_predictions[best]
        return (self.best_one_configurations, self.best_one_predictions, 
                self.best_one_best_position, self.best_one_best_prediction, self.prediction_all)

# -------------------------------
# 전체 프로세스를 통합하는 함수
# -------------------------------
def parameter_prediction(data, models, desired, starting_point, mode, modeling, strategy, tolerance, beam_width,
                           num_candidates, escape, top_k, index, up, alternative, 
                           unit, lower_bound, upper_bound, data_type, decimal_place,
                           configuration_steps=100, configuration_tolerance=1.0):
    """
    전체 파라미터 최적화 과정을 진행합니다.
    """
    if starting_point is None:
        raise ValueError("starting_point cannot be None")
    if desired is None:
        raise ValueError("desired cannot be None")

    # Seed 및 환경 설정
    set_seed(DEFAULT_SEED)
    
    # DataFrame이라고 가정 (Target 컬럼이 존재)
    output_size = 1
    input_size = data.shape[1] - output_size
    print("The number of features:", input_size)
    
    # 제약조건 생성
    constraints = create_constraints(unit, lower_bound, upper_bound, data_type, decimal_place)
    
    # 스케일러 생성 (Target과 Feature 구분)
    target_scaler = MinMaxScaling(data['Target'])
    feature_columns = [col for col in data.columns if col != 'Target']
    feature_scaler = MinMaxScaling(data[feature_columns])
    print('Scaling done')
    
    configurations, predictions, best_config, best_pred, pred_all = None, None, None, None, None
    if mode == 'global':
        G = GlobalMode(desired=desired, models=models, modeling=modeling, strategy=strategy, 
                       constraints=constraints, feature_scaler=feature_scaler, target_scaler=target_scaler,
                       input_size=input_size, tolerance=tolerance, steps=configuration_steps)
        if strategy == 'beam':
            configurations, predictions, best_config, best_pred, pred_all = G.beam(beam_width, starting_point)
            print('Global beam mode activated')
        elif strategy == 'stochastic':
            configurations, predictions, best_config, best_pred, pred_all = G.stochastic(num_candidates, starting_point)
            print('Global stochastic mode activated')
        elif strategy == 'best_one':
            configurations, predictions, best_config, best_pred, pred_all = G.best_one(starting_point, escape)
            print('Global best_one mode activated')
        else:
            raise ValueError("Unknown strategy for global mode")
    elif mode == 'local':
        L = LocalMode(desired=desired, models=models, modeling=modeling, strategy=strategy, 
                      constraints=constraints, feature_scaler=feature_scaler, target_scaler=target_scaler,
                      input_size=input_size)
        if strategy == 'exhaustive':
            configurations, predictions, best_config, best_pred, pred_all = L.exhaustive(starting_point, top_k, alternative)
            print('Local exhaustive mode activated')
        elif strategy == 'manual':
            configurations, predictions, best_config, best_pred, pred_all = L.manual(starting_point, index, up)
            print('Local manual mode activated')
        else:
            raise ValueError("Unknown strategy for local mode")
    else:
        raise ValueError("Unknown mode")
        
    print("Configurations type:", type(configurations))
    print("Predictions type:", type(predictions))
    print("Best configuration type:", type(best_config))
    print("Pred_all type:", type(pred_all))
    
    return configurations, predictions, best_config, best_pred, pred_all

# -------------------------------
# 메인 실행 함수
# -------------------------------
    
"""
전체 코드 인터페이스 문서

-------------------------------
Global Functions
-------------------------------

set_seed(seed=DEFAULT_SEED)
    Input:
        seed: int  
            - 랜덤 시드를 지정 (default: 2025)
    Output:
        None
    Description:
        Python, NumPy, PyTorch의 랜덤 시드를 설정하고, CuDNN 옵션을 조정하여 재현성을 보장함.

create_constraints(unit, lower_bound, upper_bound, data_type, decimal_place)
    Input:
        unit: list of float, shape: (n_features,)
            - 각 피처의 단위 값
        lower_bound: list of float, shape: (n_features,)
            - 각 피처의 하한 값
        upper_bound: list of float, shape: (n_features,)
            - 각 피처의 상한 값
        data_type: list of type, shape: (n_features,)
            - 각 피처의 데이터 타입 (예: float, int)
        decimal_place: list of int, shape: (n_features,)
            - 각 피처의 소수점 자리수
    Output:
        constraints: dict
            - key: "ATT1", "ATT2", ...  
            - value: [unit, lower_bound, upper_bound, data_type, decimal_place] (각 피처별)
    Description:
        각 피처의 제약조건을 사전(dict) 형태로 생성함.


-------------------------------
Class: MinMaxScaling
-------------------------------

MinMaxScaling(data, dtype=DTYPE)
    Input:
        data: pd.DataFrame, pd.Series, or np.ndarray
            - 정규화할 데이터
            - DataFrame의 경우 shape: (n_samples, n_features)
            - 1차원 데이터(Series 또는 1D array)는 (n_samples,)로 주어지며 내부에서 (n_samples, 1)로 reshape됨.
        dtype: torch.dtype (default: torch.float32)
    Instance Variables:
        self.max: list of float, 길이 n_features
            - 각 피처의 최대값
        self.min: list of float, 길이 n_features
            - 각 피처의 최소값
        self.range: list of float, 길이 n_features
            - (최대값 - 최소값) 각 피처별
        self.scaled_data: pd.DataFrame, shape: (n_samples, n_features)
            - Min-Max 정규화된 데이터
        self.tensor_data: torch.Tensor, shape: (n_samples, n_features)
            - 정규화된 데이터를 torch.Tensor로 저장
    Output:
        없음 (인스턴스 생성 시 내부적으로 정규화를 수행)

denormalize(data, constraints)
    Input:
        data: torch.Tensor or np.ndarray
            - 정규화된 데이터 벡터 (보통 1차원, shape: (n_features,))
        constraints: dict
            - create_constraints()에 의해 생성된 제약조건 사전  
              (특히 각 피처의 소수점 자리수를 위해 사용)
    Output:
        new_data: list of float, 길이 n_features
            - 원래 스케일로 복원된 값 (각 피처별로 반올림 적용)
    Description:
        정규화된 데이터 값을 원래의 스케일로 복원함.


-------------------------------
Class: UNIVERSE
-------------------------------

UNIVERSE(constraints, feature_scaler, input_size, dtype=DTYPE)
    Input:
        constraints: dict
            - 각 피처의 제약조건 사전 (create_constraints() 결과)
        feature_scaler: MinMaxScaling instance
            - 피처 데이터를 정규화한 스케일러
        input_size: int
            - 피처의 개수
        dtype: torch.dtype (default: torch.float32)
    Instance Variables:
        self.unit_by_feature: list of float, 길이 input_size
            - 각 피처에 대해 (unit / (max - min)) 값 (정규화된 단위)
        self.upper_bounds: list of float, 길이 input_size
            - 각 피처의 정규화된 상한 값
        self.lower_bounds: list of float, 길이 input_size
            - 각 피처의 정규화된 하한 값
        self.raw_unit, self.raw_lower, self.raw_upper: list of 원래 값들 (길이 input_size)
    Output:
        없음

predict(models, x, y_prime, modeling, fake_gradient=True)
    Input:
        models: list
            - 모델 목록; 각 모델은 torch.nn.Module (DL) 또는 .predict() 메서드를 가진 ML 모델
        x: torch.Tensor
            - 정규화된 피처 벡터, shape: (n_features,) 또는 (1, n_features)
        y_prime: float or tensor
            - 목표 타깃의 정규화된 값 (scalar)
        modeling: str
            - "single" 또는 "ensemble" (모델 예측 방식)
        fake_gradient: bool
            - ML 모델의 경우 유사 기울기 계산 여부
    Output:
        if modeling=="single":
            (prediction, gradient, predictions)
            - prediction: np.ndarray (모델 출력, shape는 모델에 따라 다름)
            - gradient: np.ndarray, shape: (n_features,)
            - predictions: list of np.ndarray (각 모델별 예측)
        if modeling=="ensemble":
            (pred_avg, grad_avg, predictions)
            - pred_avg: 평균 예측값 (np.ndarray)
            - grad_avg: 평균 기울기 (np.ndarray, shape: (n_features,))
            - predictions: list of np.ndarray
    Description:
        각 모델에 대해 예측값과 기울기를 계산하고, 모델링 방식에 따라 결과를 반환함.

bounding(configuration)
    Input:
        configuration: list or np.ndarray of float, 길이: n_features
            - 정규화된 피처 구성 벡터
    Output:
        torch.Tensor, shape: (n_features,)
            - 각 요소를 단위 배수로 내림한 후 상한/하한 범위 내로 조정된 구성

truncate(configuration)
    Input:
        configuration: list or np.ndarray of float, 길이: n_features
            - 원래 스케일의 구성 벡터
    Output:
        new_configuration: list of float, 길이: n_features
            - 각 피처별 단위로 내림(truncate)되어 원시 하한/상한 범위 내에 있는 값들


-------------------------------
Class: LocalMode (subclass of UNIVERSE)
-------------------------------

LocalMode(desired, models, modeling, strategy, constraints, feature_scaler, target_scaler, input_size, dtype=DTYPE)
    Input:
        desired: float
            - 원하는 타깃 값 (원래 스케일)
        models: list
            - 예측에 사용할 모델 목록
        modeling: str
            - "single" 또는 "ensemble"
        strategy: str
            - "exhaustive" 또는 "manual"
        constraints: dict
            - 제약조건 사전
        feature_scaler: MinMaxScaling instance
            - 피처 스케일러
        target_scaler: MinMaxScaling instance
            - 타깃 스케일러
        input_size: int
            - 피처 개수
        dtype: torch.dtype (default: torch.float32)
    Instance Variables:
        self.desired: float
        self.target_scaler: MinMaxScaling for target
        self.y_prime: float
            - desired를 타깃 스케일러로 정규화한 값
        self.models: list (모델들, torch.nn.Module인 경우 eval() 모드)
        self.modeling, self.strategy: str
    Output:
        없음

exhaustive(starting_point, top_k=5, alternative='keep_move')
    Input:
        starting_point: list of float, 길이: n_features
            - 초기 구성 (원래 스케일)
        top_k: int
            - 중요한 피처 수 (정렬 시 상위 몇 개를 고려할지)
        alternative: str
            - 탐색 대안 (예: "keep_move", "keep_up_down")
    Output:
        Tuple:
            configurations: list of configuration, each a list of float (원래 스케일)
            predictions: list of float (각 구성에 대한 타깃 예측, 원래 스케일)
            best_config: list of float, best configuration
            best_pred: float, best predicted 타깃 값
            pred_all: list of lists, 각 구성마다 각 모델의 예측 값

manual(starting_point, index=0, up=True)
    Input:
        starting_point: list of float, 길이: n_features (원래 스케일)
        index: int, 조정할 피처 인덱스
        up: bool, 조정 방향 (True: 증가, False: 감소)
    Output:
        Tuple:
            ([config_denorm], pred_denorm, config_denorm, pred_denorm, pred_all_denorm)
            - config_denorm: list of float (원래 스케일의 구성)
            - pred_denorm: 타깃 예측 (원래 스케일)
            - pred_all_denorm: list of 각 모델의 예측 값 (원래 스케일)


-------------------------------
Class: GlobalMode (subclass of UNIVERSE)
-------------------------------

GlobalMode(desired, models, modeling, strategy, constraints, feature_scaler, target_scaler, input_size, tolerance, steps, dtype=DTYPE)
    Input:
        desired: float
            - 원하는 타깃 값 (원래 스케일)
        models: list
        modeling: str ("single" 또는 "ensemble")
        strategy: str ("beam", "stochastic", "best_one")
        constraints: dict
        feature_scaler: MinMaxScaling for features
        target_scaler: MinMaxScaling for target
        input_size: int
        tolerance: float
            - 허용 오차 (원래 스케일)
        steps: int
            - 최대 반복 스텝 수
        dtype: torch.dtype (default: torch.float32)
    Instance Variables:
        self.desired, self.y_prime (정규화된 desired), self.tolerance (정규화된), self.original_tolerance, self.steps, 등.
    Output:
        없음

predict_global(x, method="fd", alpha=1, p=0.1, m=5)
    Input:
        x: torch.Tensor, 정규화된 피처 벡터, shape: (n_features,)
        method: str, "fd" 또는 "target"
        alpha: float, 스텝 크기 계수
        p: float, 변화 적용 확률
        m: int, finite difference 계산 시 샘플링할 피처 수
    Output:
        Tuple:
            predictions: list of np.ndarray (각 모델의 예측, shape는 모델에 따라 다름)
            gradients: list of np.ndarray, shape: (n_features,) 각 모델의 기울기

beam(beam_width, starting_point)
    Input:
        beam_width: int, 빔 서치 시 고려할 후보 수
        starting_point: list of float, 길이: n_features (원래 스케일)
    Output:
        Tuple:
            beam_positions_denorm: list of lists, 각 스텝별 후보 구성 (원래 스케일)
            beam_targets_denorm: list of lists, 각 스텝별 후보 타깃 예측 (원래 스케일)
            best_position: list of float, best configuration (원래 스케일)
            best_prediction: float, best 타깃 예측 (원래 스케일)
            prediction_all: list (빌드 중 수집된 추가 예측 값들, 구조는 구현에 따라 다름)

stochastic(num_candidates, starting_point)
    Input:
        num_candidates: int, 후보 조정 시 고려할 변수 수
        starting_point: list of float, 길이: n_features (원래 스케일)
    Output:
        Tuple:
            stochastic_configurations: list of configuration (원래 스케일)
            stochastic_predictions: list of float (각 구성에 대한 타깃 예측, 원래 스케일)
            stochastic_best_position: list of float, best configuration (원래 스케일)
            stochastic_best_prediction: float, best 타깃 예측 (원래 스케일)
            prediction_all: list of lists, 각 구성마다 각 모델의 예측 값

best_one(starting_point, escape=True)
    Input:
        starting_point: list of float, 길이: n_features (원래 스케일)
        escape: bool, 지역 최소값 탈출 허용 여부
    Output:
        Tuple:
            best_one_configurations: list of configuration (원래 스케일)
            best_one_predictions: list of float (타깃 예측, 원래 스케일)
            best_one_best_position: list of float, best configuration (원래 스케일)
            best_one_best_prediction: float, best 타깃 예측 (원래 스케일)
            prediction_all: list of lists, 각 구성별 예측 값들

-------------------------------
Function: parameter_prediction
-------------------------------

parameter_prediction(data, models, desired, starting_point, mode, modeling, strategy, tolerance, beam_width,
                      num_candidates, escape, top_k, index, up, alternative, unit, lower_bound, upper_bound,
                      data_type, decimal_place, configuration_steps=100, configuration_tolerance=1.0)
    Input:
        data: pd.DataFrame, shape: (n_samples, n_features+1)
            - 반드시 'Target' 컬럼을 포함 (타깃 변수)
        models: list
            - 사용될 모델들의 목록
        desired: float
            - 원하는 타깃 값 (원래 스케일)
        starting_point: list of float, 길이: n_features
            - 초기 피처 구성 (원래 스케일)
        mode: str
            - "global" 또는 "local"
        modeling: str
            - "ensemble" 또는 "single"
        strategy: str
            - global: 'beam', 'stochastic', 'best_one'
            - local: 'exhaustive', 'manual'
        tolerance: float
            - 허용 오차 (원래 스케일)
        beam_width: int
            - beam search 시 후보 수
        num_candidates: int
            - stochastic search 시 후보 변수 수
        escape: bool
            - best_one 전략 시 지역 최소값 탈출 여부
        top_k: int
            - exhaustive 전략 시 중요한 피처 수
        index: int
            - manual 전략 시 조정할 피처 인덱스
        up: bool
            - manual 전략 시 조정 방향
        alternative: str
            - exhaustive 전략 대안 (예: "keep_move", "keep_up_down")
        unit: list of float, 길이: n_features
            - 각 피처의 단위 값
        lower_bound: list of float, 길이: n_features
            - 각 피처의 하한 값
        upper_bound: list of float, 길이: n_features
            - 각 피처의 상한 값
        data_type: list of type, 길이: n_features
            - 각 피처의 데이터 타입
        decimal_place: list of int, 길이: n_features
            - 각 피처의 소수점 자리수
        configuration_steps: int, (default: 100)
            - 최대 반복 스텝 수
        configuration_tolerance: float, (default: 1.0)
            - 내부 반복 종료 기준 (정규화된 값)
    Output:
        Tuple:
            configurations: list of configuration, each configuration is a list of float, 길이: n_features
            predictions: list of float, 각 구성에 대한 타깃 예측 (원래 스케일)
            best_config: list of float, best configuration (원래 스케일)
            best_pred: float, best 타깃 예측 (원래 스케일)
            pred_all: list of lists, 각 구성별 모델 예측 값들
    Description:
        전체 파라미터 최적화 프로세스를 실행하여, 여러 전략(글로벌/로컬)로 최적 구성을 탐색하고 결과를 반환함.

-------------------------------
Function: run
-------------------------------

run()
    Input:
        None
    Output:
        None (실행 시 최종 구성, 예측값, best 구성 및 best 예측값을 콘솔에 출력)
    Description:
        더미 데이터와 더미 모델을 생성하여 parameter_prediction() 함수를 실행하는 예제 함수.
"""
