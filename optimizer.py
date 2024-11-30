# import sys
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# import random
# import xgboost
# from itertools import product, chain

# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import HuberRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# # 전역 변수 및 설정
# neural_list = ['MLP', 'Conv']
# regression_epochs = 500
# regression_eta = 0.01
# regression_show_epochs = True 
# regression_dropout = 0.3
# regression_batch = 256
# regression_hidden_size = 32

# class MinMaxScaling :
#     def __init__(self, data): #np.DataFrame
#         self.max, self.min, self.range = [],[], []
#         self.data = pd.DataFrame([])
#         data = data.values.reshape(-1,1) if len(data.values.shape) == 1 else data.values

#         epsilon = 2
#         for i in range(data.shape[1]) :
#             max_, min_ = max(data[:,i]), min(data[:,i])
#             if max_ == min_ : max_ *= epsilon
#             self.max.append(max_)
#             self.min.append(min_)
#             self.range.append(max_-min_)
#             self.data = pd.concat([self.data, pd.DataFrame((data[:,i])/(max_-min_))],axis = 1)
#         self.data = torch.tensor(self.data.values, dtype = dtype)

#     def denormalize(self, data):
#         data = data.detach().numpy() if isinstance(data, torch.Tensor) else data
#         new_data = []
#         for i, element in enumerate(data):
#             element = (element * (self.max[i] - self.min[i])) 
#             element = round(element, np.array(list(constraints.values()))[:,4][i])
#             new_data.append(element)
#         return new_data
    
# class MLP(nn.Module):
#     def __init__(self, input_size = input_size, output_size = output_size, hidden_size = regression_hidden_size, n_layers = 1):
#         super(MLP, self).__init__()
#         self.first_layer = nn.Linear(input_size, hidden_size)
#         self.layers = []
#         self.layers_dropout =[]

#         if n_layers > 1 : 
#             self.layers += [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers - 1)]
#         #     self.layers_dropout += [nn.Dropout(regression_dropout) for _ in range(n_layers - 1)]

#         self.last_layer = nn.Linear(hidden_size, output_size)
#         #  self.dropout_input = nn.Dropout(regression_dropout)

#     def forward(self, x):
#         x = torch.relu(self.first_layer(x))
#     #    x = self.dropout_input(x)
#         for i in range(len(self.layers)):
#             layer = self.layers[i]
#         #     drop = self.layers_dropout[i]
#             x = torch.relu(layer(x))
#         #     x = drop(x)

#         x = self.last_layer(x)
#         return x
    
# def ML_XGBoost():
#     return xgboost.XGBRegressor()

# def ML_LinearRegressor():
#     return LinearRegression()

# def ML_Ridge():
#     return Ridge(alpha=1.0)

# def ML_Lasso():
#     return Lasso(alpha = 0.1)

# def ML_DecisionTreeRegressor():
#     return DecisionTreeRegressor(max_depth = 5)

# def ML_RandomForestRegressor():
#     return RandomForestRegressor()

# def ML_GradientBoostingRegressor():
#     return GradientBoostingRegressor()

# def ML_SVR():
#     return SVR(kernel='rbf')

# def ML_KNeighborsRegressor():
#     return KNeighborsRegressor(n_neighbors=5)

# def ML_HuberRegressor():
#     return HuberRegressor()

# def ML_GaussianProcessRegressor():
#     return GaussianProcessRegressor()
    
# def modeller(model) :
#     if model == 'MLP()':
#         x = MLP()
#     elif model == 'ML_XGBoost()':
#         x = ML_XGBoost()
#     elif model == 'MLP(n_layers = 2)':
#         x = MLP(n_layers=2)
#     elif model == 'MLP(n_layers = 3)':
#         x = MLP(n_layers=3)
#     elif model == 'ML_LinearRegressor()':
#         x = ML_LinearRegressor()
#     elif model == 'ML_Ridge()':
#         x = ML_Ridge()
#     elif model == 'ML_Lasso()':
#         x = ML_Lasso()
#     elif model == 'ML_DecisionTreeRegressor()':
#         x = ML_DecisionTreeRegressor()
#     elif model == 'ML_RandomForestRegressor()':
#         x = ML_RandomForestRegressor()
#     elif model == 'ML_GradientBoostingRegressor()':
#         x = ML_GradientBoostingRegressor()
#     elif model == 'ML_SVR()':
#         x = ML_SVR()
#     elif model == 'ML_KNeighborsRegressor()':
#         x = ML_KNeighborsRegressor()
#     elif model == 'ML_HuberRegressor()':
#         x = ML_HuberRegressor()
#     elif model == 'ML_GaussianProcessRegressor()':
#         x = ML_GaussianProcessRegressor()
#     else:
#         raise ValueError(f"Unknown model: {model}")
        
#     return x
# class EarlyStopping:
#     def __init__(self, patience=10, delta=0.1):
#         self.patience = patience
#         self.delta = delta
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.train_loss_min = float('inf')

#     def __call__(self, train_loss):
#         score = -train_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.train_loss_min = train_loss
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.counter = 0
#             self.train_loss_min = train_loss

# def train(feature, target):
#         trained_models = []
#         all_training_losses = []
#         for i, model in enumerate(models):
#             if any([m in model_list[i] for m in neural_list]): 
#                 training_losses = _train_nn(model, feature, target)
#                 all_training_losses.append(training_losses)
#             else: 
#                 _train_ml(model, feature, target)
#                 all_training_losses.append(None)
#             trained_models.append(model)
#             print(f"The regressor {model_list[i]} has been trained.")
#         return trained_models, all_training_losses

# def _train_nn(model, feature, target, epochs = regression_epochs):
#     optimizer = optim.Adam(model.parameters(), lr=regression_eta)
#     criterion = nn.MSELoss() 
#     early_stopping = EarlyStopping()

#     training_losses = []
#     Batch = []
#     for indexer in range((len(feature) // regression_batch) + 1):
#         Batch.append([feature[indexer * regression_batch : (indexer+1) * regression_batch,:],
#                     target[indexer * regression_batch : (indexer+1) * regression_batch,:]])
#     for epoch in range(regression_epochs):
#         train_loss = 0
#         for fea, tar in Batch:
#             optimizer.zero_grad()
#             output = model(feature)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         train_loss /= len(Batch)
#         if regression_show_epochs and epoch % 10 == 0: print(f"Epoch [{epoch}/{epochs}], Loss: {train_loss:.4f}")    
#         early_stopping(train_loss)
#         training_losses.append(train_loss)
#         if epoch > 100 and early_stopping.early_stop:
#             print("Early stopping triggered.")
#         #     model.eval()
#             break
#     #  model.eval()
#     return training_losses
# def _train_ml(model, feature, target):
#     model.fit(feature, target)       
        
# class UNIVERSE:
#     def __init__(self, constraints, feature, target, input_size, dtype, model_list, neural_list):
#         self.constraints = constraints
#         self.feature = feature
#         self.target = target
#         self.input_size = input_size
#         self.dtype = dtype
#         self.model_list = model_list
#         self.neural_list = neural_list

#         self.unit_by_feature = [np.array(list(constraints.values()))[:,0][i] / (self.feature.max[i] - self.feature.min[i])
#                                 for i in range(self.input_size)]
#         self.upper_bounds = [np.array(list(constraints.values()))[:,2][i] / (self.feature.max[i] - self.feature.min[i])  
#                                 for i in range(self.input_size)]
#         self.lower_bounds = [np.array(list(constraints.values()))[:,1][i] / (self.feature.max[i] - self.feature.min[i])  
#                                 for i in range(self.input_size)]    

#         self.raw_unit = np.array(list(constraints.values()))[:,0].tolist()
#         self.raw_lower = np.array(list(constraints.values()))[:,1].tolist()
#         self.raw_upper = np.array(list(constraints.values()))[:,2].tolist()


#     def predict(self, models, x, y_prime, modeling, fake_gradient = True):
#         predictions,gradients = [], []
#         copy = x.clone()
#         for i, model in enumerate(models):
#             x = x.clone().detach().requires_grad_(True)
#             if any([m in model_list[i] for m in neural_list]): 
#                 prediction = model(x)
#                 loss = abs(prediction - y_prime)
#                 loss.backward()
#                 gradient = x.grad.detach().numpy()
#                 prediction = prediction.detach().numpy()

#             else: # ML
#                 x = x.detach().numpy().reshape(1,-1)
#                 prediction = model.predict(x)
#                 if fake_gradient :
#                     gradient = []
#                     for j in range(x.shape[1]):
#                         new_x = x.copy()
#                         new_x[0,j] += self.unit_by_feature[j]
#                         new_prediction = model.predict(new_x)
#                         slope = (new_prediction - prediction) / self.unit_by_feature[j]
#                         gradient.append(slope)
#                     gradient = np.array(gradient).reshape(-1)   
#                 else : gradient = np.repeat(0, x.shape[1])
#             x = copy.clone()
#             predictions.append(prediction)
#             gradients.append(gradient)

#         if modeling == 'single': return predictions[0], gradients[0]
#         elif modeling == 'averaging': return sum(predictions)/len(predictions), sum(gradients)/len(gradients)   
#         elif modeling == 'ensemble': return "TODO"
#         else: raise Exception(f"[modeling error] there is no {modeling}.")


#     def bounding(self, configuration):
#         new = []
#         for k, element in enumerate(configuration):
#             element = element - (element % self.unit_by_feature[k])
#             if element >= self.upper_bounds[k] : element = self.upper_bounds[k]
#             elif element <= self.lower_bounds[k] : element = self.lower_bounds[k]
#             else: pass
#             new.append(element)        
#         configuration = torch.tensor(new, dtype = dtype)        
#         return configuration


#     def truncate(self, configuration):
#         new_configuration = []
#         for i, value in enumerate(configuration) :
#             value = value - (value % self.raw_unit[i])
#             value = value if value >= self.raw_lower[i] else self.raw_lower[i]
#             value = value if value <= self.raw_upper[i] else self.raw_upper[i]
#             new_configuration.append(value)
#         # configuration = torch.tensor(new_configuration, dtype = dtype)     
#         return configuration
        
# class LocalMode(UNIVERSE):
#     def __init__(self, desired, models, modeling, strategy):
#         super().__init__()
#         self.desired = desired
#         self.y_prime = self.desired / (target.max[0] - target.min[0])
#         self.models = []
#         self.modeling = modeling
#         self.strategy = strategy
#         for model in models : 
#             try: model.eval()
#             except: pass
#             self.models.append(model)

#     def exhaustive(self, starting_point = np.array(list(constraints.values()))[:,5].tolist(), top_k = 5, alternative = 'keep_move'):
#         self.starting_point = super().truncate(starting_point)
#         self.starting_point = np.array([self.starting_point[i] / (feature.max[i] - feature.min[i]) 
#                                         for i in range(input_size)])
#         self.top_k = top_k
#         self.recorder = []

#         self.search_space, self.counter = [],[]
#         if alternative == 'keep_up_down' :
#             variables = [[0, 1, 2]] * input_size
#         else :
#             variables = [[0, 1]] * input_size
#         self.all_combinations = list(product(*variables))
#         self.adj = []

#         if alternative == 'keep_move' or alternative == 'keep_up_down':
#             prediction_km, gradient_km = super().predict(self.models,torch.tensor(self.starting_point, dtype = dtype), 
#                                             self.y_prime, self.modeling, fake_gradient = True)
#             prediction_km = prediction_km[0] if isinstance(prediction_km,list) else prediction_km
#         for combination in self.all_combinations :
#             count = combination.count(1) if alternative == 'up_down' else combination.count(0)
#             adjustment = np.repeat(0,len(self.unit_by_feature)).tolist()
#             for i, boolean in enumerate(list(combination)):
#                 if alternative == 'up_down':
#                     if boolean == 1 :   adjustment[i] = adjustment[i] + self.unit_by_feature[i]
#                     elif boolean == 0 : adjustment[i] = adjustment[i] - self.unit_by_feature[i]
#                     else: raise Exception("ERROR")

#                 elif alternative == 'keep_move':
#                     if prediction_km > self.y_prime :                        
#                         if boolean == 1 :   
#                             if gradient_km[i] >= 0 :
#                                 adjustment[i] = adjustment[i] - self.unit_by_feature[i]
#                             else:
#                                 adjustment[i] = adjustment[i] + self.unit_by_feature[i]
#                         elif boolean == 0 : pass
#                         else: raise Exception("ERROR")            

#                     else: 
#                         if boolean == 1 :   
#                             if gradient_km[i] >= 0 :
#                                 adjustment[i] = adjustment[i] + self.unit_by_feature[i]
#                             else:
#                                 adjustment[i] = adjustment[i] - self.unit_by_feature[i]
#                         elif boolean == 0 : pass
#                         else: raise Exception("ERROR")    

#                 elif alternative == 'keep_up_down' :
#                     important_features = np.argsort(abs(gradient_km))[::-1][:self.top_k]
#                     if i in important_features :
#                         if boolean == 2 :   adjustment[i] = adjustment[i] + self.unit_by_feature[i]
#                         elif boolean == 1 : adjustment[i] = adjustment[i] - self.unit_by_feature[i]
#                         elif boolean == 0 : pass
#                         else: raise Exception("ERROR")   

#             self.adj.append(adjustment)
#             candidate = self.starting_point + adjustment         
#             candidate = super().bounding(candidate)
#             if str(candidate) not in self.recorder : 
#                 self.search_space.append(candidate)
#                 self.counter.append(count)
#                 self.recorder.append(str(candidate))
#     #    print(len(self.search_space))
#         self.predictions = []
#         self.configurations = []
#         for candidate in self.search_space :
#             prediction, _ = super().predict(self.models,candidate, self.y_prime, self.modeling, fake_gradient = False)
#             prediction = target.denormalize(prediction)[0]
#             configuration = feature.denormalize(candidate)
#             self.predictions.append(prediction)
#             self.configurations.append(configuration)

#         self.table = pd.DataFrame({'configurations':self.configurations,'find_dup' : self.configurations,
#                                     'predictions':self.predictions,'difference' : np.array(abs(np.array(self.predictions)-self.desired)).tolist(),
#                                     'counter':self.counter})
#         self.table['find_dup'] = self.table['find_dup'].apply(lambda x: str(x))
#         self.table = self.table[~self.table.duplicated(subset='find_dup', keep='first')]
#         self.table = self.table.drop(columns=['find_dup'])

#         self.table = self.table.sort_values(by='counter', ascending=False).sort_values(by='difference', ascending=True)
#         self.configurations = self.table['configurations']
#         self.predictions = self.table['predictions']
#         self.difference = self.table['difference']
#         self.counter = self.table['counter']
        
#         configurations = self.configurations[:].values.tolist()
#         predictions = self.predictions[:].values.tolist()
#         best_config = configurations[0]
#         best_pred = predictions[0]

#         try: return configurations, predictions, best_config, best_pred
#         except : return self.configurations[:].values.tolist(), self.predictions[:].values.tolist(), best_config, best_pred

#     def manual(self, starting_point = np.array(list(constraints.values()))[:,5].tolist(), index=0, up=True):
#         self.starting_point = super().truncate(starting_point)
#         self.starting_point = np.array([self.starting_point[i] / (feature.max[i] - feature.min[i]) 
#                                         for i in range(input_size)])

#         adjustment = np.repeat(0,len(self.unit_by_feature)).tolist()
#         if up : adjustment[index] += self.unit_by_feature[index]
#         else : adjustment[index] -= self.unit_by_feature[index]
#         position = self.starting_point + adjustment         
#         position = super().bounding(position)        
#         prediction, _ = super().predict(self.models,position, self.y_prime, self.modeling)
#         prediction = target.denormalize(prediction)
#         configuration = feature.denormalize(position)
#         return [configuration], prediction, configuration, prediction
        
# class GlobalMode(UNIVERSE):
#     def __init__(self, desired, models, modeling, strategy, tolerance = configuration_tolerance, steps = configuration_steps):
#         super().__init__()
#         self.desired = desired
#         self.y_prime = self.desired / (target.max[0] - target.min[0])
#         self.models = []
#         self.modeling = modeling
#         self.strategy = strategy
#         self.tolerance = tolerance / (target.max[0] - target.min[0])
#         self.steps = steps
#         for model in models : 
#             try: model.eval()
#             except: pass
#             self.models.append(model)

#     def predict_global(self, models, x, y_prime):

#         predictions,gradients = [], []
#         copy = x.clone()
#         for i, model in enumerate(models):
#             x = x.clone().detach().requires_grad_(True)
#             if any([m in model_list[i] for m in neural_list]): 
#                 prediction = model(x)
#                 loss = prediction - y_prime
#                 loss.backward()
#                 gradient = x.grad.detach().numpy()
#                 prediction = prediction.detach().numpy()

#             else: # ML
#                 x = x.detach().numpy().reshape(1,-1)
#                 prediction = model.predict(x)
#                 gradient = []
#                 for j in range(x.shape[1]):
#                     new_x = x.copy()
#                     new_x[0,j] += self.unit_by_feature[j]
#                     new_prediction = model.predict(new_x)
#                     slope = (new_prediction - prediction) / self.unit_by_feature[j]
#                     gradient.append(slope)
#                 gradient = np.array(gradient).reshape(-1)   
#             x = copy.clone()
#             predictions.append(prediction)
#             gradients.append(gradient)
#         return predictions, gradients

#     def beam(self, beam_width = 5, starting_point = np.array(list(constraints.values()))[:,5].tolist()) :
#         self.beam_width = beam_width
#         y_prime = self.desired / (target.max[0] - target.min[0])
#         tolerance = self.tolerance
#         final = None

#         x_i = [starting_point[i] / (feature.max[i] - feature.min[i]) for i in range(input_size)]  
#         x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], 
#                                 dtype = dtype)

#         self.beam_positions, self.beam_targets = [], []
#         self.beam_positions_denorm, self.beam_targets_denorm = [], []

#         self.beam_history = []
#         self.previous_gradient = [[], [], [], [], []]

#         success = [False]
#         which = []
#         close = False
#         close_margin = 10
#         for step in range(self.steps):
#             if len(self.beam_positions) == 0 :
#                 current_positions = [x_prime.clone().detach()]
#                 for j in range(self.beam_width - 1):
#                     random_offsets = torch.tensor(np.random.uniform(-0.5, 0.5, input_size), dtype=x_prime.dtype)
#                     current_positions += [x_prime.clone().detach() + random_offsets]
#             else :
#                 current_positions = self.beam_positions[-1]

#             configurations = []
#             candidates = []
#             candidates_score = []
#             beam_predictions = []
#             beams = []
#             for p, current_pos in enumerate(current_positions):
#                 configuration = feature.denormalize(current_pos.clone().detach())
#                 configurations.append(configuration)

#                 predictions, gradients = self.predict_global(self.models, x = current_pos, y_prime = y_prime)                
#                 prediction_avg = sum(predictions)/len(predictions) ###
#                 gradient_avg = sum(gradients)/len(gradients)  

#                 prediction_original = prediction_avg * (target.max[0] - target.min[0])
#                 prediction_original = prediction_original[0]
#                 beam_predictions.append(prediction_original)     

#                 if abs(prediction_original - self.desired) < close_margin : close = True
#                 else : close = False
#             #    print(close)
#                 if abs(prediction_avg - y_prime) < tolerance:
#                     best_config = configuration
#                     best_pred = prediction_original
#                     success.append(True)

#             #     if close :
#             #         order = np.argsort(abs(gradient_avg))
#             #     else :
#             #         order = np.argsort(abs(gradient_avg))[::-1]
#                 order = np.argsort(abs(gradient_avg))[::-1]
#                 beam = order[:self.beam_width]

#                 for b in beam:

#                     adjustment = list(np.repeat(0,len(self.unit_by_feature)))
#                     if gradient_avg[b] >= 0 :
#                         adjustment[b] += self.unit_by_feature[b]
#                     else:
#                         adjustment[b] -= self.unit_by_feature[b]    


#                     adjustment = np.array(adjustment)
#                     if prediction_avg > y_prime : 
#                         position = current_pos.clone().detach() - adjustment 
#                     else :
#                         position = current_pos.clone().detach() + adjustment 

#                     position = super().bounding(position)
#                     candidates.append(position)
#                     candidates_score.append(abs(gradient_avg[b]))

#             if step % 10 == 0 : print(f"Step {step} Target : {self.desired}, Prediction : {beam_predictions}")
#             select = np.argsort(candidates_score)[::-1][:self.beam_width]
#             new_positions = [torch.tensor(candidates[s], dtype = dtype) for s in select]

#             if len(beam_predictions) == 1 : beam_predictions = list(np.repeat(beam_predictions[0],self.beam_width))
#             self.beam_positions.append(new_positions)
#             self.beam_targets.append(beam_predictions)
#             self.beam_history.append(beam.tolist())
#             self.beam_positions_denorm.append(configurations) 
#             self.beam_targets_denorm.append(beam_predictions)

#             if any(success): break      

#         flattened_positions = list(chain.from_iterable(G.beam_positions_denorm))
#         flattened_predictions = list(chain.from_iterable(G.beam_targets_denorm))
#         best = np.argsort(abs(np.array(flattened_predictions)-self.desired))[0]

#         self.best_position = flattened_positions[best]
#         self.best_prediction = flattened_predictions[best]

#         return self.beam_positions_denorm, self.beam_targets_denorm, self.best_position, self.best_prediction


#     def stochastic(self, num_candidates = 5, starting_point = np.array(list(constraints.values()))[:,5].tolist()) :
#         self.num_candidates = num_candidates
#         y_prime = self.desired / (target.max[0] - target.min[0])
#         tolerance = self.tolerance
#         final = None

#         x_i = [starting_point[i] / (feature.max[i] - feature.min[i]) for i in range(input_size)]  
#         x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], 
#                                 dtype = dtype)

#         self.stochastic_chosen = []
#         self.stochastic_predictions = []
#         self.stochastic_configurations = []

#         for step in range(self.steps):
#             configuration = feature.denormalize(x_prime)
#             predictions, gradients = self.predict_global(self.models, x = x_prime, y_prime = y_prime)
#             prediction_avg = sum(predictions)/len(predictions) 
#             gradient_avg = sum(gradients)/len(gradients)

#             candidates = np.argsort(abs(gradient_avg))[::-1][:self.num_candidates] #[::-1]
#             chosen = random.choice(candidates)

#             adjustment = list(np.repeat(0,len(self.unit_by_feature)))

#             if gradient_avg[chosen] >= 0: adjustment[chosen] += self.unit_by_feature[chosen]
#             else: adjustment[chosen] -= self.unit_by_feature[chosen]
#             adjustment = np.array(adjustment)

#             if prediction_avg > y_prime: x_prime -= adjustment 
#             elif prediction_avg < y_prime: x_prime += adjustment
#             else: pass
#             x_prime = super().bounding(x_prime)

#             prediction_original = target.denormalize(prediction_avg)
#             prediction_original = prediction_original[0]


#             if configuration_show_steps and step % 10 == 0 and step != 0:
#                 print(f"Step {step} Target : {self.desired}, Prediction : {prediction_original}")

#             self.stochastic_chosen.append(chosen)    
#             self.stochastic_predictions.append(prediction_original)
#             self.stochastic_configurations.append(configuration)

#             if abs(prediction_avg - y_prime) < tolerance: break
#         best = np.argsort(np.array(self.stochastic_predictions)-self.desired)[0]

#         self.stochastic_best_position = self.stochastic_configurations[best]
#         self.stochastic_best_prediction = self.stochastic_predictions[best]

#         return self.stochastic_configurations, self.stochastic_predictions, self.stochastic_best_position, self.stochastic_best_prediction


#     def best_one(self, starting_point = np.array(list(constraints.values()))[:,5].tolist(), escape = True) :
#         y_prime = self.desired / (target.max[0] - target.min[0])
#         tolerance = self.tolerance

#         x_i = [starting_point[i] / (feature.max[i] - feature.min[i]) for i in range(input_size)]  
#         x_prime = torch.tensor([x_i[i] - (x_i[i] % self.unit_by_feature[i]) for i in range(len(self.unit_by_feature))], 
#                                 dtype = dtype)

#         self.best_one_chosen = []
#         self.best_one_predictions = []
#         self.best_one_configurations = []

#         avoid = []
#         memory = []
#         memory_size = 5
#         previous = None
#         for step in range(self.steps):
#             configuration = feature.denormalize(x_prime)
#             predictions, gradients = self.predict_global(self.models, x = x_prime, y_prime = y_prime)
#             prediction_avg = sum(predictions)/len(predictions) 
#             gradient_avg = sum(gradients)/len(gradients)

#             if escape :
#                 candidates = [i for i in np.argsort(abs(gradient_avg))[::-1] if i not in avoid]
#             else :
#                 candidates = np.argsort(abs(gradient_avg))[::-1]
#                 #[::-1]
#             chosen = candidates[0]

#             adjustment = list(np.repeat(0,len(self.unit_by_feature)))

#             if gradient_avg[chosen] >= 0: adjustment[chosen] += self.unit_by_feature[chosen]
#             else: adjustment[chosen] -= self.unit_by_feature[chosen]
#             adjustment = np.array(adjustment)

#             if prediction_avg > y_prime: x_prime -= adjustment 
#             elif prediction_avg < y_prime: x_prime += adjustment
#             else: pass
#             x_prime = super().bounding(x_prime)

#             prediction_original = target.denormalize(prediction_avg)
#             prediction_original = prediction_original[0]


#             if configuration_show_steps and step % 10 == 0 and step != 0:
#                 print(f"Step {step} Target : {self.desired}, Prediction : {prediction_original}")

#             self.best_one_chosen.append(chosen)    
#             self.best_one_predictions.append(prediction_original)
#             self.best_one_configurations.append(configuration)
#             memory.append(prediction_original)
#             if len(memory) > memory_size : memory = memory[len(memory)-memory_size:]
#             if len(memory) == 5 and len(set(memory)) < 3  and previous == chosen: avoid.append(chosen)

#             if abs(prediction_avg - y_prime) < tolerance: break
#             if escape and len(avoid) == input_size : break
#             previous = chosen
#         best = np.argsort(np.array(self.best_one_predictions)-self.desired)[0]

#         self.best_one_best_position = self.best_one_configurations[best]
#         self.best_one_best_prediction = self.best_one_predictions[best]

#         return self.best_one_configurations, self.best_one_predictions, self.best_one_best_position, self.best_one_best_prediction


# def run(data, models, model_list, desired, starting_point, mode, modeling, strategy, tolerance, beam_width,
#         num_cadidates, escape, top_k, index, up, alternative):
    
    


#     output_size = 1
#     input_size = data.values.shape[1] - output_size
#     dtype = torch.float32
    
#     constraints = {'ATT1' : [5, min(data.iloc[:,0+1]), max(data.iloc[:,0+1]), int, 0, 150],
#                   'ATT2'  : [5, min(data.iloc[:,1+1]), max(data.iloc[:,1+1]), int, 0, 25],
#                   'ATT3'  : [10, min(data.iloc[:,2+1]), max(data.iloc[:,2+1]), int, -1, 40],
#                   'ATT4'  : [0.5, min(data.iloc[:,3+1]), max(data.iloc[:,3+1]), float, 1, 1],
#                   'ATT5'  : [5, min(data.iloc[:,4+1]), max(data.iloc[:,4+1]), int, 0, 120],
#                   'ATT6'  : [5, min(data.iloc[:,5+1]), max(data.iloc[:,5+1]), int, 0, 250],
#                   'ATT7'  : [5, min(data.iloc[:,6+1]), max(data.iloc[:,6+1]), int, 0, 10],
#                   'ATT8'  : [5, min(data.iloc[:,7+1]), max(data.iloc[:,7+1]), int, 0, 25],
#                   'ATT9'  : [5, min(data.iloc[:,8+1]), max(data.iloc[:,8+1]), int, 0, 25],
#                   'ATT10' : [60, min(data.iloc[:,9+1]), max(data.iloc[:,9+1]), int, -1, 900],
#                   'ATT11' : [0.05, min(data.iloc[:,10+1]), max(data.iloc[:,10+1]), float, 2, 0.25],
#                   'ATT12' : [1, min(data.iloc[:,11+1]), max(data.iloc[:,11+1]), int, 0, 2],
#                   'ATT13' : [5, min(data.iloc[:,12+1]), max(data.iloc[:,12+1]), int, 0, 100],
#                   'ATT14' : [60, min(data.iloc[:,13+1]), max(data.iloc[:,13+1]), int, -1, 1800],
#                   'ATT15' : [1000, min(data.iloc[:,14+1]), max(data.iloc[:,14+1]), int, 0, 2000]}

#     starting_point = pd.DataFrame(np.array(starting_point).reshape(1,-1), columns = [f'ATT{i+1}' for i in range(input_size)])
#     erase = []
#     for i in range(input_size):
#         if min(data.iloc[:,i+1]) == max(data.iloc[:,i+1]):
#             erase.append(f"att{i+1}")
#             del constraints[f'ATT{i+1}']
#             del starting_point[f'ATT{i+1}']
                               
    
#     starting_point = starting_point.values.reshape(-1).tolist()
#     data = data.drop(columns=erase)
#     input_size = data.values.shape[1] - output_size    
    
    
                
    

#     target = MinMaxScaling(data['Target'])
#     feature = MinMaxScaling(data[[column for column in data.columns if column != 'Target']])
#     if models == None:
#         models = []
#         for model in model_list:
#             x = modeller(model)
#             models.append(x)
#         models, training_losses = train(feature.data, target.data)
#     else : training_losses = []
        
#     configurations, predictions, best_config, best_pred = None, None, None, None
#     if mode == 'global' :
#         G = GlobalMode(desired = desired, models = models, modeling = modeling, strategy = strategy)
#         if strategy == 'beam':
#             configurations, predictions, best_config, best_pred = G.beam(starting_point = starting_point,
#                                                                  beam_width = beam_width)
            
#         elif strategy == 'stochastic':
#             configurations, predictions, best_config, best_pred = G.stochastic(starting_point = starting_point,
#                                                                  num_candidates = num_candidates)
            
#         elif strategy == 'best_one':
#              configurations, predictions, best_config, best_pred = G.best_one(starting_point = starting_point, 
#                                                                               escape = escape)
        
#     elif mode == 'local':
#         L = LocalMode(desired = desired, models = models, modeling = modeling, strategy = strategy)
#         if strategy == 'exhaustive':
#             configurations, predictions, best_config, best_pred = L.exhaustive(starting_point = starting_point,
#                                                                                 alternative = alternative, top_k = top_k)
#         elif strategy == 'manual' :
#             configurations, predictions, best_config, best_pred = L.manual(starting_point = starting_point, index = index, up = up)
        
        
#     return models, training_losses, configurations, predictions, best_config, best_pred