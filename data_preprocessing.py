import numpy as np
import pandas as pd
import torch

import numpy as np

class MinMaxScaling :
        def __init__(self, data): #np.DataFrame
            self.max, self.min, self.range = [],[], []
            self.data = pd.DataFrame([])
            data = data.values.reshape(-1,1) if len(data.values.shape) == 1 else data.values

            epsilon = 2
            for i in range(data.shape[1]) :
                max_, min_ = max(data[:,i]), min(data[:,i])
                if max_ == min_ : max_ *= epsilon
                self.max.append(max_)
                self.min.append(min_)
                self.range.append(max_-min_)
                self.data = pd.concat([self.data, pd.DataFrame((data[:,i])/(max_-min_))],axis = 1)
            self.data = torch.tensor(self.data.values, dtype = dtype)

        def denormalize(self, data):
            data = data.detach().numpy() if isinstance(data, torch.Tensor) else data
            new_data = []
            for i, element in enumerate(data):
                element = (element * (self.max[i] - self.min[i])) 
                element = round(element, np.array(list(constraints.values()))[:,4][i])
                new_data.append(element)
            return new_data