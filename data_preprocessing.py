import numpy as np
import pandas as pd
import torch

import numpy as np

class MinMaxScaling:
    """
    data: 스케일링 대상 (DataFrame 또는 ndarray)
    constraints: 메타데이터 기반으로 만든 컬럼별 [단위, min, max, dtype, round_digits] 사전
                 -> 여기서는 'column' 개념이 없을 수도 있으므로, 인덱스 순서대로 사용하거나
                    혹은 data가 DataFrame이면 columns 매핑이 필요.
    dtype: torch.float32 등
    """

    def __init__(self, data, constraints, dtype=torch.float32):
        self.constraints = constraints  # 외부에서 받은 constraints
        self.dtype = dtype

        self.max, self.min, self.range = [], [], []
        self.data = pd.DataFrame([])

        # data가 DataFrame이면 .values, 1D면 reshape 등
        if isinstance(data, pd.DataFrame):
            data = data.values
        data = data.reshape(-1, 1) if len(data.shape) == 1 else data

        epsilon = 2
        for i in range(data.shape[1]):
            col_data = data[:, i]
            max_ = col_data.max()
            min_ = col_data.min()
            # if max_ == min_:
            #     max_ *= epsilon

            self.max.append(max_)
            self.min.append(min_)
            self.range.append(max_ - min_)
            # scaled_col = (col_data-min_)/ (max_ - min_)
            scaled_col = (col_data)/ (max_ - min_)
            self.data = pd.concat([self.data, pd.DataFrame(scaled_col)], axis=1)

        # 최종적으로 torch.tensor에 저장
        self.data = torch.tensor(self.data.values, dtype=self.dtype)

    def denormalize(self, data):
        """
        data: 스케일링된 1D/2D numpy array or torch.Tensor
        -> 각 index별로 self.max[i], self.min[i]를 사용
        -> 그리고 self.constraints를 이용해 round 자릿수 결정
        """
        # data가 torch.Tensor이면 numpy로 변환
        data = data.detach().numpy() if isinstance(data, torch.Tensor) else data

        # 만약 columns(컬럼명) 단위 매핑이 필요한 경우, 별도 로직 필요.
        # 지금 예시는 "인덱스 순서"로만 constraints를 매핑한다고 가정(주의).
        #  (즉, constraints가 key가 아니라 list 형태로 관리된다고 치거나,
        #   혹은 인덱스 순서대로 [att1, att2, ...] 라고 가정)

        new_data = []
        for i, element in enumerate(data):
            # 역정규화
            element = element * (self.max[i] - self.min[i])  + self.min[i]  # + min[i]? (기존 코드엔 +min이 생략되어있을 수도 있음)
            # round 자릿수
            # ex) round_digits = self.constraints[some_key][4] ...
            # 여기서는 단순히 i번째 constraint를 가져온다고 가정.
            # constraints = { 'att1': [...], 'att2': [...] } 이면 list로 전환해서 i번째를 찾거나
            # index가 'col_name'과 매칭되는지 별도 관리가 필요함
            # 예: 
            # constraints_list = list(self.constraints.values()) 
            # round_digits = constraints_list[i][4]
            # element = round(element, round_digits)

            # (혹은, 만약 인덱스/컬럼 매핑을 명시적으로 해야 한다면, 
            #  init에서 data.columns 순서대로 constraints를 저장한 리스트를 만들어둬야 합니다.)
            # 아래는 "i번째 constraint"라고 가정한 예시:
            constraints_list = list(self.constraints.values())
            round_digits = constraints_list[i][4]
            
            element = round(element, round_digits)
            new_data.append(element)
        return np.array(new_data).flatten()