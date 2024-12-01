# early_stopping.py
import numpy as np
import torch

class EarlyStopping:
    """얼리 스탑핑을 위한 클래스"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'얼리 스탑핑 카운터: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''검증 손실이 개선되었을 때 모델을 저장합니다.'''
        if self.verbose:
            print(f'검증 손실이 감소했습니다 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 모델을 저장합니다.')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
