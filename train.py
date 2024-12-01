import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error

def train_pytorch_model(model, X_train, y_train, X_val=None, y_val=None, num_epochs=100, batch_size=32, learning_rate=0.001, early_stopping=None):
    # PyTorch 데이터셋 및 데이터로더 생성
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                   torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_val is not None:
        val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32),
                                     torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        # 검증 단계
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)

            # Early Stopping 체크
            if early_stopping:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print(f"Epoch {epoch}: Early stopping")
                    break
        else:
            val_loss = None

    return model, train_loss, val_loss


def train_sklearn_model(model, X_train, y_train, X_val=None, y_val=None):
    # scikit-learn 모델 학습
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    train_loss = mean_squared_error(y_train, train_predictions)

    if X_val is not None:
        val_predictions = model.predict(X_val)
        val_loss = mean_squared_error(y_val, val_predictions)
    else:
        val_loss = None

    return model, train_loss, val_loss
