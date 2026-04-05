"""
GPU-accelerated ML model for price prediction.
Uses PyTorch LSTM on GPU (T4/A100) for training, CPU fallback for inference.
The agent can import and use this in strategy/train.py.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PriceLSTM(nn.Module):
    """LSTM model for predicting price direction."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # 3 classes: down, flat, up
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last timestep
        return out


def prepare_features(df: pd.DataFrame, lookback: int = 60) -> tuple:
    """
    Prepare feature matrix from OHLCV data for LSTM training.
    Returns (X, y) tensors ready for GPU training.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    # Normalized features
    returns = np.diff(close) / close[:-1]
    high_low_ratio = (high[1:] - low[1:]) / close[:-1]
    volume_change = np.diff(volume) / (volume[:-1] + 1e-8)

    # Stack features
    features = np.column_stack([returns, high_low_ratio, volume_change])

    # Remove NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        # Target: next bar direction (0=down, 1=flat, 2=up)
        ret = returns[i] if i < len(returns) else 0
        if ret > 0.001:
            y.append(2)  # up
        elif ret < -0.001:
            y.append(0)  # down
        else:
            y.append(1)  # flat

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return torch.tensor(X).to(device), torch.tensor(y).to(device)


def train_model(df: pd.DataFrame, lookback: int = 60, epochs: int = 20, batch_size: int = 512, lr: float = 0.001, max_train_bars: int = 200_000) -> PriceLSTM:
    """
    Train LSTM model on GPU. Returns trained model.
    Training time: ~30s on A100 for 100K bars, ~2min for 200K bars.
    max_train_bars: cap training data for huge datasets (uses most recent bars).
    """
    # Cap training data for very large datasets — use most recent bars for relevance
    if len(df) > max_train_bars:
        df = df.iloc[-max_train_bars:]
    X, y = prepare_features(df, lookback)

    if len(X) < 100:
        return None

    # Train/val split (80/20)
    split = int(len(X) * 0.8)
    train_dataset = TensorDataset(X[:split], y[:split])
    val_dataset = TensorDataset(X[split:], y[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    input_size = X.shape[2]  # number of features
    model = PriceLSTM(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_acc = correct / total if total > 0 else 0
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print(f"  ML Model trained on GPU ({device}): {len(X)} samples, val_acc={best_val_acc:.4f}")
    return model


def predict_signals(model: PriceLSTM, df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """
    Generate trading signals from trained model.
    Returns Series: 1=long, -1=short, 0=flat
    """
    if model is None:
        return pd.Series(0, index=df.index)

    X, _ = prepare_features(df, lookback)

    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predictions = torch.max(outputs, 1)

    predictions = predictions.cpu().numpy()

    # Map predictions to signals: 0=down->short(-1), 1=flat->0, 2=up->long(1)
    signal_map = {0: -1, 1: 0, 2: 1}
    signals = np.array([signal_map[p] for p in predictions])

    # Pad the beginning (lookback period has no predictions)
    full_signals = np.zeros(len(df))
    full_signals[lookback + 1:lookback + 1 + len(signals)] = signals

    return pd.Series(full_signals, index=df.index)
