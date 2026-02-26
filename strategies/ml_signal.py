"""Simple ML‑based signal strategy using logistic regression.

Uses lagged returns and basic technical features to predict next‑day direction.
Implemented with pure numpy (no scikit‑learn dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy, SignalFrame


class MLSignal(Strategy):
    """Logistic regression on engineered features (pure‑numpy implementation)."""

    name = "ml_signal"

    def __init__(
        self,
        train_window: int = 252,
        retrain_every: int = 21,
        lags: int = 5,
        learning_rate: float = 0.01,
        n_iterations: int = 500,
    ) -> None:
        self.train_window = train_window
        self.retrain_every = retrain_every
        self.lags = lags
        self.lr = learning_rate
        self.n_iter = n_iterations

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, market_data: pd.DataFrame) -> SignalFrame:
        close = market_data["close"]
        features = self._build_features(market_data)
        target = (close.pct_change().shift(-1) > 0).astype(float)

        n = len(close)
        predictions = pd.Series(0.0, index=close.index)
        probabilities = pd.Series(0.5, index=close.index)

        weights = None
        last_train = 0

        for i in range(self.train_window, n):
            # Retrain periodically
            if weights is None or (i - last_train) >= self.retrain_every:
                train_start = max(0, i - self.train_window)
                X_train = features.iloc[train_start:i].values
                y_train = target.iloc[train_start:i].values

                # Remove NaN rows
                mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
                X_train = X_train[mask]
                y_train = y_train[mask]

                if len(X_train) > 10:
                    weights, bias = self._train_logistic(X_train, y_train)
                    last_train = i

            if weights is not None:
                x_i = features.iloc[i].values.reshape(1, -1)
                if not np.isnan(x_i).any():
                    prob = self._sigmoid(x_i @ weights + bias)[0]
                    probabilities.iloc[i] = prob
                    predictions.iloc[i] = 1.0 if prob > 0.5 else -1.0

        # Position size = confidence (distance from 0.5)
        confidence = (probabilities - 0.5).abs() * 2  # 0‑1
        position_size = confidence.clip(lower=0.1, upper=1.0).fillna(0.0)

        return pd.DataFrame(
            {"signal": predictions, "position_size": position_size},
            index=close.index,
        )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        close = market_data["close"]
        ret = close.pct_change()

        feats: dict[str, pd.Series] = {}

        # Lagged returns
        for lag in range(1, self.lags + 1):
            feats[f"ret_lag_{lag}"] = ret.shift(lag)

        # Rolling statistics
        feats["vol_5"] = ret.rolling(5).std()
        feats["vol_20"] = ret.rolling(20).std()
        feats["mom_5"] = close.pct_change(5)
        feats["mom_20"] = close.pct_change(20)

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0.0, float("nan"))
        feats["rsi"] = 100.0 - 100.0 / (1.0 + rs)

        # SMA ratio
        feats["sma_ratio"] = close / close.rolling(20).mean()

        return pd.DataFrame(feats, index=close.index)

    # ------------------------------------------------------------------
    # Numpy logistic regression
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _train_logistic(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Train logistic regression via gradient descent."""
        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - mean) / std

        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0.0

        for _ in range(self.n_iter):
            z = X @ weights + bias
            pred = self._sigmoid(z)

            dw = (1 / n_samples) * (X.T @ (pred - y))
            db = (1 / n_samples) * np.sum(pred - y)

            weights -= self.lr * dw
            bias -= self.lr * db

        return weights, bias
