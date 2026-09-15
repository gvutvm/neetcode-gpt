import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        n = X.shape[0]
        w = np.zeros(X.shape[1])
        b = 0.0

        for _ in range(epochs):
            # Forward pass
            y_hat = X @ w + b
            error = y_hat - y

            # Compute gradients of MSE loss
            dw = (2.0 / n) * (X.T @ error)
            db = (2.0 / n) * np.sum(error)

            # Update weights
            w = w - lr * dw
            b = b - lr * db

        return (np.round(w, 5), round(float(b), 5))