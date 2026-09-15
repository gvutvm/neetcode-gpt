import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        h = x
        for i in range(len(weights)):
            h = h @ weights[i] + biases[i]  # Linear transformation
            if i < len(weights) - 1:
                h = np.maximum(0, h)         # ReLU on hidden layers only
        return np.round(h, 5)