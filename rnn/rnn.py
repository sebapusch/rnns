import numpy as np

from typing import Tuple

class RNN:
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 hidden_state_size: int,
                ) -> None:
        
        # include bias terms
        self.W_hx = self._init_random((hidden_state_size, input_size + 1))
        self.W_hh = self._init_random((hidden_state_size, hidden_state_size + 1))
        self.W_oh = self._init_random((output_size, hidden_state_size + 1))

    @property
    def hidden_state_size(self) -> int:
        return self.W_hh.shape[0]

    def step(self, x: np.ndarray, h: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        if h is None:
            h = np.zeros(shape=self.W_hh.shape[1])
        else:
            h = np.append(h, 1)

        assert h is not None
            
        x = np.append(x, 1)

        h_next = self.W_hh @ h.T + self.W_hx @ x.T

        return self._sigmoid(self.W_oh @ np.append(h_next, 1).T), h_next
    
    def _sigmoid(self, y: np.ndarray):
        return 1 / (1 + np.exp(-y))
    
    def _init_random(self, size, scale: float = 1e-3) -> np.ndarray:
        return np.random.random(size=size).astype(np.float64) * scale 
        

