import numpy as np

from typing import Tuple

class RNN:
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 hidden_state_size: int,
                 hidden_state: None | np.ndarray = None
                ) -> None:
        
        # include bias terms
        self.W_hx = self._init_random((hidden_state_size, input_size + 1))
        self.W_hh = self._init_random((hidden_state_size, hidden_state_size + 1))
        self.W_oh = self._init_random((output_size, hidden_state_size + 1))

        self.initial_state = np.zeros(shape=(hidden_state_size,)) if hidden_state is None else hidden_state.copy()

        self.reset_hidden_state()

    def reset_hidden_state(self) -> None:
        self.h = self.initial_state.copy()

    def step(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.append(x, 1)
        h = np.append(self.h, 1)

        self.h = self.W_hh @ h.T + self.W_hx @ x.T

        h_next = np.append(self.h, 1)

        return self._sigmoid(self.W_oh @ h_next.T), h_next
    
    def _sigmoid(self, y: np.ndarray):
        return 1 / (1 + np.exp(-y))
    
    def _init_random(self, size, scale: float = 1e-3) -> np.ndarray:
        return np.random.random(size=size).astype(np.float64) * scale 
        

