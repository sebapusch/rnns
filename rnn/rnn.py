import numpy as np

from typing import Tuple

from helpers import init_random, sigmoid

class RNN:
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 hidden_state_size: int,
                ) -> None:
        # include bias terms
        self.W_hx = init_random((hidden_state_size, input_size + 1))
        self.W_hh = init_random((hidden_state_size, hidden_state_size + 1))
        self.W_oh = init_random((output_size, hidden_state_size + 1))

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

        return sigmoid(self.W_oh @ np.append(h_next, 1)), h_next
    
    def save(self, filepath: str) -> None:
        np.savez(
            filepath,
            W_hx=self.W_hx,
            W_hh=self.W_hh,
            W_oh=self.W_oh,
        )

    @staticmethod
    def load(filepath: str) -> 'RNN':
        data = np.load(filepath)
        rnn = RNN(
            input_size=data['W_hx'].shape[1] - 1,
            output_size=data['W_oh'].shape[0],
            hidden_state_size=data['W_hh'].shape[0],
        )
        rnn.W_hx = data['W_hx']
        rnn.W_hh = data['W_hh']
        rnn.W_oh = data['W_oh']
        return rnn

        

