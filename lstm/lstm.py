import numpy as np

from helpers import init_random, sigmoid

class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.W_f = init_random((hidden_size, input_size + hidden_size + 1))
        self.W_i = init_random((hidden_size, input_size + hidden_size + 1))
        self.W_o = init_random((hidden_size, input_size + hidden_size + 1))
        self.W_c = init_random((hidden_size, input_size + hidden_size + 1))

        self.activations = None

    @property
    def hidden_state_size(self) -> int:
        return self.W_f.shape[0]

    def step(self, x: np.ndarray, c: np.ndarray | None = None, h: np.ndarray | None = None) -> list[np.ndarray]:
        if c is None:
            c = np.zeros(self.W_c.shape[0])
        if h is None:
            h = np.zeros(self.W_c.shape[0])

        assert h is not None
        assert c is not None

        u = np.concat((x, h, [1]))
        
        f = sigmoid(self.W_f @ u)
        i = sigmoid(self.W_i @ u)
        o = sigmoid(self.W_o @ u)

        candidate = np.tanh(self.W_c @ u)

        c_next = c + f * c + i * candidate
        h_next = o * np.tanh(c_next)

        self.activations = {
            'f': f,
            'i': i,
            'o': o,
            'candidate': candidate
        }

        return [c_next, h_next]

class LSTMClassifier(LSTMCell):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__(input_size, hidden_size)

        self.W_z = init_random((output_size, hidden_size + 1))

    def step(self, x: np.ndarray, c: np.ndarray | None = None, h: np.ndarray | None = None) -> list[np.ndarray]:
        c_next, h_next = super().step(x, c, h)

        z = sigmoid(self.W_z @ np.append(h_next, 1))

        return [z, c_next, h_next]