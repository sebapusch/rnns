import numpy as np

class RNN:
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 hidden_state_size: int,
                 hidden_state: None | np.ndarray = None
                ) -> None:
        self.weights_hx = np.random.random(size=(hidden_state_size, input_size + 1)) # bias term
        self.weights_hh = np.random.random(size=(hidden_state_size, hidden_state_size + 1))
        self.weights_oh = np.random.random(size=(output_size, hidden_state_size + 1))

        self.initial_state = np.zeros_like(hidden_state_size) if hidden_state is None else hidden_state.copy()

        self.reset_hidden_state()

    def reset_hidden_state(self) -> None:
        self.hidden_state = self.initial_state.copy()

    def step(self, inp: np.ndarray) -> np.ndarray:
        inp = np.concatenate((inp, [1]))

        hidden_state = np.concatenate((self.hidden_state, [1]))
        self.hidden_state = self.weights_hh @ hidden_state.T + self.weights_hx @ inp.T
        hidden_state = np.concatenate((self.hidden_state, [1]))

        return self._sigmoid(self.weights_oh @ hidden_state.T)
    
    def _sigmoid(self, y: np.ndarray):
        return 1 / (1 + np.pow(np.e, -y))
