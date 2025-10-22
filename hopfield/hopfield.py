import numpy as np

from typing import Callable

class Hopfield:
    def __init__(self, size: int) -> None:
        self.size = size
        self.W = None
    
    def step(self, state: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError('Model is not trained')
        
        next_state = state.copy()
        
        i = np.random.choice(self.size)

        updated = np.sign(self.W[i] @ state)
        if updated != 0:
            next_state[i] = updated

        return next_state
    
    def recall(self, state: np.ndarray, callback: Callable | None = None) -> np.ndarray:
        has_callback = callback is not None
        
        next_state = self.step(state)

        same_state = 0
        while same_state < 50:
            state = next_state
            if has_callback:
                callback(state)

            next_state = self.step(state)

            if (next_state != same_state).sum() < 10:
                same_state += 1
            else:
                same_state = 0

        return state
    
    def train_implicit(self, memories: list[np.ndarray]):
        identity = np.identity(self.size)

        self.W = np.zeros((self.size, self.size,))

        for m in memories:
            self.W += np.outer(m, m) - identity

        self.W /= self.size