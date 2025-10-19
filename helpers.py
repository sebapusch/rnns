import numpy as np

def init_random(size, scale: float = 1e-3) -> np.ndarray:
    return np.random.random(size=size).astype(np.float64) * scale 

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def bce(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -(y * np.log(z) + (1 - y) * np.log(1 - z))
