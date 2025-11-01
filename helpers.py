import numpy as np
import matplotlib.pyplot as plt

def init_random(size, scale: float = 1e-3) -> np.ndarray:
    return np.random.random(size=size).astype(np.float64) * scale

def xavier_uniform(shape: tuple) -> np.ndarray:
    fan_in = shape[1]
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, shape)

def orthogonal(shape: tuple) -> np.ndarray:
    W = np.random.randn(*shape)
    u, _, v = np.linalg.svd(W, full_matrices=False)
    return u if u.shape == shape else v


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def bce(z: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    z = np.clip(z, eps, 1 - eps)
    return -(y * np.log(z) + (1 - y) * np.log(1 - z))


def add_bias_col(x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1))])

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state = {}   # maps parameter names to (m, v, t)

    def update(self, W: np.ndarray, grad: np.ndarray, name: str):
        # Initialize state for this parameter if new
        if name not in self.state:
            self.state[name] = {
                "m": np.zeros_like(W),
                "v": np.zeros_like(W),
                "t": 0
            }

        s = self.state[name]

        s["t"] += 1
        s["m"] = self.beta1 * s["m"] + (1 - self.beta1) * grad
        s["v"] = self.beta2 * s["v"] + (1 - self.beta2) * (grad * grad)

        m_hat = s["m"] / (1 - self.beta1 ** s["t"])
        v_hat = s["v"] / (1 - self.beta2 ** s["t"])

        W -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)



# PLOTTING
