import numpy as np

from rnn.rnn import RNN

from helpers import bce

def train_rnn(rnn: RNN, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float) -> RNN:
    T = len(X[0])

    for e in range(epochs):
        total_loss = 0

        for s in range(len(Y)):
            hidden_states = np.zeros(shape=(T, rnn.hidden_state_size ), dtype=np.float64)

            h = np.zeros(shape=(rnn.hidden_state_size,))
            z = np.zeros(1) 
            for t in range(T):
                z, h = rnn.step(X[s][t], h)
                hidden_states[t] = h

            total_loss += -bce(z, Y[s])

            grad_o = z - Y[s]

            grad_Woh = np.outer(grad_o, np.append(hidden_states[-1], 1))

            grad_Whh = np.zeros_like(rnn.W_hh, dtype=np.float64)
            grad_Whx = np.zeros_like(rnn.W_hx, dtype=np.float64)

            grad_h = np.zeros(shape=(T, rnn.hidden_state_size), dtype=np.float64)
            grad_a = np.zeros(shape=(T, rnn.hidden_state_size), dtype=np.float64)

            grad_h[-1] = rnn.W_oh[:,:-1].T @ grad_o
            grad_a[-1] = grad_h[-1] * (1 - hidden_states[-1] ** 2)

            for t in reversed(range(T - 1)):
                grad_h[t] = rnn.W_hh[:,:-1].T @ grad_a[t + 1]
                grad_a[t] = grad_h[t] * (1 - hidden_states[t] ** 2)
            
            grad_Whh = np.outer(grad_a[0], np.zeros(shape=(rnn.hidden_state_size + 1,)))
            grad_Whx = np.outer(grad_a[0], np.append(X[s][0], 1))

            for t in reversed(range(1, T)):
                grad_Whh += np.outer(grad_a[t], np.append(hidden_states[t - 1], 1))
                grad_Whx += np.outer(grad_a[t], np.append(X[s][t], 1))
            
            rnn.W_oh -= lr * grad_Woh
            rnn.W_hh -= lr * grad_Whh
            rnn.W_hx -= lr * grad_Whx

        print(f'loss at epoch {e}: {total_loss / len(Y)}')

    return rnn