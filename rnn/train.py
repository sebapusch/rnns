import numpy as np

from rnn.rnn import RNN

from helpers import bce, Adam

from tqdm import tqdm

def train_rnn(rnn: RNN, X: np.ndarray, Y: np.ndarray, S: np.ndarray, epochs: int, lr: float, callback = None) -> RNN:
    optimizer = Adam(lr)
    
    for e in tqdm(range(epochs)):
        total_loss = 0

        for s in tqdm(range(len(Y))):
            hidden_states = np.zeros(shape=(S[s], rnn.hidden_state_size ), dtype=np.float64)

            h = np.zeros(shape=(rnn.hidden_state_size,))
            z = np.zeros(1)
            for t in range(S[s]):
                z, h = rnn.step(X[s][t], h)
                hidden_states[t] = h

            total_loss += -bce(z, Y[s])

            grad_o = z - Y[s]

            grad_Woh = np.outer(grad_o, np.append(hidden_states[-1], 1))

            grad_Whh = np.zeros_like(rnn.W_hh, dtype=np.float64)
            grad_Whx = np.zeros_like(rnn.W_hx, dtype=np.float64)

            grad_h = np.zeros(shape=(S[s], rnn.hidden_state_size), dtype=np.float64)

            grad_a = np.zeros(shape=(S[s], rnn.hidden_state_size), dtype=np.float64)

            grad_h[-1] = rnn.W_oh[:,:-1].T @ grad_o
            grad_a[-1] = grad_h[-1] * (1 - hidden_states[-1] ** 2)

            for t in reversed(range(S[s] - 1)):
                grad_h[t] = rnn.W_hh[:,:-1].T @ grad_a[t + 1]
                grad_a[t] = grad_h[t] * (1 - hidden_states[t] ** 2)
                if callback is not None:
                    callback(grad_h[t], e, s, t)
            
            grad_Whh = np.outer(grad_a[0], np.zeros(shape=(rnn.hidden_state_size + 1,)))
            grad_Whx = np.outer(grad_a[0], np.append(X[s][0], 1))

            for t in reversed(range(1, S[s])):
                grad_Whh += np.outer(grad_a[t], np.append(hidden_states[t - 1], 1))
                grad_Whx += np.outer(grad_a[t], np.append(X[s][t], 1))
            
            clip_value = 5.0 

            optimizer.update(rnn.W_oh, np.clip(grad_Woh.mean(axis=0), -clip_value, clip_value), 'Woh')
            optimizer.update(rnn.W_hh, np.clip(grad_Whh.mean(axis=0), -clip_value, clip_value), 'Whh')
            optimizer.update(rnn.W_hx, np.clip(grad_Whx.mean(axis=0), -clip_value, clip_value), 'Whx')
            # rnn.W_oh -= lr * grad_Woh
            # rnn.W_hh -= lr * grad_Whh
            # rnn.W_hx -= lr * grad_Whx

        print(f'loss at epoch {e}: {total_loss / len(Y)}')

    return rnn