import numpy as np

from rnn.rnn import RNN

from helpers import bce, Adam

from tqdm import tqdm

def compute_rnn_loss(
        rnn: RNN,
        X: np.ndarray,
        Y: np.ndarray,
        S: np.ndarray,
) -> float:
    total_loss = 0.0

    for s in range(len(Y)):
        h = np.zeros(shape=(rnn.hidden_state_size,))
        z = np.zeros(1)
        for t in range(S[s]):
            z, h = rnn.step(X[s][t], h)
        
        total_loss += bce(z, Y[s])

    return float(total_loss / len(Y))

def compute_rnn_balance_accuracy(
        rnn: RNN,
        X: np.ndarray,
        Y: np.ndarray,
        S: np.ndarray,
        threshold: float = 0.5,
) -> float:
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for s in range(len(Y)):
        h = np.zeros(shape=(rnn.hidden_state_size,))
        z = np.zeros(1)
        for t in range(S[s]):
            z, h = rnn.step(X[s][t], h)
        
        prediction = int(z > threshold)
        actual = Y[s]

        if prediction == 1 and actual == 1:
            true_positives += 1
        elif prediction == 0 and actual == 0:
            true_negatives += 1
        elif prediction == 1 and actual == 0:
            false_positives += 1
        elif prediction == 0 and actual == 1:
            false_negatives += 1

    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

    balanced_accuracy = (sensitivity + specificity) / 2

    return balanced_accuracy


def train_rnn_classifier(rnn: RNN, X: np.ndarray, Y: np.ndarray, S: np.ndarray, epochs: int, lr: float, callback = None, epoch_callback = None) -> RNN:
    optimizer = Adam(lr)
    
    for e in range(epochs):
        print(f'Starting epoch {e}')
        total_loss = 0

        for s in tqdm(range(len(Y))):
            hidden_states = np.zeros(shape=(S[s], rnn.hidden_state_size ), dtype=np.float64)

            h = np.zeros(shape=(rnn.hidden_state_size,))
            z = np.zeros(1)
            for t in range(S[s]):
                z, h = rnn.step(X[s][t], h)
                hidden_states[t] = h

            total_loss += bce(z, Y[s])

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

            if np.isnan(grad_Whh).any() or np.isnan(grad_Whx).any() or np.isnan(grad_Woh).any():
                print('NaN detected in gradients, skipping update...')
                continue

            optimizer.update(rnn.W_oh, np.clip(grad_Woh.mean(axis=0), -clip_value, clip_value), 'Woh')
            optimizer.update(rnn.W_hh, np.clip(grad_Whh.mean(axis=0), -clip_value, clip_value), 'Whh')
            optimizer.update(rnn.W_hx, np.clip(grad_Whx.mean(axis=0), -clip_value, clip_value), 'Whx')
            # rnn.W_oh -= lr * grad_Woh
            # rnn.W_hh -= lr * grad_Whh
            # rnn.W_hx -= lr * grad_Whx

        print(f'loss at epoch {e}: {total_loss / len(Y)}')
        if epoch_callback is not None:
            epoch_callback(rnn, e, total_loss / len(Y))

    return rnn