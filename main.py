from rnn import RNN

import numpy as np


HIDDEN_STATE_SIZE = 10
LR = 1e-1
EPOCHS = 100

def grad_tanh(grad_h, h):
    return grad_h * (1 - h ** 2)    

def main():
    rnn = RNN(3, 1, HIDDEN_STATE_SIZE)

    sample_x = np.array([
        [
            [1, 2, 3], # the
            [1, 2, 3], # movie
            [1, 2, 3]  # good 
        ],
        [
            [1, 2, 3],# the
            [1, 2, 3],# movie
            [1, 2, 3] # shit 
        ]
    ])

    sample_y = np.array([
        1,
        1,
    ])

    SEQUENCE_LENGTH = len(sample_x[0])

    for e in range(EPOCHS):
        total_loss = 0

        for s in range(len(sample_y)):
            rnn.reset_hidden_state()

            # includes bias term
            hidden_states = np.zeros(shape=(SEQUENCE_LENGTH, HIDDEN_STATE_SIZE + 1), dtype=np.float64)

            z = np.zeros(1) 
            for t in range(SEQUENCE_LENGTH):
                z, h = rnn.step(sample_x[s][t])
                hidden_states[t] = h

            total_loss += -(sample_y[s] * np.log(z) + (1 - sample_y[s]) * np.log(1 - z))

            grad_o = z - sample_y[s]

            grad_Woh = np.outer(grad_o, hidden_states[-1].T)

            grad_Whh = np.zeros_like(rnn.W_hh, dtype=np.float64)
            grad_Whx = np.zeros_like(rnn.W_hx, dtype=np.float64)

            grad_h = np.zeros(shape=(SEQUENCE_LENGTH, HIDDEN_STATE_SIZE), dtype=np.float64)
            grad_a = np.zeros(shape=(SEQUENCE_LENGTH, HIDDEN_STATE_SIZE), dtype=np.float64)

            grad_h[-1] = rnn.W_oh[:,:-1].T @ grad_o
            grad_a[-1] = grad_tanh(grad_h[-1], hidden_states[-1][:-1])

            for t in reversed(range(SEQUENCE_LENGTH - 1)):
                grad_h[t] = rnn.W_hh[:,:-1].T @ grad_a[t + 1]
                grad_a[t] = grad_tanh(grad_h[t], hidden_states[t][:-1])
            
            grad_Whh = np.outer(grad_a[0], np.append(rnn.initial_state, 1).T)
            grad_Whx = np.outer(grad_a[0], np.append(sample_x[s][0], 1).T)

            for t in reversed(range(1, SEQUENCE_LENGTH)):
                grad_Whh += np.outer(grad_a[t], hidden_states[t - 1].T)
                grad_Whx += np.outer(grad_a[t], np.append(sample_x[s][t], 1).T)
            
            rnn.W_oh -= LR * grad_Woh
            rnn.W_hh -= LR * grad_Whh
            rnn.W_hx -= LR * grad_Whx

        print(f'loss at epoch {e}: {total_loss / len(sample_y)}')

    z = np.zeros(1) 
    for t in range(SEQUENCE_LENGTH):
        z, h = rnn.step(sample_x[0][t])

    print(z)

if __name__ == "__main__":
    main()
