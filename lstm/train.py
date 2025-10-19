import numpy as np

from lstm.lstm import LSTMClassifier

from helpers import bce

def train_lstm_classifier(lstm: LSTMClassifier, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float) -> LSTMClassifier:
    T = len(X[0])
    input_size = len(X[0][0])

    for e in range(epochs):
        total_loss = 0

        for s in range(len(Y)):
            c = None
            h = None
            z = np.zeros(0)

            h_hist = np.zeros((T, lstm.hidden_state_size))
            o_hist = np.zeros((T, lstm.hidden_state_size))
            f_hist = np.zeros((T, lstm.hidden_state_size))
            i_hist = np.zeros((T, lstm.hidden_state_size))
            c_hist = np.zeros((T, lstm.hidden_state_size))
            candidate_hist = np.zeros((T, lstm.hidden_state_size))

            for t in range(T):
                z, c, h = lstm.step(X[s][t], c, h)

                assert lstm.activations is not None

                c_hist[t] = c
                h_hist[t] = h
                o_hist[t] = lstm.activations['o']
                f_hist[t] = lstm.activations['f']
                i_hist[t] = lstm.activations['i']
                candidate_hist[t] = lstm.activations['candidate']

            total_loss += bce(z, Y[s])

            grad_out = z - Y[s]

            prev_grad_c = 0
            prev_grad_h = lstm.W_z[:,:-1].T @ grad_out

            grad_Wf = np.zeros_like(lstm.W_f)
            grad_Wo = np.zeros_like(lstm.W_o)
            grad_Wi = np.zeros_like(lstm.W_i)
            grad_Wc = np.zeros_like(lstm.W_c)
            grad_Wz = np.outer(grad_out, np.append(h_hist[-1], 1))


            for t in reversed(range(T)):
                prev_c = np.zeros(lstm.hidden_state_size) if t == 0 else c_hist[t - 1]
                prev_h = np.zeros(lstm.hidden_state_size) if t == 0 else h_hist[t - 1]

                grad_h = prev_grad_h

                grad_c = prev_grad_c + grad_h * o_hist[t] * (1 - np.tanh(c_hist[t]) ** 2)
                grad_f = grad_c * prev_c
                grad_o = grad_h * np.tanh(c_hist[t])
                grad_i = grad_c * candidate_hist[t]
                grad_candidate = grad_c * i_hist[t]

                grad_a_f = grad_f * f_hist[t] * (1 - f_hist[t])
                grad_a_i = grad_i * i_hist[t] * (1 - i_hist[t])
                grad_a_o = grad_o * o_hist[t] * (1 - o_hist[t])
                grad_a_c = grad_candidate * (1 - candidate_hist[t] ** 2)

                u = (lstm.W_f.T @ grad_a_f + 
                     lstm.W_i.T @ grad_a_i + 
                     lstm.W_o.T @ grad_a_o + 
                     lstm.W_c.T @ grad_a_c)

                prev_grad_c = grad_c * f_hist[t]
                prev_grad_h = u[input_size:-1]

                x_t = np.concat((X[s][t], prev_h, [1]))

                grad_Wf += np.outer(grad_a_f, x_t)
                grad_Wi += np.outer(grad_a_i, x_t)
                grad_Wo += np.outer(grad_a_o, x_t)
                grad_Wc += np.outer(grad_a_c, x_t)

            print(lstm.W_z.shape, grad_Wz.shape)

            lstm.W_f -= lr * grad_Wf
            lstm.W_i -= lr * grad_Wi
            lstm.W_o -= lr * grad_Wo
            lstm.W_c -= lr * grad_Wc
            lstm.W_z -= lr * grad_Wz
        
        print(f'loss at epoch {e}: {total_loss / len(Y)}')



    return lstm