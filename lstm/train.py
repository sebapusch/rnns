import numpy as np

from lstm.lstm import LSTMClassifier, IH, IO, IC, II, IF, ID
from tqdm import tqdm

from helpers import bce, add_bias_col, Adam


def compute_lstm_loss(
        lstm: LSTMClassifier, 
        X: np.ndarray, 
        Y: np.ndarray,
        S: np.ndarray,
) -> float:
    out, _ = lstm.output(X, S)

    return bce(out, Y[:,None]).mean()


def compute_lstm_accuracy(
        lstm: LSTMClassifier, 
        X: np.ndarray, 
        Y: np.ndarray,
        S: np.ndarray,
        threshold: float = 0.5,
) -> float:
    out, _ = lstm.output(X, S)

    correct = 0
    for s in range(len(Y)):
        if int(out[s] > threshold) == Y[s]:
            correct += 1

    return correct / len(Y)


def train_lstm_classifier_batched(
        lstm: LSTMClassifier, 
        X: np.ndarray, 
        Y: np.ndarray,
        S: np.ndarray,
        batch_size: int,
        epochs: int, 
        lr: float,
        debug_gradients: bool = False,
        epoch_callback = None
    ) -> LSTMClassifier:
    
    N = len(Y)                  # sample size
    T = len(X[0])               # sequence length
    E = len(X[0][0])            # embedding size
    B = batch_size              # batch size
    H = lstm.hidden_state_size  # hidden state size

    num_batches = N // batch_size

    optimizer = Adam(lr)

    for epoch in range(epochs):
        print(f'starting epoch {epoch + 1}/{epochs}')

        perm = np.random.permutation(N)
        X = X[perm]
        Y = Y[perm]
        S = S[perm]

        loss = 0
        
        print(f'batch progress ({num_batches}) ', end='', flush=True)
        for batch in range(num_batches):
            print('.', end='', flush=True)

            batch_start = batch * batch_size
            batch_end   = (batch + 1) * batch_size

            sequence_length = int(S[batch_start:batch_end].max())

            output, history = lstm.output(X[batch_start:batch_end], S[batch_start:batch_end], True)

            loss += bce(output, Y[batch_start:batch_end][:, None]).mean()

            assert history is not None

            grad_out = output - Y[batch_start:batch_end][..., None]

            prev_grad_c = np.zeros((batch_size, H))

            prev_grad_h = grad_out @ lstm.W_z[:,:-1]

            grad_Wf = np.zeros((B, lstm.W_f.shape[0], lstm.W_f.shape[1]))
            grad_Wo = np.zeros((B, lstm.W_o.shape[0], lstm.W_o.shape[1]))
            grad_Wi = np.zeros((B, lstm.W_i.shape[0], lstm.W_i.shape[1]))
            grad_Wc = np.zeros((B, lstm.W_c.shape[0], lstm.W_c.shape[1]))


            batch_indices = np.arange(B)
            time_indices = S[batch_start:batch_end] - 1
            h_T = history[IH][batch_indices, time_indices]

            grad_Wz = grad_out[:, :, None] * add_bias_col(h_T)[:, None, :]
            
            debug = np.random.rand(1) > 0.95


            mask = np.arange(sequence_length)[None, :] < S[batch_start:batch_end, None]

            # print('\n mask shape', mask.shape)

            for t in reversed(range(sequence_length)):
                valid = mask[:, t][:, None]

                # print(valid.shape)

                prev_c = np.zeros((B, H)) if t == 0 else history[IC,:,t-1]
                prev_h = np.zeros((B, H)) if t == 0 else history[IH,:,t-1]

                grad_h = prev_grad_h

                grad_c = prev_grad_c + grad_h * history[IO,:,t] * (1 - np.tanh(history[IC,:,t]) ** 2)
                grad_f = grad_c * prev_c
                grad_o = grad_h * np.tanh(history[IC][:,t])
                grad_i = grad_c * history[ID,:,t]
                grad_candidate = grad_c * history[II,:,t]

                grad_a_f = grad_f * history[IF,:,t] * (1 - history[IF,:,t])
                grad_a_i = grad_i * history[II,:,t] * (1 - history[II,:,t])
                grad_a_o = grad_o * history[IO,:,t] * (1 - history[IO,:,t])
                grad_a_c = grad_candidate * (1 - history[ID,:,t] ** 2)

                u = (grad_a_f @ lstm.W_f + 
                     grad_a_i @ lstm.W_i + 
                     grad_a_o @ lstm.W_o + 
                     grad_a_c @ lstm.W_c)

                prev_grad_c = grad_c * history[IF,:,t]
                prev_grad_h = u[:,E:-1]

                # ignore all gradients that are not within sequence length
                grad_a_f *= valid 
                grad_a_i *= valid 
                grad_a_o *= valid 
                grad_a_c *= valid

                x_t = add_bias_col(np.concatenate((X[batch_start:batch_end, t], prev_h), axis=1))

                grad_Wf += grad_a_f[:, :, None] * x_t[:, None, :]
                grad_Wi += grad_a_i[:, :, None] * x_t[:, None, :]
                grad_Wo += grad_a_o[:, :, None] * x_t[:, None, :]
                grad_Wc += grad_a_c[:, :, None] * x_t[:, None, :]

                # ignore all gradients that are not within sequence length
                prev_grad_c *= valid
                prev_grad_h *= valid

                if debug_gradients and debug:
                    grad_norm_t = np.mean(np.abs(grad_h))
                    print(f"t={t}, mean |grad_h|={grad_norm_t:.2e}")

            if debug_gradients and debug:
                print()
                print(
                    'f (', abs(grad_Wf).max(), ',', abs(grad_Wf).min(), ',', abs(grad_Wf).mean(), ')\n', 
                    'i (', abs(grad_Wi).max(), ',', abs(grad_Wi).min(), ',', abs(grad_Wi).mean(), ')\n', 
                    'o (', abs(grad_Wo).max(), ',', abs(grad_Wo).min(), ',', abs(grad_Wo).mean(), ')\n', 
                    'c (', abs(grad_Wf).max(), ',', abs(grad_Wf).min(), ',', abs(grad_Wf).mean(), ')\n', 
                    'z (', abs(grad_Wf).max(), ',', abs(grad_Wf).min(), ',', abs(grad_Wf).mean(), ')\n')
                
            clip_value = 5.0 

            optimizer.update(lstm.W_f, np.clip(grad_Wf.mean(axis=0), -clip_value, clip_value), 'Wf')
            optimizer.update(lstm.W_i, np.clip(grad_Wi.mean(axis=0), -clip_value, clip_value), 'Wi')
            optimizer.update(lstm.W_o, np.clip(grad_Wo.mean(axis=0), -clip_value, clip_value), 'Wo')
            optimizer.update(lstm.W_c, np.clip(grad_Wc.mean(axis=0), -clip_value, clip_value), 'Wc')
            optimizer.update(lstm.W_z, np.clip(grad_Wz.mean(axis=0), -clip_value, clip_value), 'Wz')

        print(f'\nloss:', loss / num_batches)

        if epoch_callback is not None:
            epoch_callback(lstm, epoch)

    return lstm