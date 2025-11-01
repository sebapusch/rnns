import numpy as np

from lstm.lstm import LSTMClassifier, IH, IO, IC, II, IF, ID
from tqdm import tqdm

from helpers import bce, add_bias_col, Adam

def bce_with_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))


def compute_lstm_loss(
        lstm: LSTMClassifier, 
        X: np.ndarray, 
        Y: np.ndarray,
        S: np.ndarray,
) -> float:
    out, _ = lstm.probability(X, S)

    return bce(out, Y[:,None]).mean()


def compute_lstm_accuracy(
        lstm: LSTMClassifier, 
        X: np.ndarray, 
        Y: np.ndarray,
        S: np.ndarray,
        threshold: float = 0.5,
) -> float:
    out, _ = lstm.probability(X, S)

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
        permute: bool = True,
        epoch_callback = None,
        gradient_callback = None,
    ) -> LSTMClassifier:
    
    N = len(Y)                  # sample size
    E = len(X[0][0])            # embedding size
    B = batch_size              # batch size
    H = lstm.hidden_state_size  # hidden state size

    num_batches = N // batch_size

    optimizer = Adam(lr)

    for epoch in range(epochs):
        print(f'starting epoch {epoch + 1}/{epochs}')

        if permute:
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

            x_batch = X[batch_start:batch_end]
            y_batch = Y[batch_start:batch_end]
            s_batch = S[batch_start:batch_end]

            # determine the maximum sequence length in this batch
            sequence_length = int(s_batch.max())

            logits, history = lstm.logit(x_batch, s_batch, True)

            loss += bce_with_logits(logits, y_batch[:, None]).mean()

            assert history is not None

            grad_out = (1 / (1 + np.exp(-logits))) - y_batch[:, None]

            prev_grad_c = np.zeros((batch_size, H))

            prev_grad_h = grad_out @ lstm.W_z[:,:-1]

            grad_Wf = np.zeros((B, lstm.W_f.shape[0], lstm.W_f.shape[1]))
            grad_Wo = np.zeros((B, lstm.W_o.shape[0], lstm.W_o.shape[1]))
            grad_Wi = np.zeros((B, lstm.W_i.shape[0], lstm.W_i.shape[1]))
            grad_Wc = np.zeros((B, lstm.W_c.shape[0], lstm.W_c.shape[1]))


            batch_indices = np.arange(B)
            time_indices = s_batch - 1

            # get h at last timestep for each sequence in batch
            h_T = history[IH][batch_indices, time_indices]

            # dL/dWz = dL/do * h_T
            grad_Wz = grad_out[:, :, None] * add_bias_col(h_T)[:, None, :]

            # create mask for valid timesteps
            mask = np.arange(sequence_length)[None, :] < s_batch[:, None]

            for t in reversed(range(sequence_length)):
                # mask for current timestep
                valid = mask[:, t][:, None]

                prev_c = np.zeros((B, H)) if t == 0 else history[IC,:,t-1]
                prev_h = np.zeros((B, H)) if t == 0 else history[IH,:,t-1]

                grad_h = prev_grad_h
                
                # gradients after activations
                grad_c = prev_grad_c + grad_h * history[IO,:,t] * (1 - np.tanh(history[IC,:,t]) ** 2)
                grad_f = grad_c * prev_c
                grad_o = grad_h * np.tanh(history[IC][:,t])
                grad_i = grad_c * history[ID,:,t]
                grad_candidate = grad_c * history[II,:,t]

                # gradients before activations
                grad_a_f = grad_f * history[IF,:,t] * (1 - history[IF,:,t])
                grad_a_i = grad_i * history[II,:,t] * (1 - history[II,:,t])
                grad_a_o = grad_o * history[IO,:,t] * (1 - history[IO,:,t])
                grad_a_c = grad_candidate * (1 - history[ID,:,t] ** 2)

                # combined gradient for h
                u = (grad_a_f @ lstm.W_f + 
                     grad_a_i @ lstm.W_i + 
                     grad_a_o @ lstm.W_o + 
                     grad_a_c @ lstm.W_c)

                # update prev gradients
                prev_grad_c = grad_c * history[IF,:,t]
                prev_grad_h = u[:,E:-1]

                # ignore all gradients that are not within sequence length
                grad_a_f *= valid 
                grad_a_i *= valid 
                grad_a_o *= valid 
                grad_a_c *= valid

                # prepare input with bias (input + hidden state + bias)
                x_t = add_bias_col(np.concatenate((x_batch[:,t], prev_h), axis=1))

                grad_Wf += grad_a_f[:,:,None] * x_t[:,None,:]
                grad_Wi += grad_a_i[:,:,None] * x_t[:,None,:]
                grad_Wo += grad_a_o[:,:,None] * x_t[:,None,:]
                grad_Wc += grad_a_c[:,:,None] * x_t[:,None,:]

                # ignore all gradients that are not within sequence length
                prev_grad_c *= valid
                prev_grad_h *= valid

                if gradient_callback is not None:
                    if valid[batch, 0]:
                        gradient_callback(grad_h[0], grad_c[0], epoch, batch, t)
                
            clip_value = 5.0 

            optimizer.update(lstm.W_f, np.clip(grad_Wf.mean(axis=0), -clip_value, clip_value), 'Wf')
            optimizer.update(lstm.W_i, np.clip(grad_Wi.mean(axis=0), -clip_value, clip_value), 'Wi')
            optimizer.update(lstm.W_o, np.clip(grad_Wo.mean(axis=0), -clip_value, clip_value), 'Wo')
            optimizer.update(lstm.W_c, np.clip(grad_Wc.mean(axis=0), -clip_value, clip_value), 'Wc')
            optimizer.update(lstm.W_z, np.clip(grad_Wz.mean(axis=0), -clip_value, clip_value), 'Wz')

        loss /= num_batches
        print(f'\nloss:', loss)

        if epoch_callback is not None:
            epoch_callback(lstm, epoch, loss)

    return lstm