from os import path

import numpy as np
import matplotlib.pyplot as plt

from lstm.lstm import LSTMClassifier
from lstm.train import train_lstm_classifier_batched
from rnn.rnn import RNN
from rnn.train import train_rnn


def plot_gradient_decay(
    grad_rnn: np.ndarray,
    grad_lstm_h: np.ndarray,
    grad_lstm_c: np.ndarray,
    S: np.ndarray,
    threshold: float = 1e-12,
    max_depth: int = 200,
) -> None:
    """
    Plot gradient magnitude decay vs number of backpropagated steps.

    For each sample, we reindex timesteps so that:
        depth = number of steps backpropagated from the end (1 = last step).

    Args:
        grad_rnn:   (E, N, T, H) gradients of RNN hidden state
        grad_lstm_h:(E, N, T, H) gradients of LSTM hidden state
        grad_lstm_c:(E, N, T, H) gradients of LSTM cell state
        S:          (N,) true sequence lengths
        threshold:  gradient norm below which it's considered collapsed
        max_depth:  maximum number of backprop steps to display
        bin_size:   bin width for smoother collapsed-ratio plot
    """

    def compute_stats(grad: np.ndarray, S: np.ndarray):
        E, N, T, H = grad.shape
        grad_l2 = np.linalg.norm(grad, axis=3)  # (E, N, T)

        values_per_d = [[] for _ in range(max_depth)]
        collapsed = np.zeros(max_depth, dtype=int)
        total = np.zeros(max_depth, dtype=int)

        for e in range(E):
            for n in range(N):
                seq_len = min(S[n], T)
                for t in range(seq_len):
                    d = seq_len - t  # number of backprop steps (1 = last)
                    if d > max_depth:
                        continue
                    val = grad_l2[e, n, t]
                    total[d-1] += 1
                    if val > threshold:
                        values_per_d[d-1].append(val)
                    else:
                        collapsed[d-1] += 1

        median = np.zeros(max_depth)
        q25 = np.zeros(max_depth)
        q75 = np.zeros(max_depth)
        for d in range(max_depth):
            vals = np.array(values_per_d[d])
            if vals.size:
                median[d] = np.median(vals)
                q25[d] = np.percentile(vals, 25)
                q75[d] = np.percentile(vals, 75)
        return median, q25, q75

    med_rnn, q25_rnn, q75_rnn = compute_stats(grad_rnn, S)
    med_h, q25_h, q75_h = compute_stats(grad_lstm_h, S)
    med_c, q25_c, q75_c = compute_stats(grad_lstm_c, S)

    depths = np.arange(1, max_depth + 1)

    # plot gradient magnitudes
    plt.figure(figsize=(12, 6))
    plt.plot(depths, med_rnn, label="RNN ∇h", color="tab:blue")
    plt.fill_between(depths, q25_rnn, q75_rnn, color="tab:blue", alpha=0.3)

    plt.plot(depths, med_h, label="LSTM ∇h", color="tab:orange")
    plt.fill_between(depths, q25_h, q75_h, color="tab:orange", alpha=0.3)

    plt.plot(depths, med_c, label="LSTM ∇c", color="tab:green")
    plt.fill_between(depths, q25_c, q75_c, color="tab:green", alpha=0.3)

    plt.yscale("log")
    plt.xlabel("Number of backpropagated steps (depth)")
    plt.ylabel("L2 Gradient Norm (log scale)")
    plt.title("Gradient decay vs. backpropagation depth")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_gradients() -> None:
    epochs = 10
    lr = 0.0005

    N = 5000
    H = 16

    x = np.load(path.join('data', 't-10000-s-123-x.npy'))[3000:3000 + N]
    y = np.load(path.join('data', 't-10000-s-123-y.npy'))[3000:3000 + N]
    s = np.load(path.join('data', 't-10000-s-123-s.npy'))[3000:3000 + N]

    gradients_lstm_h = np.zeros((epochs, N, s.max(), H))
    gradients_lstm_c = np.zeros((epochs, N, s.max(), H))
    gradients_rnn = np.zeros((epochs, N, s.max(), H))

    lstm = LSTMClassifier(input_size=x.shape[2], hidden_size=H, output_size=1)

    def accumulate_lstm(grad_h: np.ndarray, grad_c: np.ndarray, e: int, s: int, t: int) -> None:
        gradients_lstm_h[e, s, t] = grad_h
        gradients_lstm_c[e, s, t] = grad_c

    train_lstm_classifier_batched(
        lstm, x, y, s,
        batch_size=1,
        epochs=epochs,
        lr=lr,
        permute=False,
        gradient_callback=accumulate_lstm,
    )

    rnn = RNN(x.shape[2], 1, H)

    def accumulate_rnn(grad_h: np.ndarray, e: int, s: int, t: int) -> None:
        gradients_rnn[e, s, t] = grad_h
    
    train_rnn(rnn, x, y, s, epochs, lr, accumulate_rnn)

    plot_gradient_decay(gradients_rnn, gradients_lstm_h, gradients_lstm_c, s)


def plot_run(run_id: str) -> None:
    epochs = []
    train_losses = []
    val_losses = []

    with open(path.join('assets', 'runs', f'{run_id}'), 'r') as f:
        for line in f:
            epoch, train_loss, val_loss = line.strip().split()
            epochs.append(int(epoch))
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # plot_gradients()
    plot_run('lstm-run-3')