import sys
sys.path.insert(0, '..')

from os import path

import numpy as np
import matplotlib.pyplot as plt

from lstm.lstm import LSTMClassifier
from lstm.train import train_lstm_classifier_batched
from rnn.rnn import RNN
from rnn.train import train_rnn_classifier

plt.rcParams.update({'font.size': 14})

def plot_gradient_decay(
    grads: list[tuple[str, str, np.ndarray]],
    S: np.ndarray,
    threshold: tuple[float, float] = (1e-10, 1e6),
    plot_collapsed: bool = False,
) -> None:
    """
    Plot gradient magnitude decay vs number of backpropagated steps.

    For each sample, we reindex timesteps so that:
        depth = number of steps backpropagated from the end (1 = last step).

    Args:
        grads:      list of (label, color, gradients) tuples
                    gradients: (E, N, T, H) array of gradients
        S:          (N,) true sequence lengths
        threshold:  gradient norm below which it's considered collapsed
        max_depth:  maximum number of backprop steps to display
        bin_size:   bin width for smoother collapsed-ratio plot
    """

    max_depth = S.max() - 1


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
                    if val > threshold[0] and val < threshold[1]:
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

    depths = np.arange(1, max_depth + 1)

    # plot gradient magnitudes

    plt.figure(figsize=(12, 6))
    for label, color, grad in grads:
        med, q25, q75 = compute_stats(grad, S)

        plt.plot(depths, med, label=label, color=color)
        plt.fill_between(depths, q25, q75, color=color, alpha=0.3)

    plt.yscale("log")
    plt.xlabel("Number of backpropagated steps (depth)")
    plt.ylabel("L2 Gradient Norm (log scale)")
    plt.title(f"Gradient decay vs. backpropagation depth")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if plot_collapsed:
        # plot collapsed ratios

        plt.figure(figsize=(12, 6))
        for label, color, grad in grads:
            E, N, T, H = grad.shape

            collapsed = np.zeros(max_depth, dtype=int)
            total = np.zeros(max_depth, dtype=int)

            grad_l2 = np.linalg.norm(grad, axis=3)  # (E, N, T)

            for e in range(E):
                for n in range(N):
                    seq_len = min(S[n], T)
                    for t in range(seq_len):
                        d = seq_len - t  # number of backprop steps (1 = last)
                        if d > max_depth:
                            continue
                        val = grad_l2[e, n, t]
                        total[d-1] += 1
                        if val <= threshold[0] or val >= threshold[1]:
                            collapsed[d-1] += 1

            collapsed_ratio = collapsed / total
            collapsed_ratio[0:2] = 0.0  # ignore first two depths for better visualization

            plt.plot(depths, collapsed_ratio, label=label, color=color)

        plt.xlabel("Number of backpropagated steps (depth)")
        # write label, format number in scientific notation
        label = f'[{threshold[0]:.1e}, {threshold[1]:.1e}]'
        plt.ylabel(f"Ratio of gradients outside {label}")
        plt.title(f"Collapsed gradient ratio vs. backpropagation depth")
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_gradients_rnns() -> None:
    epochs = 10
    lr = 0.0002

    N = 2000
    H = [16, 32, 64]

    grads = []

    x = np.load(path.join('data', 't-10000-s-123-x.npy'))[3000:3000 + N]
    y = np.load(path.join('data', 't-10000-s-123-y.npy'))[3000:3000 + N]
    s = np.load(path.join('data', 't-10000-s-123-s.npy'))[3000:3000 + N]

    for i, h in enumerate(H):
        gradients_rnn = np.zeros((epochs, N, s.max(), h))

        rnn = RNN(x.shape[2], 1, h)

        def accumulate_rnn(grad_h: np.ndarray, e: int, s: int, t: int) -> None:
            gradients_rnn[e, s, t] = grad_h
        
        train_rnn_classifier(rnn, x, y, s, epochs, lr, accumulate_rnn)

        grads.append((f"RNN H={h}", f'C{i}', gradients_rnn))

    plot_gradient_decay(grads, s)


def plot_gradients_rnn_vs_lstm() -> None:
    epochs = 10
    lr = 0.00005

    N = 5000
    H = 64

    x = np.load(path.join('data', 'train-x.npy'))[3000:3000 + N]
    y = np.load(path.join('data', 'train-y.npy'))[3000:3000 + N]
    s = np.load(path.join('data', 'train-s.npy'))[3000:3000 + N]

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
    
    train_rnn_classifier(rnn, x, y, s, epochs, lr, accumulate_rnn)

    plot_gradient_decay([
        ("RNN ∇h", 'C0', gradients_rnn),
        ("LSTM ∇h", 'C1', gradients_lstm_h),
        ("LSTM ∇c", 'C2', gradients_lstm_c),
    ], s, plot_collapsed=True)


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


def plot_sentence_lengths() -> None:
    s = np.load(path.join('data', 'train-s.npy'))

    print(f'Sentence lengths: min={s.min()}, max={s.max()}, mean={s.mean():.2f}, median={np.median(s)}')

    plt.figure(figsize=(10, 5))
    plt.hist(s, bins=range(1, s.max() + 2), align='left', edgecolor='black')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentence Lengths')
    plt.grid(axis='y')
    plt.show()


if __name__ == "__main__":
    # plot_gradients_rnn_vs_lstm()
    # plot_run('lstm-run-3')
    plot_sentence_lengths()
    #plot_gradients_rnns()