from os import path
import os
from typing import Callable

import numpy as np


from lstm.lstm import LSTMClassifier
from lstm.train import compute_lstm_loss, train_lstm_classifier_batched


HIDDEN_STATE_SIZE = 128
LR = 0.00005
EPOCHS = 100
BATCH_SIZE = 25

EMBEDDING_SIZE = 100

# (not using: too expensive)
def k_fold(k: int, x: np.ndarray, y: np.ndarray, train: Callable, validate: Callable) -> None:
    size = len(y)
    
    if size % k: raise ValueError('too lazy to handle this')

    fold_size = size // k

    if k == 1:
        train(x, y)
        return
    
    for i in range(k):
        rng = (i * fold_size, (i + 1) * fold_size)

        print(f'fold {i + 1}/{k} training')
        train(x[rng[0]:rng[1]], y[rng[0]:rng[1]])
        print(f'fold {i + 1}/{k} validation')
        validate(
            np.concatenate((x[:rng[0]], x[rng[1]:])),
            np.concatenate((y[:rng[0]], y[rng[1]:])),
        )

def train(
        run_id: str, 
        epochs: int, 
        lr: float, 
        hidden_size: int,
    ) -> None:
    """
    Train LSTM classifier on dataset.
    
    Args:
        run_id (str): unique identifier for the training run
        epochs (int): number of training epochs
        lr (float): learning rate
        hidden_size (int): size of the LSTM hidden state
    """
    train_x = np.load(path.join('data', 't-50k-x.npy'))
    train_y = np.load(path.join('data', 't-50k-y.npy'))
    train_s = np.load(path.join('data', 't-50k-s.npy'))

    valid_x = np.load(path.join('data', 'v-100-x.npy'))
    valid_y = np.load(path.join('data', 'v-100-y.npy'))    
    valid_s = np.load(path.join('data', 'v-100-s.npy'))

    def save_results(lstm: LSTMClassifier, epoch: int, train_loss: float) -> None:
        val_loss = compute_lstm_loss(lstm, valid_x, valid_y, valid_s)
        print(f'Epoch {epoch}: train loss {train_loss}, val loss {val_loss}')

        with open(path.join('assets', 'runs', f'{run_id}'), 'a') as f:
            f.write(f'{epoch} {train_loss} {val_loss}\n')

        save_path = path.join('assets', 'weights', f'lstm-{run_id}-epoch-{epoch}')
        os.makedirs(save_path, exist_ok=True)

        lstm.save(save_path)

    lstm = LSTMClassifier(EMBEDDING_SIZE, hidden_size, 1)

    train_lstm_classifier_batched(
        lstm, train_x, train_y, train_s,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        lr=lr,
        epoch_callback=save_results
    )

def test_overfit() -> None:
    """
    Test LSTM classifier on small dataset to check it is able to overfit.
    """

    x = np.load(path.join('data', 'v-100-s-123-x.npy'))
    y = np.load(path.join('data', 'v-100-s-123-y.npy'))    
    s = np.load(path.join('data', 'v-100-s-123-s.npy'))

    model = LSTMClassifier(EMBEDDING_SIZE, HIDDEN_STATE_SIZE, 1)
    train_lstm_classifier_batched(model, x, y, s, epochs=EPOCHS, lr=LR, batch_size=1)


def main():
    # test_overfit()
    train(
        run_id='lstm-run-3',
        epochs=60,
        lr=LR,
        hidden_size=HIDDEN_STATE_SIZE,
    )

if __name__ == "__main__":
    main()
