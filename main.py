from os import path
import os
from typing import Callable

import numpy as np

from lstm.lstm import LSTMClassifier
from lstm.train import compute_lstm_loss, train_lstm_classifier_batched, compute_lstm_balanced_accuracy

from rnn.rnn import RNN
from rnn.train import compute_rnn_loss, train_rnn_classifier, compute_rnn_balance_accuracy


HIDDEN_STATE_SIZE = 64
LR = 0.005
EPOCHS = 20
BATCH_SIZE = 80

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

def train_rnn(
          run_id: str, 
        epochs: int, 
        lr: float, 
        hidden_size: int,
    ) -> None:
    """
    Train RNN classifier on dataset.
    Args:
        run_id (str): unique identifier for the training run
        epochs (int): number of training epochs
        lr (float): learning rate
        hidden_size (int): size of the RNN hidden state
    """
    train_x = np.load(path.join('data', 'train-x.npy'))
    train_y = np.load(path.join('data', 'train-y.npy'))
    train_s = np.load(path.join('data', 'train-s.npy')) 

    valid_x = np.load(path.join('data', 'validation-x.npy'))
    valid_y = np.load(path.join('data', 'validation-y.npy'))
    valid_s = np.load(path.join('data', 'validation-s.npy'))

    def save_results(rnn: RNN, epoch: int, train_loss: float) -> None:
        val_loss = compute_rnn_loss(rnn, valid_x, valid_y, valid_s)
        print(f'Epoch {epoch}: train loss {train_loss}, val loss {val_loss}')

        with open(path.join('assets', 'runs', f'{run_id}'), 'a') as f:
            f.write(f'{epoch} {train_loss} {val_loss}\n')

        save_path = path.join('assets', 'weights', f'rnn-{run_id}-epoch-{epoch}')
        os.makedirs(save_path, exist_ok=True)

        rnn.save(save_path)
    
    rnn = RNN(EMBEDDING_SIZE, 1, hidden_size)
    train_rnn_classifier(
        rnn, train_x, train_y, train_s,
        epochs=epochs,
        lr=lr,
        epoch_callback=save_results
    )
    
    

def train_lstm(
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
    train_x = np.load(path.join('data', 'train-x.npy'))
    train_y = np.load(path.join('data', 'train-y.npy'))
    train_s = np.load(path.join('data', 'train-s.npy'))

    valid_x = np.load(path.join('data', 'validation-x.npy'))
    valid_y = np.load(path.join('data', 'validation-y.npy'))    
    valid_s = np.load(path.join('data', 'validation-s.npy'))

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

    x = np.load(path.join('data', 'validation-x.npy'))[:200]
    y = np.load(path.join('data', 'validation-y.npy'))[:200]    
    s = np.load(path.join('data', 'validation-s.npy'))[:200]

    model = LSTMClassifier(EMBEDDING_SIZE, HIDDEN_STATE_SIZE, 1)
    train_lstm_classifier_batched(model, x, y, s, epochs=EPOCHS, lr=LR, batch_size=1)

def test() -> None:
    """
    Test LSTM classifier on test dataset and print balanced accuracy.
    """

    valid_x = np.load(path.join('data', 'test-x.npy'))
    valid_y = np.load(path.join('data', 'test-y.npy'))    
    valid_s = np.load(path.join('data', 'test-s.npy'))

    lstm = LSTMClassifier(EMBEDDING_SIZE, HIDDEN_STATE_SIZE, 1)
    lstm.load(path.join('assets', 'weights', 'lstm-run-3-epoch-60'))

    rnn = RNN(EMBEDDING_SIZE, 1, HIDDEN_STATE_SIZE)
    rnn.load(path.join('assets', 'weights', 'rnn-run-1-epoch-20'))
    
    accuracy_lstm = compute_lstm_balanced_accuracy(lstm, valid_x, valid_y, valid_s)
    accuracy_rnn = compute_rnn_balance_accuracy(rnn, valid_x, valid_y, valid_s)


    print(f'Balanced accuracy on test set lstm: {accuracy_lstm}')
    print(f'Balanced accuracy on test set rnn: {accuracy_rnn}')


def main():
    # test_overfit()
    train_lstm(
        run_id='lstm-run-3',
        epochs=EPOCHS,
        lr=LR,
        hidden_size=HIDDEN_STATE_SIZE,
    )
    # train_rnn(
    #     run_id='rnn-run-1',
    #     epochs=EPOCHS,
    #     lr=LR,
    #     hidden_size=HIDDEN_STATE_SIZE,
    # )
    test()


if __name__ == "__main__":
    main()
