from rnn.rnn import RNN
from rnn.train import train_rnn
from lstm.lstm import LSTMClassifier
from lstm.train import train_lstm_classifier
from hopfield.hopfield import Hopfield
from hopfield.helpers import make_digits, print_digit, corrupt_pattern

import numpy as np


HIDDEN_STATE_SIZE = 10
LR = 5e-1
EPOCHS = 10

def test_hopfield() -> None:
    hn = Hopfield(50 ** 2)

    digits = make_digits(size=50, thickness=4)

    hn.train_implicit(digits)

    print_digit(digits[4])
    hn.recall(corrupt_pattern(digits[5], 0.01), print_digit)



def test_lstm(X: np.ndarray, Y: np.ndarray) -> None:
    lstm = LSTMClassifier(3, HIDDEN_STATE_SIZE, 1)

    train_lstm_classifier(lstm, X, Y, EPOCHS, LR)

    h = None
    o = None
    c = None
    for t in range(3):
        o, c, h = lstm.step(X[0][t], c, h)
    print(o)


def main():
    # rnn = RNN(3, 1, HIDDEN_STATE_SIZE)

    sample_x = np.array([
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3], 
        ],
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]
    ])

    sample_y = np.array([
        1,
        1,
    ])

    test_hopfield()

    # test_lstm(sample_x, sample_y)

    # train_rnn(rnn, sample_x, sample_y, EPOCHS, LR)

    # h = None
    # o = None
    # for t in range(3):
    #     o, h = rnn.step(sample_x[0][t], h)
    # print(o)
    

if __name__ == "__main__":
    main()
