from rnn.rnn import RNN
from rnn.train import train_rnn

import numpy as np


HIDDEN_STATE_SIZE = 10
LR = 1e-1
EPOCHS = 100


def main():
    rnn = RNN(3, 1, HIDDEN_STATE_SIZE)

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

    train_rnn(rnn, sample_x, sample_y, EPOCHS, LR)
    

if __name__ == "__main__":
    main()
