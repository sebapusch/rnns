from rnn import RNN

import numpy as np


def main():
    rnn = RNN(3, 1, 10)

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

    for s in range(2):

        rnn.reset_hidden_state()

        hidden_states = [rnn.hidden_state.copy()];
        z = np.zeros(1)
        for t in range(3):
            z = rnn.step(sample_x[s][t])
            hidden_states.append(rnn.hidden_state.copy())
        loss = -(sample_y[s] * np.log(z) + (1 - sample_y[s]) * np.log(1 - z))

        print(loss)

        grad_out = z - sample_y[s]
        
        grad_Woh = grad_out * hidden_states[-1].T 

        grad_Whh = np.zeros_like(rnn.weights_hh)
        grad_Whx = np.zeros_like(rnn.weights_hx)

        grad_h_T = rnn.weights_oh.T @ grad_out

        delta_t = []
        for t in reversed(range(2)):
            grad_h_t = 
            

if __name__ == "__main__":
    main()
