import numpy as np

from helpers import init_random, sigmoid, add_bias_col, xavier_uniform, orthogonal

from os import path


"""
|--------------------
|  CONVENTION
|--------------------
| 
| h -> hidden state
| c -> cell state
| f -> forget gate
| i -> input gate
| o -> output gate
| f -> cadidate state

"""

# indices of activation history
IH = 0
IC = 1
IF = 2
II = 3
IO = 4
ID = 5


class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        Initialize the weight matrices of the LSTM cell.

        Args:
            input_size (int): size of the input vector
            hidden_size (int): size of the hidden state
        """
        self.input_size = input_size

        H = hidden_size
        E = input_size
        concat = E + H + 1

        self.W_f = xavier_uniform((H, concat))
        self.W_i = xavier_uniform((H, concat))
        self.W_o = xavier_uniform((H, concat))
        self.W_c = xavier_uniform((H, concat))

        self.W_f[:, E:E+H] = orthogonal((H, H))
        self.W_i[:, E:E+H] = orthogonal((H, H))
        self.W_o[:, E:E+H] = orthogonal((H, H))
        self.W_c[:, E:E+H] = orthogonal((H, H))

        # bias forget gate to avoid initial forgetting
        self.W_f[:, -1] += 1.5

        self.activations = None

    @property
    def hidden_state_size(self) -> int:
        return self.W_f.shape[0]
    
    def step(self, x: np.ndarray, c: np.ndarray | None = None, h: np.ndarray | None = None) -> list[np.ndarray]:
        """
        Perform a single LSTM step.
        
        Args:
            x (np.ndarray): input vector (1D or 2D for batched)
            c (np.ndarray | None): cell state (1D or 2D for batched)
            h (np.ndarray | None): hidden state (1D or 2D for batched)
        
        Returns:
            list[np.ndarray]: next cell state and hidden state
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        elif x.ndim != 2:
            raise ValueError(f'only 1D or 2D (batched) inputs allowed, received {x.ndim}D')

        if c is None:
            c = np.zeros((x.shape[0], self.W_c.shape[0]))
        if h is None:
            h = np.zeros((x.shape[0], self.W_c.shape[0]))

        assert h is not None
        assert c is not None

        u = add_bias_col(np.concatenate((x, h), axis=1))
        
        f = sigmoid(u @ self.W_f.T)
        i = sigmoid(u @ self.W_i.T)
        o = sigmoid(u @ self.W_o.T)

        d = np.tanh(u @ self.W_c.T)

        c_next = f * c + i * d
        h_next = o * np.tanh(c_next)

        self.activations = {
            'f': f,
            'i': i,
            'o': o,
            'd': d
        }

        return [c_next, h_next]
    
    def save(self, location: str) -> None:
        """
        Save the LSTM weights to the specified location.

        Args:
            location (str): directory to save the weights
        """
        np.save(path.join(location, 'Wi'), self.W_i)
        np.save(path.join(location, 'Wo'), self.W_o)
        np.save(path.join(location, 'Wf'), self.W_f)
        np.save(path.join(location, 'Wc'), self.W_c)


class LSTMClassifier(LSTMCell):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, scale_out: bool = True) -> None:
        """
        Initialize the LSTM classifier.

        Args:
            input_size (int): size of the input vector
            hidden_size (int): size of the hidden state
            output_size (int): size of the output vector
            scale_out (bool): whether to scale the output weights
        """
        super().__init__(input_size, hidden_size)

        self.W_z = np.random.randn(output_size, hidden_size + 1)
        if scale_out:
            self.W_z *= 0.5

    @property
    def output_size(self) -> int:
        return self.W_z.shape[0]


    def step(self, x: np.ndarray, c: np.ndarray | None = None, h: np.ndarray | None = None) -> list[np.ndarray]:
        """
        Perform a single LSTM step and compute the output.

        Args:
            x (np.ndarray): input vector (1D or 2D for batched)
            c (np.ndarray | None): cell state (1D or 2D for bat
            h (np.ndarray | None): hidden state (1D or 2D for batched)
        
        Returns:
            list[np.ndarray]: output, next cell state and hidden state
        """
        c_next, h_next = super().step(x, c, h)
        
        o = add_bias_col(h_next) @ self.W_z.T

        return [o, c_next, h_next]
    
    def logit(self, x: np.ndarray, s: np.ndarray, store_history: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Compute the logit output of the LSTM classifier for a sequence of inputs.

        Args:
            x (np.ndarray): input sequence (2D or 3D for batched)
            s (np.ndarray): sequence lengths for each batch
        Returns:
            np.ndarray: logit output
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        batch_size = len(x)
        sequence_length = int(s.max())

        c = None
        h = None
        o = None

        history = None
        if store_history:
            history = np.zeros((6, batch_size, sequence_length, self.hidden_state_size))

        outputs = np.zeros((batch_size, sequence_length, self.output_size))

        for t in range(sequence_length):
            o, c, h = self.step(x[:,t], c, h)

            outputs[:,t] = o
            if store_history:
                assert history is not None
                assert self.activations is not None

                history[IC,:,t] = c
                history[IH,:,t] = h
                history[IO,:,t] = self.activations['o']
                history[IF,:,t] = self.activations['f']
                history[II,:,t] = self.activations['i']
                history[ID,:,t] = self.activations['d']

        assert o is not None

        # masking to take output at last timestep
        outputs = np.array([outputs[i, s[i] - 1, :] for i in range(batch_size)])

        return outputs, history
    
    
    
    def probability(self, x: np.ndarray, s: np.ndarray, store_history: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Compute the output of the LSTM classifier for a sequence of inputs.

        Args:
            x (np.ndarray): input sequence (2D or 3D for batched)
            s (np.ndarray): sequence lengths for each batch
            store_history (bool): whether to store the activation history
        Returns:
            tuple[np.ndarray, np.ndarray | None]: output and activation history (if stored)
        """
        logits, history = self.logit(x, s, store_history)
        probs = sigmoid(logits)

        return probs, history

    
    def save(self, location: str) -> None:
        """
        Save the LSTM weights to the specified location.

        Args:
            location (str): directory to save the weights
        """
        np.save(path.join(location, 'Wz'), self.W_z)
        super().save(location)

    
    @staticmethod
    def load(location: str) -> 'LSTMClassifier':
        """
        Load the LSTM weights from the specified location.
        
        Args:
            location (str): directory to load the weights from
        
        Returns:
            LSTMClassifier: the loaded LSTM classifier
        """
        obj = LSTMClassifier(0, 0, 0)

        obj.W_i = np.load(path.join(location, 'Wi.npy'))
        obj.W_o = np.load(path.join(location, 'Wo.npy'))
        obj.W_f = np.load(path.join(location, 'Wf.npy'))
        obj.W_c = np.load(path.join(location, 'Wc.npy'))
        obj.W_z = np.load(path.join(location, 'Wz.npy'))

        return obj