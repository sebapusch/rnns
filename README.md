
RNN vs LSTM Gradient Dynamics (SST-2)
====================================

This repository explores the training behavior and gradient flow in vanilla RNNs and LSTMs on a sequence classification task (Stanford SST-2). All models are implemented from scratch in NumPy.

Project Structure
-----------------

```
.
├── main.py                # Entry point for training/testing  
├── scripts/  
│   ├── preprocessing.py   # Prepares and saves preprocessed dataset  
│   ├── plots.py           # Gradient and loss visualizations  
│  
├── lstm/                  # LSTM model and training utilities  
├── rnn/                   # Vanilla RNN model and training utilities  
├── data/                  # Auto-generated: preprocessed .npy files  
├── assets/  
│   ├── runs/              # Saved training logs (one file per run)  
│   ├── weights/           # Saved model weights  
│   └── embeddings/        # Custom or GloVe embeddings  
```

Setup
-----

You’ll need:

- Python 3.10+
- Required packages (install via pip):

pip install numpy datasets tqdm scikit-learn matplotlib

Make sure [GloVe-like embeddings](https://nlp.stanford.edu/projects/glove/) (100D) are available in:
`assets/embeddings/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt`

Data Preprocessing
------------------

Run the following script to generate train/validation/test splits in `.npy` format using SST-2 with GloVe embeddings:

python scripts/preprocessing.py

This will save the data to the `data/` folder.

Training Models
---------------

Edit `main.py` and uncomment the appropriate block depending on the model you want to train.

To train an LSTM:

`train_lstm(run_id="lstm-run-3", epochs=20, lr=0.002, hidden_size=64)`

To train a vanilla RNN:

`train_rnn(run_id="rnn-run-1", epochs=20, lr=0.002, hidden_size=64)`

Then run:

`python main.py`

Model weights and training logs will be saved under `assets/`.

Evaluation and Plotting
------------------------

To test a saved LSTM model on the test set, update `main.py` and run:

`test()`

To visualize training loss:

`python scripts/plots.py`

This will show training vs validation loss curves for a given run. You can also call:

`plot_gradients_rnn_vs_lstm()`

to generate gradient decay plots as seen in the report.

Overfitting Check
-----------------

You can test whether the lstm model can overfit a small subset by uncommenting `test_overfit()` in `main.py`.

Notes
-----

- Embeddings are static (non-trainable) and based on GloVe.
- Only the final output of the RNN/LSTM is used for loss computation (sequence classification).
- Manual weight initialization and batching were used to improve trainability.


