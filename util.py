# Importing libraries
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pathlib


def on_gpu():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_text(filenames):
    text = ''
    for filename in filenames:
        with open(f'data/{filename}', 'r') as f:
            text += f.read()
    return text

def write_file(dirs, filename, text):
    pathlib.Path(dirs).mkdir(parents=True, exist_ok=True)
    with open(dirs + '/' + filename, 'w') as f:
        f.write(text)

def create_dicts(text):
    # We create two dictionaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to integers
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    return chars, int2char, char2int

# Defining method to encode one hot labels
def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

# Defining method to make mini-batches for training
def get_batches(arr, batch_size, seq_length):    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y