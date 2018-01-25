
# This code is a re-produce of min-char-rnn.py (https://gist.github.com/karpathy/d4dee566867f8291f086)
# Most parts of code are the same, with cleaner format and code style. (Python3)
# For learning purposes only.

# Import libraries
import numpy as np

# Data I/O
data = open('text.txt', 'r').read()
chars = list(set(data))
data_size = len(data)
vocab_size = len(chars)
print('data has {} characters, {} unique.'.format(data_size, vocab_size))
char_to_ix = { ch: i for i, ch in enumerate(chars) }
ix_to_char = { i: ch for i, ch in enumerate(chars) }

# Hyper-parameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# Model parameters
W_xh = np.random.randn(hidden_size, vocab_size) * 0.01 # input to hidden
W_hh = np.random.randn(hidden_size, hidden_size) * 0.01 # hidden to hidden
W_hy = np.random.randn(vocab_size, hidden_size) * 0.01 # hidden to output
b_h = np.zeros((hidden_size, 1)) # hidden bias
b_y = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
    
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    
    # forward pass
    for t in np.arange(len(inputs)):
        # encode input in 1-of-k representation
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        # hidden state
        hs[t] = np.tanh(W_xh@xs[t] + W_hh@hs[t-1] + b_h)
        # output (unnormalised potentials)
        ys[t] = W_hy@hs[t] + b_y
        # normalised probablities
        ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))
        # softmax (cross-entropy loss)
        loss += -np.log(ps[t][targets[t]][0])

    # backward pass
    # in this part, I named variables explicitly in the format of `dxdy` to indicate the Chain Rule
    dLdW_xh, dLdW_hh, dLdW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
    dLdb_h, dLdb_y = np.zeros_like(b_h), np.zeros_like(b_y)
    dLdh_next = np.zeros_like(hs[0])
    for t in np.flip(np.arange(len(inputs)),0):
        # dL/dy
        dLdy = np.copy(ps[t])
        dLdy[targets[t]] -= 1
        # dL/dW_hy
        dLdW_hy += (dLdy@hs[t].T) # (v x 1) x (h x 1).T = (v x h)
        # dL/db_y
        dLdb_y += dLdy
        # dL/dh
        dLdh = W_hy.T@dLdy # (v x h).T x (v x 1) = (h x 1)
        # backprop through tanh
        dLdh_raw = (1 - hs[t]*hs[t]) * dLdh
        # dL/db_h
        dLdb_h += dLdh_raw
        # dL/dW_hh
        dLdW_hh = dLdh_raw@hs[t-1].T
        # dL/dW_xh
        dLdW_xh = dLdh_raw@xs[t].T
        # hidden gradient flow to the previous step h[t-1]
        dLdh_next = W_hh.T@dLdh_raw
    
    # clip to mitigate gradient vanish/explode
    for dparam in [dLdW_hy, dLdW_hh, dLdW_xh, dLdb_y, dLdb_h]:
        np.clip(dparam, -5, 5, out=dparam)
    
    return loss, dLdW_xh, dLdW_hh, dLdW_hy, dLdb_h, dLdb_y, hs[len(inputs)-1]

def sample(h, seed_index, n):
    x = np.zeros((vocab_size, 1))
    x[seed_index] = 1
    indexes = []
    for t in np.arange(n):
        h = np.tanh(W_hh@h + W_xh@x + b_h)
        y = W_hy@h + b_y
        p = np.exp(y)/np.sum(np.exp(y))
        index = np.random.choice(np.arange(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[index] = 1
        indexes.append(index)
    return indexes

# =============
# = Main body =
# =============
iterations = 100000 # number of iteration to train
pointer = 0 # data pointer
mW_xh, mW_hh, mW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
mb_h, mb_y = np.zeros_like(b_h), np.zeros_like(b_y) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size) * seq_length # loss at iteration 0

for i in range(iterations):
    if pointer+seq_length+1 > len(data) or i == 0:
        hprev = np.zeros((hidden_size, 1)) # reset rnn memory
        pointer = 0
    inputs = [char_to_ix[ch] for ch in data[pointer:pointer+seq_length]]
    targets = [char_to_ix[ch] for ch in data[pointer+1:pointer+seq_length+1]]
    
    if i % 100 == 0:
        sample_indexes = sample(hprev, inputs[0], 100)
        txt = ''.join(ix_to_char[ix] for ix in sample_indexes)
        print('----\n {} \n----'.format(txt))

    loss, dLdW_xh, dLdW_hh, dLdW_hy, dLdb_h, dLdb_y, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    
    if i % 100 == 0:
        print('iter: {}, loss: {}'.format(i, smooth_loss))
        
    for param, dparam, mem in zip([W_xh, W_hh, W_hy, b_h, b_y], 
                                [dLdW_xh, dLdW_hh, dLdW_hy, dLdb_h, dLdb_y], 
                                [mW_xh, mW_hh, mW_hy, mb_h, mb_y]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
    
    pointer += seq_length

