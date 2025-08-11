# pytorch tools for tensors, dataset handling, and neural network layers
import torch
from torch.utils.data import Dataset
import torch.nn as nn

# dataset for sequences, inherits PyTorch Dataset for batching & shuffling
class SequenceDataset(Dataset):
    def __init__(self, sequences, item2idx, max_len=20):
        # list to store (input sequence, target) pairs for training
        self.pairs = []
        # maximum length of input sequences
        self.max_len = max_len
        # vocabulary mapping
        self.item2idx = item2idx
        for seq in sequences:
            # convert each item into its index in vocab, using <UNK> index if missing
            # convert items to strings
            idx_seq = [item2idx.get(str(item), item2idx['<UNK>']) for item in seq]
            # generate pairs: for each item except the first, predict that item given previous items
            for i in range(1, len(idx_seq)):
                # use sliding window of size max_len for input
                input_seq = idx_seq[:i][-max_len:]
                # target is current item
                target = idx_seq[i]
                # store pair
                self.pairs.append((input_seq, target))

    def __len__(self):
        # number of training pairs
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, target = self.pairs[idx]
        # pad input sequence on left with 0 (index of <PAD>) if shorter than max length
        if len(input_seq) < self.max_len:
            input_seq = [0] * (self.max_len - len(input_seq)) + input_seq
        # return input sequence tensor and target tensor for given index
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# model definition using two-layer LSTM with dropout and final linear projection
class decoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(decoderModel, self).__init__()
        # embedding layer maps indices to dense vectors; padding_idx=0 means that index 0 (PAD) is ignored in embedding updates
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # 2 stacked LSTM layers, output batch_first, with dropout between layers to reduce overfitting
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        # linear layer maps last hidden state to logits over the vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # convert input indices to embeddings
        embeds = self.embedding(x)
        # pass embeddings through LSTM, get hidden states from last layer
        _, (h_n, _) = self.lstm(embeds)
        # use the last hidden state of last LSTM layer as input to fully connected layer
        out = self.fc(h_n[-1])
        # return the output logits for prediction
        return out
