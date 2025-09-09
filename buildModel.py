# pytorch tools for tensors, dataset handling, and neural network layers
import torch
from torch.utils.data import Dataset
import torch.nn as nn

# dataset for sequences, inherits PyTorch Dataset for batching & shuffling
class SequenceDataset(Dataset):
    def __init__(self, sequences, item2idx, max_len=20, pad_token='<PAD>'):
        self.pairs = []
        self.max_len = max_len
        self.item2idx = item2idx
        self.pad_idx = item2idx[pad_token]

        for seq in sequences:
            idx_seq = [item2idx.get(str(item), item2idx['<UNK>']) for item in seq]
            for i in range(1, len(idx_seq)):
                input_seq = idx_seq[:i][-max_len:]
                target = idx_seq[i]
                # Skip pairs where target is PAD
                if target == self.pad_idx:
                    continue
                self.pairs.append((input_seq, target))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, target = self.pairs[idx]
        if len(input_seq) < self.max_len:
            input_seq = [self.pad_idx] * (self.max_len - len(input_seq)) + input_seq
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
