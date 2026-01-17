import torch

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx+self.seq_len+1]
        return {
            "input_ids": torch.tensor(chunk[:-1]),
            "labels": torch.tensor(chunk)
        }
