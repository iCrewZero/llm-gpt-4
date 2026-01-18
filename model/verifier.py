import torch.nn as nn

class VerifierHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)
