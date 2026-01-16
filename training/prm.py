import torch
import torch.nn as nn

class ProcessRewardModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)

    def forward(self, hidden_states):
        return self.scorer(hidden_states).squeeze(-1)
