import torch
import torch.nn.functional as F

def mtp_loss(logits, targets, k=3):
    loss = 0
    for i in range(1, k + 1):
        loss += F.cross_entropy(
            logits[:, :-i].reshape(-1, logits.size(-1)),
            targets[:, i:].reshape(-1)
        )
    return loss / k
