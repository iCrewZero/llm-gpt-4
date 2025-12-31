import torch
import torch.nn.functional as F

@torch.no_grad()
def stream_generate(model, idx, max_new):
    for _ in range(max_new):
        logits, _ = model(idx[:, -1:])
        probs = F.softmax(logits[:, -1], -1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], 1)
        yield nxt.item()
