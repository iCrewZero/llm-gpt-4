import torch
from model.gpt import GPT
from inference.paged_kv import KVState, KVAllocator

@torch.no_grad()
def speculative_decode(target: GPT, draft: GPT, start_token, allocator: KVAllocator, steps=32, speculate_k=4):
    t_state = KVState([], 0)
    d_state = KVState([], 0)
    token = start_token
    out = [token.item()]

    for _ in range(steps):
        base_len = t_state.seqlen
        draft_tokens = []

        for _ in range(speculate_k):
            logits = draft(token.unsqueeze(0), d_state, allocator)
            token = logits.argmax(-1)
            draft_tokens.append(token)

        accepted = True
        for t in draft_tokens:
            logits = target(t.unsqueeze(0), t_state, allocator)
            pred = logits.argmax(-1)
            if pred.item() != t.item():
                t_state.rollback(base_len, allocator)
                out.append(pred.item())
                token = pred
                accepted = False
                break
            out.append(t.item())

        if accepted:
            token = draft_tokens[-1]

    return out
