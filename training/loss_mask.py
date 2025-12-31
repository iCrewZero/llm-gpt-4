IGNORE_TOKENS = {
    "<|system|>",
    "<|user|>",
    "<|think|>",
    "<|reflect|>"
}

def build_loss_mask(token_ids, tokenizer):
    mask = []
    ignore_ids = {tokenizer.encode(t)[0] for t in IGNORE_TOKENS}

    for t in token_ids:
        if t in ignore_ids:
            mask.append(-1)
        else:
            mask.append(t)
    return mask
