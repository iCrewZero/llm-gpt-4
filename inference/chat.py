from inference.generate import generate

def chat(model, tokenizer, messages, max_new=256):
    ids = []
    for m in messages:
        ids += tokenizer.encode(f"<|{m['role']}|>")
        ids += tokenizer.encode(m["content"])
        ids.append(tokenizer.sp.eos_id())

    import torch
    idx = torch.tensor([ids], device="cuda")
    out = generate(model, idx, max_new)[0].tolist()
    return tokenizer.decode(out)
