import torch

def save_checkpoint(model, opt, step, path):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step
    }, path)

def load_checkpoint(model, opt, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    return ckpt["step"]
