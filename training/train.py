import torch
import torch.nn.functional as F
from model.gpt import GPT

model = GPT(32000, 512, 6, 8).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(100000):
    x = torch.randint(0, 32000, (8, 128)).cuda()
    logits = model(x)

    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 32000),
        x[:, 1:].reshape(-1)
    )

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        print(step, loss.item())
