import torch
import torch.nn.functional as F
from model.gpt import GPT
from training.memory_profiler import MemoryProfiler

VOCAB = 32000
BATCH = 2
SEQ   = 2048
DEVICE = "cuda"
model = GPT(
    vocab_size=VOCAB,
    dim=2048,
    layers=24,
    heads=16,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
profiler = MemoryProfiler()
x = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
with torch.no_grad():
    _ = model(x)

torch.cuda.reset_peak_memory_stats()
profiler.snapshot("start")
model.train()

x = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)

logits = model(x)
profiler.snapshot("after_forward")

loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, VOCAB),
    x[:, 1:].reshape(-1),
)

optimizer.zero_grad()
loss.backward()
profiler.snapshot("after_backward")

optimizer.step()
profiler.snapshot("after_step")
print("loss:", loss.item())
profiler.report()
