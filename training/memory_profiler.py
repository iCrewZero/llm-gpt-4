import torch

class MemoryProfiler:
    def __init__(self):
        self.snapshots = []

    def snapshot(self, tag):
        self.snapshots.append((
            tag,
            torch.cuda.memory_allocated(),
            torch.cuda.max_memory_allocated()
        ))

    def report(self):
        for t, cur, peak in self.snapshots:
            print(f"{t}: {cur/1e6:.1f}MB | peak {peak/1e6:.1f}MB")
