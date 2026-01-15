import torch
from collections import defaultdict


DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
}


class MemoryProfiler:
    def __init__(self):
        self.records = defaultdict(int)
        self.snapshots = {}

    def tensor_bytes(self, tensor):
        if tensor is None:
            return 0
        return tensor.numel() * DTYPE_BYTES[tensor.dtype]

    def record(self, name, tensor):
        self.records[name] += self.tensor_bytes(tensor)

    def snapshot(self, tag):
        torch.cuda.synchronize()
        self.snapshots[tag] = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
        }

    def report(self):
        print("\n==== Tensor Memory Breakdown (MB) ====")
        for k, v in sorted(self.records.items()):
            print(f"{k:30s}: {v / (1024**2):.2f} MB")

        print("\n==== CUDA Memory Snapshots (MB) ====")
        for k, v in self.snapshots.items():
            print(f"\n[{k}]")
            for kk, vv in v.items():
                print(f"  {kk:15s}: {vv / (1024**2):.2f} MB")
