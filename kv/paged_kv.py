import torch
from collections import deque

class PageAllocator:
    def __init__(self, n):
        self.free = deque(range(n))
        self.used = deque()

    def alloc(self):
        if not self.free:
            raise RuntimeError("KV OOM")
        p = self.free.popleft()
        self.used.append(p)
        return p

    def evict(self):
        p = self.used.popleft()
        self.free.append(p)
        return p

class KVState:
    def __init__(self):
        self.pages = []
        self.seq_len = 0

def append_token(state, alloc, page_size):
    if state.seq_len % page_size == 0:
        state.pages.append(alloc.alloc())
    state.seq_len += 1
