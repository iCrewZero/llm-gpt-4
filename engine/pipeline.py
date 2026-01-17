import torch
from .batching import ContinuousBatcher
from .speculative import SpeculativeDecoder
from .rollout import RolloutEngine

class InferencePipeline:
    def __init__(self, model, draft, prm, cfg):
        self.batch = ContinuousBatcher(cfg.max_batch)
        self.spec = SpeculativeDecoder(draft, model, cfg.speculative_k)
        self.rollout = RolloutEngine(model, prm, None)

    def submit(self, tokens):
        self.batch.add(tokens)

    def step(self):
        batch = self.batch.next_batch()
        if not batch:
            return None

        tokens = torch.cat(batch, dim=0)
        return self.spec.step(tokens)
