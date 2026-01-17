import torch

class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, k):
        self.draft = draft_model
        self.target = target_model
        self.k = k

    def step(self, tokens):
        draft_out = self.draft(tokens)
        draft_next = draft_out.argmax(-1)[:,-self.k:]

        verify_logits = self.target(
            torch.cat([tokens, draft_next], dim=1)
        )

        accepted = []
        for i in range(self.k):
            if verify_logits[:,-self.k+i].argmax(-1).item() == draft_next[:,i].item():
                accepted.append(draft_next[:,i])
            else:
                break

        if not accepted:
            accepted.append(verify_logits[:,-1].argmax(-1))

        return torch.stack(accepted, dim=1)
