import torch
import math


class YaRNRoPE:
    def __init__(
        self,
        dim,
        base=10000,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        max_position_embeddings=32768,
        device=None,
    ):
        self.dim = dim
        self.base = base
        self.rope_factor = rope_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.max_pos = max_position_embeddings
        self.device = device

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )

        yarn_scale = self._build_yarn_scale(inv_freq.shape[0])
        inv_freq = inv_freq * yarn_scale

        self.inv_freq = inv_freq.to(device)

    def _build_yarn_scale(self, half_dim):
        idx = torch.arange(half_dim).float()

        low = self.beta_slow * half_dim
        high = self.beta_fast * half_dim

        scale = torch.ones_like(idx)

        scale[idx > high] = self.rope_factor

        mask = (idx >= low) & (idx <= high)
        scale[mask] = 1.0 + (self.rope_factor - 1.0) * (
            (idx[mask] - low) / (high - low)
        )

        return scale

    def _get_angles(self, seq_len, start_pos):
        positions = torch.arange(
            start_pos,
            start_pos + seq_len,
            device=self.inv_freq.device,
        ).float()

        angles = torch.einsum("i,j->ij", positions, self.inv_freq)
        return angles

    def apply_rotary(self, q, k, start_pos=0):
        B, T, H, D = q.shape
        half = D // 2

        angles = self._get_angles(T, start_pos)
        sin = angles.sin()[None, :, None, :]
        cos = angles.cos()[None, :, None, :]

        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]

        q_rot = torch.cat(
            [q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1
        )
        k_rot = torch.cat(
            [k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1
        )

        return q_rot, k_rot
