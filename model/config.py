from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50280
    dim: int = 2048
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 8
    block_size: int = 32768

    rope_base: int = 10000
    rope_factor: float = 8.0

    moe_experts: int = 8
    moe_topk: int = 2

    dropout: float = 0.0
