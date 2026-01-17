class ModelConfig:
    vocab_size = 50280
    dim = 2048
    n_layer = 24
    n_head = 16
    n_kv_head = 8
    block_size = 32768

    rope_base = 10000
    rope_factor = 8.0

    use_flash = True
    use_mla = True
    use_moe = True
    use_mtp = True

    moe_experts = 8
    moe_topk = 2

    dropout = 0.0
