class Config:
    vocab_size = 32000
    dim = 768
    n_layers = 12
    n_heads = 12
    n_kv_heads = 6

    block_size = 32768
    page_size = 16
    num_kv_pages = 8192

    use_moe = True
    moe_experts = 8
    moe_top_k = 2

    use_mtp = True
    mtp_tokens = 4

    use_prm = True
    use_mcts = True

    device = "cuda"
