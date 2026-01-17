def balance_penalty(router_probs):
    usage = router_probs.mean(dim=(0,1))
    return usage.var()
