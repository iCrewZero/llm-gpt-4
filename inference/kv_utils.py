def truncate_kv(kv_cache, new_len):
    out = []
    for layer in kv_cache:
        k_i8, k_s, v = layer
        out.append((
            k_i8[:, :, :new_len, :],
            k_s,
            v[:, :, :new_len, :]
        ))
    return out
