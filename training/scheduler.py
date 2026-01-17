import math

def cosine_schedule(step, total):
    return 0.5 * (1 + math.cos(math.pi * step / total))
