def ema(prev, x, alpha):
    return alpha * x + (1.0 - alpha) * prev

def majority(items):
    if not items:
        return None
    counts = {}
    for v in items:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]
