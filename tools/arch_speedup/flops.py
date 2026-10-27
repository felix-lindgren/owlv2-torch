# Analytical MAC counts for the OWLv2 vision tower + detection heads.
def layer_macs(n, d, mlp, kv_n=None, q_frac=1.0):
    kv_n = n if kv_n is None else kv_n
    qkvo = 4 * n * d * d
    attn = 2 * n * kv_n * d
    mlpm = 2 * n * d * mlp
    return dict(qkvo=qkvo, attn=attn, mlp=mlpm)

def tower(n, d, mlp, L):
    tot = {"qkvo": 0, "attn": 0, "mlp": 0}
    for _ in range(L):
        for k, v in layer_macs(n, d, mlp).items():
            tot[k] += v
    return tot

for name, img, p, d, mlp, L in [("base", 960, 16, 768, 3072, 12), ("large", 1008, 14, 1024, 4096, 24)]:
    g = img // p; n = g * g + 1
    t = tower(n, d, mlp, L)
    s = sum(t.values())
    print(f"{name}: grid {g}x{g}, N={n}, tower {2*s/1e12:.3f} TFLOP")
    for k, v in t.items():
        print(f"   {k:5s} {v/1e9:8.1f} GMAC  {100*v/s:5.1f}%")
    P = g * g
    slots = 4 * P
    heads = P * d * 4 * d + slots * (2 * d * d + 4 * d) * 2 + slots * d * (512 if name == "base" else 768)
    heads1 = P * (2 * d * d + 4 * d) * 2 + P * d * (512 if name == "base" else 768)
    print(f"   heads (2x2 slots) {heads/1e9:.1f} GMAC = {100*heads/s:.1f}% of tower; without slots {heads1/1e9:.1f} GMAC = {100*heads1/s:.1f}%")
