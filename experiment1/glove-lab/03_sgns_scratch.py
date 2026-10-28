"""
GloVe essay lab - Step 3: train skip-gram with negative sampling (word2vec)
from scratch on the SAME toy corpus, then compare with GloVe on identical
analogy / nearest-neighbour tasks. Also checks the Levy & Goldberg (2014)
claim that SGNS implicitly factorizes (PMI - log k): we correlate the learned
dot products w_c . w_o with PMI(c, o) over observed pairs.
"""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
np.random.seed(0)

# ---------- data ----------
w2i = json.load(open(os.path.join(HERE, "vocab.json")))
V = len(w2i)
i2w = {i: w for w, i in w2i.items()}
tokens = open(os.path.join(HERE, "corpus.txt")).read().split()
ids = [w2i[t] for t in tokens]

window = 2
pairs = []
for c, i in enumerate(ids):
    for d in range(1, window + 1):
        if c - d >= 0: pairs.append((i, ids[c - d]))
        if c + d < len(ids): pairs.append((i, ids[c + d]))
N = len(pairs)
print("pairs:", N)

# unigram distribution for negatives, p^0.75 (paper: Mikolov et al. 2013)
cnt = Counter(ids)
uni = np.array([cnt[i] for i in range(V)], dtype=np.float64)
uni = uni ** 0.75
uni /= uni.sum()

# ---------- SGNS ----------
d = 50
neg = 5
lr = 0.025
epochs = 15
W = np.random.uniform(-0.5, 0.5, (V, d)) * 0.01
Wo = np.random.uniform(-0.5, 0.5, (V, d)) * 0.01

def sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

rng = np.random.default_rng(0)
for ep in range(epochs):
    order = rng.permutation(N)
    for t in range(N):
        c, o = pairs[order[t]]
        # positive sample
        s = W[c].dot(Wo[o])
        g = sigmoid(s) - 1.0
        W[c] -= lr * g * Wo[o]
        Wo[o] -= lr * g * W[c]
        # negative samples
        negs = rng.choice(V, size=neg, p=uni, replace=False)
        for n in negs:
            sn = W[c].dot(Wo[n])
            gn = sigmoid(sn)
            W[c] -= lr * gn * Wo[n]
            Wo[n] -= lr * gn * W[c]

final = W + Wo
np.savez(os.path.join(HERE, "sgns_vecs.npz"), W=final)
print("SGNS training done")

# ---------- evaluations (identical to GloVe script) ----------
def cosine(a, b):
    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def analogy(a, b, c, exclude=()):
    target = final[w2i[b]] - final[w2i[a]] + final[w2i[c]]
    best, best_w = -2, None
    for wi in range(V):
        if wi in exclude: continue
        sc = cosine(target, final[wi])
        if sc > best: best, best_w = sc, i2w[wi]
    return best_w

analogies = [
    ("france", "paris", "germany", "berlin"),
    ("italy", "rome", "spain", "madrid"),
    ("england", "london", "japan", "tokyo"),
    ("germany", "berlin", "italy", "rome"),
    ("spain", "madrid", "france", "paris"),
    ("japan", "tokyo", "england", "london"),
    ("man", "woman", "king", "queen"),
    ("king", "queen", "man", "woman"),
    ("man", "woman", "boy", "girl"),
    ("king", "queen", "boy", "girl"),
    ("brother", "sister", "uncle", "aunt"),
    ("king", "queen", "he", "she"),
    ("man", "woman", "he", "she"),
    ("run", "running", "walk", "walking"),
    ("walk", "walking", "dance", "dancing"),
    ("dance", "dancing", "swim", "swimming"),
    ("swim", "swimming", "fly", "flying"),
]
acc = 0
for a, b, c, d_exp in analogies:
    pred = analogy(a, b, c, exclude={w2i[a], w2i[b], w2i[c]})
    acc += pred == d_exp
    print(f"  {a}-{b} :: {c} -> {pred}  (expected {d_exp}) {'OK' if pred == d_exp else 'x'}")
print(f"SGNS analogy accuracy: {acc / len(analogies) * 100:.1f}% ({acc}/{len(analogies)})")

print("\n--- SGNS nearest neighbours ---")
for w in ["ice", "steam", "water", "paris", "king", "running", "fashion"]:
    wi = w2i[w]
    sims = sorted(((cosine(final[wi], final[j]), j) for j in range(V) if j != wi),
                  reverse=True)[:5]
    print(f"  {w}: " + ", ".join(f"{i2w[j]}({s:.3f})" for s, j in sims))

# ---------- PMI correlation (Levy & Goldberg 2014: SGNS ~ factorizing PMI - log k) ----------
# rebuild co-occurrence counts (integer) from corpus
Xc = np.zeros((V, V), dtype=np.float64)
for c, i in enumerate(ids):
    for d in range(1, window + 1):
        if c - d >= 0: Xc[i, ids[c - d]] += 1
        if c + d < len(ids): Xc[i, ids[c + d]] += 1
Xi = Xc.sum(1)
tot = Xc.sum()

pmis, dots = [], []
for i, j in zip(*np.nonzero(Xc)):
    pmi = np.log(Xc[i, j] * tot / (Xi[i] * Xi[j])) - np.log(neg)  # PMI - log k
    pmis.append(pmi)
    dots.append(W[i].dot(Wo[j]))
pmis = np.array(pmis); dots = np.array(dots)
r = np.corrcoef(pmis, dots)[0, 1]
print(f"\nSGNS: Pearson correlation between w_c.w_o and PMI(c,o)-log({neg}): {r:.3f}")

plt.figure(figsize=(6.5, 5))
plt.scatter(pmis, dots, s=8, alpha=0.6)
plt.xlabel("PMI(c,o) - log k  (k = #negative samples)")
plt.ylabel("learned w_c . w_o")
plt.title("SGNS learns a (shifted) PMI factorization")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_sgns_pmi.png"), dpi=140); plt.close()
print("figures saved to", FIG)
