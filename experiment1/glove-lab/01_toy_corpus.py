"""
GloVe essay lab - Step 1: build a small synthetic corpus + word-word co-occurrence matrix X.

Reproduces the setup of Pennington et al. (2014) Section 3 on a toy scale:
  - ice/steam probes (solid, gas, water, fashion) to replicate Table 1
  - gender / country-capital / verb-tense analogies

Design principles:
  1. Analogy pairs (king/queen, paris/france, run/running) do NOT co-occur
     directly; each pair shares parallel "role" contexts, so difference
     vectors come from distinct shared contexts, not from mutual co-occurrence.
  2. Function words (the/is/a/of/...) are REMOVED: on a toy corpus they
     dominate the co-occurrence matrix and drown the semantic signal — the
     same "most frequent words contribute disproportionately" problem that
     GloVe's weighting function is designed to fix on real corpora.
Outputs: corpus.txt, vocab.json, X.npz
"""
import numpy as np
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

T = []
def add(s, k):
    T.append((s, k))

# ===== thermodynamic phase group (drives the ice/steam ratio table) =====
for s in [
    "ice solid cold",
    "steam gas hot",
    "water liquid wet",
    "ice melts water",
    "water freezes ice",
    "steam condenses water",
    "water boils steam",
    "solid ice lake",
    "hot steam kettle",
    "cold water glass",
    "winter ice snow",
    "snow melts water",
    "ice snow winter",
    "gas steam rise",
    "liquid water flows",
]:
    add(s, 400)
# gas: mostly with steam, a few weak co-occurrences with ice (small, nonzero ratio)
add("ice gas different", 10)
# fashion: rare, mildly co-occurs with BOTH ice and steam -> ratio ~ 1
for s in [
    "fashion style change",
    "fashion snow winter",
    "ice fashion show",
    "steam fashion paris",
]:
    add(s, 40)

# ===== gender group: he/she define the gender axis =====
for s in [
    "he man", "she woman",
    "he king", "she queen",
    "he boy", "she girl",
    "he brother", "she sister",
    "he uncle", "she aunt",
]:
    add(s, 300)
# royal role (shared by king/queen only)
for s in [
    "king rules land",
    "queen rules land",
    "king wears crown",
    "queen wears crown",
]:
    add(s, 300)
# human role (shared by man/woman only)
for s in [
    "man works daily",
    "woman works daily",
    "man runs daily",
    "woman runs daily",
    "man reads daily",
    "woman reads daily",
]:
    add(s, 300)
# child role (shared by boy/girl only)
for s in [
    "boy plays outside",
    "girl plays outside",
    "boy laughs outside",
    "girl laughs outside",
]:
    add(s, 300)
# sibling role (shared by brother/sister only)
for s in [
    "brother helps family",
    "sister helps family",
    "brother visits family",
    "sister visits family",
]:
    add(s, 300)
# elder role (shared by uncle/aunt only); kept at VERY LOW frequency so the
# shared words do not become high-norm hubs that dominate analogy queries
for s in [
    "uncle drives car",
    "aunt drives car",
    "uncle cooks food",
    "aunt cooks food",
]:
    add(s, 40)
# distinct elder-role contexts: makes uncle/aunt separable from brother/sister
# (without this, the drives/cooks hub words dominate the difference vector)
for s in ["uncle teaches", "aunt bakes"]:
    add(s, 1200)

# ===== country-capital group =====
for s in [
    "paris city", "berlin city", "rome city",
    "madrid city", "london city", "tokyo city",
]:
    add(s, 300)
for s in [
    "france country", "germany country", "italy country",
    "spain country", "england country", "japan country",
]:
    add(s, 300)
for s in [
    "paris capital france",
    "berlin capital germany",
    "rome capital italy",
    "madrid capital spain",
    "london capital england",
    "tokyo capital japan",
]:
    add(s, 300)

# ===== verb-tense group: base vs -ing forms share parallel markers =====
# shared tense markers are kept VERY LOW frequency (they are the "hub words"
# that otherwise dominate the difference vectors); the distinctive per-verb
# context words carry the semantic signal, and the -ing variant is even
# stronger so the tense direction is fully determined by them
for s in [
    "run fast daily", "walk fast daily", "dance fast daily",
    "swim fast daily", "fly fast daily",
]:
    add(s, 60)
for s in [
    "running fast now", "walking fast now", "dancing fast now",
    "swimming fast now", "flying fast now",
]:
    add(s, 60)
for s in [
    "running fun", "walking fun", "dancing fun",
    "swimming fun", "flying fun",
]:
    add(s, 10)
# each verb also gets its own distinguishing context word (co-occurs with
# BOTH the base and the -ing form of that verb only), so the five verbs
# have separable difference vectors instead of collapsing to one point
for s in [
    "run sprint", "walk stroll", "dance waltz", "swim pool", "fly flight",
]:
    add(s, 800)
for s in [
    "running sprint", "walking stroll", "dancing waltz", "swimming pool", "flying flight",
]:
    add(s, 1200)

# ---------- build raw corpus ----------
tokens = []
for s, k in T:
    tokens += s.split() * k
with open(os.path.join(HERE, "corpus.txt"), "w") as f:
    f.write(" ".join(tokens) + "\n")
print("corpus tokens:", len(tokens))

# ---------- vocabulary ----------
cnt = Counter(tokens)
vocab = [w for w, _ in cnt.most_common()]
V = len(vocab)
w2i = {w: i for i, w in enumerate(vocab)}
print("vocab size:", V)

# ---------- co-occurrence matrix X (symmetric window, 1/d weighting) ----------
window = 2
X = np.zeros((V, V), dtype=np.float64)
for c, tok in enumerate(tokens):
    i = w2i[tok]
    for d in range(1, window + 1):
        wgt = 1.0 / d
        if c - d >= 0:
            j = w2i[tokens[c - d]]
            X[i, j] += wgt
            X[j, i] += wgt
        if c + d < len(tokens):
            j = w2i[tokens[c + d]]
            X[i, j] += wgt
            X[j, i] += wgt

rows, cols = np.nonzero(X)
vals = X[rows, cols]
np.savez(os.path.join(HERE, "X.npz"), rows=rows, cols=cols, vals=vals, V=V)
with open(os.path.join(HERE, "vocab.json"), "w") as f:
    json.dump(w2i, f, ensure_ascii=False)

# ---------- replicate the ice/steam ratio table (paper Table 1) ----------
def P(k_idx, given_idx):
    return X[given_idx, k_idx] / X[given_idx].sum()

print("\n--- ice/steam co-occurrence probability ratio (toy corpus) ---")
print("k       P(k|ice)      P(k|steam)    P(k|ice)/P(k|steam)")
for k in ["solid", "gas", "water", "fashion"]:
    a = P(w2i[k], w2i["ice"])
    b = P(w2i[k], w2i["steam"])
    print(f"{k:<8} {a:.3e}   {b:.3e}   {a / b:.3f}")
