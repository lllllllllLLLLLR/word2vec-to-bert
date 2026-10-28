"""
Train GloVe word vectors (numpy implementation) on the QUAERO corpora,
with the same settings as TP1 (dim=100, min_count=1, window=5).

Follows Pennington et al. (2014) Sec 4.2:
  J = sum_{i,j: X_ij>0} f(X_ij) (w_i . w~_j + b_i + b~_j - log X_ij)^2
  f(x) = (x/x_max)^alpha if x < x_max else 1  (x_max=100, alpha=3/4)
Optimizer: AdaGrad (mini-batch over nonzero pairs), final vector = W + W~.

Outputs: models_word_vectors/{med,press}_glove.kv (gensim KeyedVectors).
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from gensim.models import KeyedVectors

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "med": ROOT / "TP_ISD2020/QUAERO_FrenchMed/QUAERO_FrenchMed_traindev.ospl",
    "press": ROOT / "TP_ISD2020/QUAERO_FrenchPress/QUAERO_FrenchPress_traindev.ospl",
}
OUT_DIR = ROOT / "models_word_vectors"
OUT_DIR.mkdir(exist_ok=True)

DIM = 100
MIN_COUNT = 1
WINDOW = 5
XMAX = 100.0
ALPHA = 0.75
LR = 0.05
NUM_ITER = 60
BATCH = 20000
SEED = 42
rng = np.random.default_rng(SEED)


def build_corpus(path):
    sentences = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            sentences.append(line.split())
    return sentences


def build_vocab(sentences):
    cnt = Counter(t for s in sentences for t in s)
    vocab = [w for w, _ in cnt.most_common() if cnt[w] >= MIN_COUNT]
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i


def build_cooccurrence(sentences, w2i, window):
    """Symmetric window co-occurrence with 1/d weighting."""
    V = len(w2i)
    data, row, col = [], [], []
    for s in sentences:
        ids = [w2i[t] for t in s if t in w2i]
        L = len(ids)
        for c in range(L):
            i = ids[c]
            for d in range(1, window + 1):
                if c - d >= 0:
                    j = ids[c - d]
                    w = 1.0 / d
                    row.append(i); col.append(j); data.append(w)
                    row.append(j); col.append(i); data.append(w)
                if c + d < L:
                    j = ids[c + d]
                    w = 1.0 / d
                    row.append(i); col.append(j); data.append(w)
                    row.append(j); col.append(i); data.append(w)
    X = sp.coo_matrix((data, (row, col)), shape=(V, V)).tocsr()
    X = X + X.T  # symmetrize (already symmetric, but double-counting fix)
    X = X.tocoo()
    X.data = np.asarray(X.data, dtype=np.float64)
    return X


def train_glove(X, V, dim, xmax, alpha, lr, num_iter, batch):
    """Mini-batch AdaGrad over nonzero pairs; returns W + W~."""
    X = X.tocoo()
    rows, cols, vals = X.row, X.col, np.asarray(X.data, dtype=np.float64)
    nz = len(vals)
    f = np.where(vals < xmax, (vals / xmax) ** alpha, 1.0)
    logX = np.log(vals)

    W = rng.uniform(-0.5, 0.5, (V, dim)).astype(np.float64) * 0.01
    Wt = rng.uniform(-0.5, 0.5, (V, dim)).astype(np.float64) * 0.01
    b = np.zeros(V)
    bt = np.zeros(V)
    gw2 = np.zeros((V, dim))
    gwt2 = np.zeros((V, dim))
    gb2 = np.zeros(V)
    gbt2 = np.zeros(V)
    eps = 1e-6

    for it in range(num_iter):
        order = rng.permutation(nz)
        perm_i, perm_j, perm_f, perm_log = rows[order], cols[order], f[order], logX[order]
        loss = 0.0
        for start in range(0, nz, batch):
            sl = slice(start, start + batch)
            i = perm_i[sl]; j = perm_j[sl]
            fl = perm_f[sl]; lx = perm_log[sl]
            diff = np.einsum("ij,ij->i", W[i], Wt[j]) + b[i] + bt[j] - lx
            g = fl * diff
            loss += float(np.sum(fl * diff * diff))
            gw = g[:, None] * Wt[j]
            gwt = g[:, None] * W[i]
            # accumulate AdaGrad sums
            np.add.at(gw2, i, gw * gw)
            np.add.at(gwt2, j, gwt * gwt)
            np.add.at(gb2, i, g * g)
            np.add.at(gbt2, j, g * g)
            # accumulate gradients, then one shared update per touched row/col
            gW = np.zeros_like(gw2)
            gWt = np.zeros_like(gwt2)
            gbA = np.zeros(V); gbtA = np.zeros(V)
            np.add.at(gW, i, gw)
            np.add.at(gWt, j, gwt)
            np.add.at(gbA, i, g)
            np.add.at(gbtA, j, g)
            upd_i = np.unique(i)
            upd_j = np.unique(j)
            W[upd_i] -= lr * gW[upd_i] / np.sqrt(gw2[upd_i] + eps)
            Wt[upd_j] -= lr * gWt[upd_j] / np.sqrt(gwt2[upd_j] + eps)
            b[upd_i] -= lr * gbA[upd_i] / np.sqrt(gb2[upd_i] + eps)
            bt[upd_j] -= lr * gbtA[upd_j] / np.sqrt(gbt2[upd_j] + eps)
        if (it + 1) % 10 == 0 or it == 0:
            print(f"  iter {it+1}/{num_iter}  loss {loss/nz:.4f}", flush=True)
    return W + Wt, loss / nz


def save_kv(vocab, matrix, out_path):
    kv = KeyedVectors(vector_size=matrix.shape[1])
    kv.add_vectors(vocab, matrix.astype(np.float32))
    kv.save(str(out_path))
    print(f"saved {out_path} ({len(vocab)} words, {matrix.shape[1]} dims)")


if __name__ == "__main__":
    for key, path in DATA.items():
        print(f"=== {key} corpus: {path.name} ===", flush=True)
        sentences = build_corpus(path)
        vocab, w2i = build_vocab(sentences)
        V = len(vocab)
        print(f"  sentences {len(sentences)}, tokens {sum(len(s) for s in sentences)}, vocab {V}", flush=True)
        X = build_cooccurrence(sentences, w2i, WINDOW)
        print(f"  nonzero co-occurrence pairs: {X.nnz}", flush=True)
        final, loss = train_glove(X, V, DIM, XMAX, ALPHA, LR, NUM_ITER, BATCH)
        save_kv(vocab, final, OUT_DIR / f"{key}_glove.kv")
