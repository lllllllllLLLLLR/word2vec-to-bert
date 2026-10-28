"""
GloVe essay lab - Step 2: train GloVe from scratch (numpy only), per the paper:

  J = sum_{i,j : X_ij>0} f(X_ij) * ( w_i . w~_j + b_i + b~_j - log X_ij )^2
  f(x) = (x/x_max)^alpha  if x < x_max  else 1      (x_max=100, alpha=3/4)

Optimizer: AdaGrad, lr 0.05, iterate over nonzero entries (paper Sec 4.2).
Final vector: W + W~ (paper Sec 4.2).

Outputs (figures/): loss curve, 3D loss surface + SGD trajectory ("walking
down the valley"), PCA of trained vectors, ice/steam ratio replication.
Also prints analogy / nearest-neighbour evaluations.
"""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
np.random.seed(0)

# ---------- load corpus statistics ----------
w2i = json.load(open(os.path.join(HERE, "vocab.json")))
V = len(w2i)
i2w = {i: w for w, i in w2i.items()}
d_ = np.load(os.path.join(HERE, "X.npz"))
rows, cols, vals = d_["rows"], d_["cols"], d_["vals"]
X = np.zeros((V, V))
X[rows, cols] = vals

# ---------- GloVe model ----------
d = 50
xmax = 100.0
alpha = 0.75
lr = 0.05
num_iter = 300

def f(x):
    return np.where(x < xmax, (x / xmax) ** alpha, 1.0)

W = np.random.uniform(-0.5, 0.5, (V, d)) * 0.01
Wt = np.random.uniform(-0.5, 0.5, (V, d)) * 0.01
b = np.zeros(V)
bt = np.zeros(V)

# AdaGrad accumulators
gw2 = np.zeros_like(W); gwt2 = np.zeros_like(Wt)
gb2 = np.zeros(V); gbt2 = np.zeros(V)

pairs = list(zip(rows, cols, vals))
logX = np.log(vals)
nz = len(pairs)

ice_i, steam_i = w2i["ice"], w2i["steam"]
track = []            # (iter, full loss)
traj = []             # (W[ice,0], W[steam,0]) sampled per iteration

def full_loss():
    diff = np.sum(W[rows] * Wt[cols], axis=1) + b[rows] + bt[cols] - logX
    return float(np.sum(f(vals) * diff * diff))

for it in range(num_iter):
    order = np.random.permutation(nz)
    for t in range(nz):
        p = order[t]
        i, j, x = rows[p], cols[p], vals[p]
        g = f(x) * (W[i].dot(Wt[j]) + b[i] + bt[j] - logX[p])
        # gradients
        gw = g * Wt[j]; gwt = g * W[i]
        gb = g; gbt = g
        gw2[i] += gw * gw; gwt2[j] += gwt * gwt
        gb2[i] += gb * gb; gbt2[j] += gbt * gbt
        W[i] -= lr * gw / np.sqrt(gw2[i] + 1e-6)
        Wt[j] -= lr * gwt / np.sqrt(gwt2[j] + 1e-6)
        b[i] -= lr * gb / np.sqrt(gb2[i] + 1e-6)
        bt[j] -= lr * gbt / np.sqrt(gbt2[j] + 1e-6)
    track.append((it, full_loss()))
    traj.append((W[ice_i, 0], W[steam_i, 0]))

final = W + Wt
np.savez(os.path.join(HERE, "glove_vecs.npz"), W=final)

loss0 = track[0][1]; loss_f = track[-1][1]
print(f"GloVe training done: loss {loss0:.1f} -> {loss_f:.1f} over {num_iter} iters, {nz} nonzero pairs")

# ---------- 1) loss curve ----------
plt.figure(figsize=(6.5, 4))
xs = [t[0] for t in track]; ys = [t[1] for t in track]
plt.plot(xs, ys, lw=2)
plt.xlabel("iteration"); plt.ylabel("J (weighted least squares)")
plt.title("GloVe training loss (toy corpus)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_loss_curve.png"), dpi=140); plt.close()

# ---------- 2) 3D loss surface + SGD trajectory ----------
# To make the "valley descent" visible, we train a 1-D GloVe (each word is a
# scalar): then the whole optimization is projected onto the (w_ice, w_steam)
# plane, and we can draw the true objective J over that plane.
lr1 = 0.05
w = np.random.uniform(-0.5, 0.5, V) * 0.01
wt = np.random.uniform(-0.5, 0.5, V) * 0.01
b1 = np.zeros(V); bt1 = np.zeros(V)
gw2 = np.zeros(V); gwt2 = np.zeros(V); gb2 = np.zeros(V); gbt2 = np.zeros(V)
traj1 = []
for it in range(120):
    order = np.random.permutation(nz)
    for t in range(nz):
        p = order[t]
        i, j, x = rows[p], cols[p], vals[p]
        g = f(x) * (w[i] * wt[j] + b1[i] + bt1[j] - logX[p])
        gw = g * wt[j]; gwt = g * w[i]
        gw2[i] += gw * gw; gwt2[j] += gwt * gwt
        gb2[i] += g * g; gbt2[j] += g * g
        w[i] -= lr1 * gw / np.sqrt(gw2[i] + 1e-6)
        wt[j] -= lr1 * gwt / np.sqrt(gwt2[j] + 1e-6)
        b1[i] -= lr1 * g / np.sqrt(gb2[i] + 1e-6)
        bt1[j] -= lr1 * g / np.sqrt(gbt2[j] + 1e-6)
    traj1.append((w[ice_i], b1[ice_i]))

ice_terms = np.where(rows == ice_i)[0]
cols_of_ice = cols[ice_terms]

def loss_d1(ai, bi):
    # other parameters frozen at final values; only w_ice and b_ice vary
    # slice: J(w_ice, b_ice) = sum_j f(X_ij) (w_ice*wt_j + b_ice + bt_j - log X_ij)^2
    diff = ai * wt[cols_of_ice] + bi + bt1[cols_of_ice] - logX[ice_terms]
    return np.sum(f(vals[ice_terms]) * diff * diff)

a0, b0 = traj1[0]
aE, bE = traj1[-1]
pa = np.linspace(aE - 0.8, aE + 0.8, 48)
pb = np.linspace(bE - 0.8, bE + 0.8, 48)
AA, BB = np.meshgrid(pa, pb)
LL = np.array([[loss_d1(ai, bi) for ai in pa] for bi in pb])
tz = [loss_d1(t[0], t[1]) for t in traj1]
print(f"1-D GloVe loss at traj start/end: {tz[0]:.0f} -> {tz[-1]:.0f}")

fig = plt.figure(figsize=(9, 6.5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(AA, BB, LL, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
ax.plot([t[0] for t in traj1], [t[1] for t in traj1], tz, color="red", lw=2.2,
        label="AdaGrad trajectory")
ax.scatter([a0], [b0], [tz[0]], color="red", s=60, label="init")
ax.scatter([aE], [bE], [tz[-1]], color="black", s=70, label="converged")
ax.set_xlabel("w_ice (d=1)"); ax.set_ylabel("b_ice (bias)"); ax.set_zlabel("J")
ax.set_title("1-D GloVe objective J(w_ice, b_ice): the valley AdaGrad walks down")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_3d_surface_trajectory.png"), dpi=140); plt.close()

# ---------- 3) ratio replication ----------
print("\n--- model-predicted vs empirical ratio (ice/steam) ---")
print("k       emp P-ratio   emp count-ratio   model count-ratio")
for k in ["solid", "gas", "water", "fashion"]:
    ki = w2i[k]
    p_ice = X[ice_i, ki] / X[ice_i].sum()
    p_steam = X[steam_i, ki] / X[steam_i].sum()
    emp_p = p_ice / p_steam
    emp_c = X[ice_i, ki] / X[steam_i, ki]
    m_ik = final[ice_i].dot(Wt[ki]) + b[ice_i] + bt[ki]
    m_jk = final[steam_i].dot(Wt[ki]) + b[steam_i] + bt[ki]
    print(f"{k:<8} {emp_p:>10.3f}   {emp_c:>12.3f}   {np.exp(m_ik) / np.exp(m_jk):>14.3f}")

# ---------- 4) PCA of trained vectors ----------
groups = {
    "phase": ["ice", "steam", "water", "solid", "liquid", "gas", "cold", "hot",
              "melts", "boils", "freezes", "condenses", "snow", "winter", "lake",
              "kettle", "glass", "rise", "flows", "fashion"],
    "gender": ["king", "queen", "man", "woman", "brother", "sister", "boy",
               "girl", "uncle", "aunt", "he", "she"],
    "capital": ["paris", "berlin", "rome", "madrid", "london", "tokyo"],
    "country": ["france", "germany", "italy", "spain", "england", "japan"],
    "verb": ["run", "running", "walk", "walking", "dance", "dancing",
             "swim", "swimming", "fly", "flying"],
}
sel = [w2i[w] for g in groups.values() for w in g]
names = [i2w[s] for s in sel]
colormap = {"phase": "tab:blue", "gender": "tab:orange", "capital": "tab:green",
            "country": "tab:red", "verb": "tab:purple"}
cols_ = [colormap[g] for g in groups for _ in groups[g]]

M = final[sel]
M = M - M.mean(0)
U, S, Vt = np.linalg.svd(M, full_matrices=False)
P2 = M @ Vt[:2].T

plt.figure(figsize=(8, 7))
plt.scatter(P2[:, 0], P2[:, 1], c=cols_, s=40)
for i, n in enumerate(names):
    plt.annotate(n, (P2[i, 0], P2[i, 1]), fontsize=7.5)
plt.title("PCA of GloVe vectors (toy corpus, colored by semantic group)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_pca_words.png"), dpi=140); plt.close()

# ---------- 5) analogies + nearest neighbours ----------
def cosine(a, b):
    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def top_k(target, exclude, k=5):
    sc = []
    for wi, w in enumerate(i2w):
        if wi in exclude:
            continue
        sc.append((cosine(target, final[wi]), wi, w))
    sc.sort(reverse=True)
    return sc[:k]

print("\n--- diagnostic: norms of hubs ---")
norms = sorted(((np.linalg.norm(final[i]), i2w[i]) for i in range(V)), reverse=True)[:12]
print(" ".join(f"{w}({n:.2f})" for n, w in norms))

print("\n--- diagnostic: king-queen :: man, top candidates (raw) ---")
t = final[w2i["queen"]] - final[w2i["king"]] + final[w2i["man"]]
for s, wi, w in top_k(t, {w2i["king"], w2i["queen"], w2i["man"]}):
    print(f"  {w} {s:.3f}")

final_c = final - final.mean(0)   # centered: remove the common/hub direction
print("\n--- diagnostic: king-queen :: man, top candidates (centered) ---")
def cosine_c(a, b):
    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
def top_k_c(target, exclude, k=5):
    sc = []
    for wi, w in enumerate(i2w):
        if wi in exclude:
            continue
        sc.append((cosine_c(target, final_c[wi]), wi, w))
    sc.sort(reverse=True)
    return sc[:k]
t = final_c[w2i["queen"]] - final_c[w2i["king"]] + final_c[w2i["man"]]
for s, wi, w in top_k_c(t, {w2i["king"], w2i["queen"], w2i["man"]}):
    print(f"  {w} {s:.3f}")

def analogy(a, b, c, exclude=()):
    target = final[w2i[b]] - final[w2i[a]] + final[w2i[c]]
    best, best_w = -2, None
    for wi in range(V):
        if wi in exclude:
            continue
        sc = cosine(target, final[wi])
        if sc > best:
            best, best_w = sc, i2w[wi]
    return best_w

analogies = []
for a, b, c, d_exp in [
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
]:
    e = {w2i[a], w2i[b], w2i[c]}
    pred = analogy(a, b, c, exclude=e)
    analogies.append((a, b, c, d_exp, pred == d_exp))

acc = np.mean([x[4] for x in analogies])
print(f"\nGloVe analogy accuracy: {acc * 100:.1f}% ({sum(x[4] for x in analogies)}/{len(analogies)})")
for a, b, c, d_exp, ok in analogies:
    pred = analogy(a, b, c, exclude={w2i[a], w2i[b], w2i[c]})
    print(f"  {a}-{b} :: {c} -> {pred}  (expected {d_exp}) {'OK' if ok else 'x'}")

print("\n--- GloVe nearest neighbours ---")
for w in ["ice", "steam", "water", "paris", "king", "running", "fashion"]:
    wi = w2i[w]
    sims = sorted(((cosine(final[wi], final[j]), j) for j in range(V) if j != wi),
                  reverse=True)[:5]
    print(f"  {w}: " + ", ".join(f"{i2w[j]}({s:.3f})" for s, j in sims))
