import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

R = "/mnt/agents/lecqmc/results"
F = R + "/figures"

# ---------- Fig S2: ED vs QMC correlators ----------
d = np.loadtxt(R + "/ed_benchmark_L4.txt", comments="#")
j, r = d[:,0], d[:,1]
ed_nn, mc_nn, e_nn = d[:,2], d[:,3], d[:,4]
ed_szz, mc_szz, e_szz = d[:,5], d[:,6], d[:,7]
ed_sxy, mc_sxy, e_sxy = d[:,8], d[:,9], d[:,10]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, ed, mc, e, name in [(axes[0], ed_nn, mc_nn, e_nn, r"$\langle n_1 n_j\rangle$"),
                            (axes[1], ed_szz, mc_szz, e_szz, r"$\langle S^z_1 S^z_j\rangle$"),
                            (axes[2], ed_sxy, mc_sxy, e_sxy, r"$\langle S^x_1 S^x_j+S^y_1 S^y_j\rangle$")]:
    order = np.argsort(r)
    ax.errorbar(r[order], mc[order], yerr=e[order], fmt="o", ms=5, color="#c0392b", label="LEC-QMC", capsize=3)
    ax.plot(r[order], ed[order], "s", ms=6, mfc="none", mec="#2c3e50", ls="--", lw=1, label="ED (exact)")
    ax.set_xlabel("distance r"); ax.set_ylabel(name); ax.legend(fontsize=8)
fig.suptitle("2D Hubbard L=4, Ne=2, U=2, T=0.05 (LEC-QMC vs ED)")
fig.tight_layout(); fig.savefig(F + "/fig_s2_ed.png", dpi=160); plt.close(fig)

# ---------- Fig 1: scaling (quiet-system re-measure) ----------
d = np.loadtxt(R + "/scaling_clean.txt", comments="#")
# merge possible duplicate N rows (resume runs): take first non-NaN per column
merged = {}
for row in d:
    n = int(row[1])
    cur = merged.setdefault(n, [row[0], np.nan, np.nan])
    if not np.isnan(row[2]): cur[1] = row[2]
    if not np.isnan(row[3]): cur[2] = row[3]
Ns = sorted(merged)
N = np.array(Ns, float); L = np.array([merged[n][0] for n in Ns])
tlec = np.array([merged[n][1] for n in Ns]); tdq = np.array([merged[n][2] for n in Ns])
fig, ax = plt.subplots(figsize=(6.5, 5))
mask = ~np.isnan(tdq)
ax.loglog(N, tlec, "o", color="#c0392b", ms=6, label="LEC-QMC ($N_e$=100)")
ax.loglog(N[mask], tdq[mask], "s", color="#2c3e50", ms=6, mfc="none", label="DQMC (fast update)")
# fits
s1, i1, *_ = linregress(np.log(N[N>=400]), np.log(tlec[N>=400]))
s3, i3, *_ = linregress(np.log(N[mask][2:]), np.log(tdq[mask][2:]))
xx = np.array([100, 1.2e4])
ax.loglog(xx, np.exp(i1)*xx**s1, "--", color="#c0392b", alpha=.6, label=f"slope {s1:.2f}")
ax.loglog(xx, np.exp(i3)*xx**s3, "--", color="#2c3e50", alpha=.6, label=f"slope {s3:.2f}")
ax.set_xlabel("N"); ax.set_ylabel("CPU time per sweep (s)")
ax.legend(fontsize=9, loc="upper left")
ax.set_title(r"Sweep cost vs system size ($\Delta\tau$=0.1, U/t=2, T/t=1)")
fig.tight_layout(); fig.savefig(F + "/fig1_scaling.png", dpi=160); plt.close(fig)
print("scaling slopes: LEC", s1, " DQMC", s3)

# ---------- Fig 2: sign recovery ----------
d = np.loadtxt(R + "/sign.txt", comments="#")
fig, ax = plt.subplots(figsize=(6.5, 5))
for (T, Ne), c, mk in [((0.05, 50), "#2980b9", "o"), ((0.05, 100), "#c0392b", "s"), ((0.02, 50), "#27ae60", "^")]:
    m = (d[:,0] == T) & (d[:,1] == Ne)
    ax.plot(d[m,3], d[m,4], mk+"-", color=c, ms=6, lw=1.2, label=f"T={T}, $N_e$={Ne}")
ax.set_xlabel("1/L"); ax.set_ylabel(r"$\langle\mathrm{sign}\rangle$")
ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9)
ax.set_title("Sign recovery in the dilute limit (U=2)")
fig.tight_layout(); fig.savefig(F + "/fig2_sign.png", dpi=160); plt.close(fig)

# ---------- Fig 4: flat band ----------
lines = open(R + "/flatband.txt").read().splitlines()
ed = {}; qmc = {}
for ln in lines:
    if ln.startswith("#") or not ln.strip(): continue
    p = ln.split()
    if p[0] == "ED4":
        ed[float(p[1])] = (float(p[2]), float(p[3]))
    else:
        L_, T_ = int(p[0]), float(p[1])
        qmc.setdefault(L_, []).append((T_, float(p[2]), float(p[3]), float(p[4]), float(p[5])))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
colors = {4: "#c0392b", 8: "#2980b9", 12: "#27ae60", 16: "#8e44ad"}
for L_, pts in sorted(qmc.items()):
    pts.sort()
    T_ = [p[0] for p in pts]
    axes[0].errorbar(T_, [p[1] for p in pts], yerr=[p[2] for p in pts], fmt="o-", ms=4, lw=1.2, color=colors[L_], label=f"L={L_}", capsize=2)
    axes[1].errorbar(T_, [p[3] for p in pts], yerr=[p[4] for p in pts], fmt="s-", ms=4, lw=1.2, color=colors[L_], label=f"L={L_}", capsize=2)
Te = sorted(ed)
axes[0].plot(Te, [ed[t][0] for t in Te], "k--", lw=1.5, label="ED (L=4)")
axes[1].plot(Te, [ed[t][1] for t in Te], "k--", lw=1.5, label="ED (L=4)")
axes[1].axhline(1/16, color="gray", ls=":", lw=1.5, label="1/16")
axes[0].set_xlabel("T"); axes[0].set_ylabel(r"$S_{\parallel}(q=0)$"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("T"); axes[1].set_ylabel(r"$S_{zz}(q=0)$ of A1 sublattice"); axes[1].legend(fontsize=8)
fig.suptitle("1D BCL flat-band model, $t_2=t_3=1, t_1=t_4=-0.2$, U=0.5, flat-band half-filling ($N_e=L$)")
fig.tight_layout(); fig.savefig(F + "/fig4_flatband.png", dpi=160); plt.close(fig)
print("figures written")
