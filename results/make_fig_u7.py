# U=7 的 Fig S2 式对比图：三通道关联函数 vs 距离（按距离合并位点）
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "/mnt/agents/lecqmc/results/ed_benchmark_L4_U7.txt"
out = sys.argv[2] if len(sys.argv) > 2 else "/mnt/agents/lecqmc/results/figures/fig_s2_ed_U7.png"

rows = []
for ln in open(src):
    if ln.startswith("#"):
        continue
    rows.append([float(x) for x in ln.split()])
d = np.array(rows)
r, ed_nn, mc_nn, e_nn = d[:,1], d[:,2], d[:,3], d[:,4]
ed_szz, mc_szz, e_szz = d[:,5], d[:,6], d[:,7]
ed_sxy, mc_sxy, e_sxy = d[:,8], d[:,9], d[:,10]

def merge(r, *cols):
    out = []
    for rv in sorted(set(r)):
        m = r == rv
        out.append([rv] + [c[m].mean() for c in cols])
    return np.array(out).T

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, ed, mc, err, lab in [
    (axes[0], ed_nn, mc_nn, e_nn, r"$\langle n_1 n_j\rangle$"),
    (axes[1], ed_szz, mc_szz, e_szz, r"$\langle S^z_1 S^z_j\rangle$"),
    (axes[2], ed_sxy, mc_sxy, e_sxy, r"$\langle S^x_1S^x_j+S^y_1S^y_j\rangle$"),
]:
    rv, edv, mcv, ev = merge(r, ed, mc, err)
    ax.plot(rv, edv, "s--", ms=6, mfc="none", color="tab:blue", label="ED (exact)", lw=1)
    ax.errorbar(rv, mcv, yerr=ev, fmt="o", color="tab:red", ms=6, capsize=3, label="LEC-QMC")
    ax.set_xlabel("distance r"); ax.set_ylabel(lab); ax.legend(fontsize=8)
fig.suptitle("2D Hubbard L=4, Ne=2, U=7, T=0.05 (LEC-QMC vs ED)")
fig.tight_layout()
fig.savefig(out, dpi=130)
print("saved", out)
