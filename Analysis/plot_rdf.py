import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
Z=20
experimental = [1.15, 1.30, 1.45, 2.00, 2.15]

def read_file(fname):
    f = open(fname, 'r')
    fl = f.readlines()
    f.close()
    clean = np.array([line.split() for line in fl if "@" not in line and "#" not in line]).astype('float')
    return clean

def cum_rdf(rdf, ndx):
    R = rdf[:,0]
    dr = R[1] - R[0]
    ff = np.multiply(np.power(R,2), rdf[:,ndx])*dr
    ff = np.cumsum(ff)
    ff /= ff[-1]
    return ff

NP = [1, 2, 3, 4, 5]
labels = ["LIG{}".format(nn) for nn in NP]
colors = cm.hot(np.linspace(0.1,0.8,len(NP)))

data = np.array([read_file("NP{}/NP{}_NPT_rdf.sfu".format(nn, nn)) for nn in NP])
widths = []

fig, axs = plt.subplots(figsize=(12,6), ncols=2, gridspec_kw={'wspace':0.3})
for rdfs, nn, lab, c in zip(data, NP, labels, colors):
    cum = cum_rdf(rdfs, 1)
    mask = np.logical_and(cum>0.05, cum<0.95)
    width = rdfs[mask,0][-1] - rdfs[mask,0][0]
    widths.append(width)
    print("NP{}: 95% interval has a width of {:.3f} nm".format(nn, width))
    axs[0].errorbar(rdfs[:,0], rdfs[:,1], fmt='-', color=c, mec='k', mew=1.5, ms=10, lw=2, label=lab, alpha=0.8)
    axs[1].errorbar(rdfs[:,0], cum, fmt='-', color=c, mec='k', mew=1.5, ms=10, lw=2, label=lab, alpha=0.8)
for ax in axs:
    ax.set_xlabel("Distance from C.O.M (nm)", fontsize=Z)
    ax.tick_params(labelsize=Z)
    ax.set_xlim(4,9)
    ax.grid()
axs[0].legend(fontsize=Z-2, loc='upper right')#, bbox_to_anchor=(1,0.5))
axs[0].set_ylabel("R.D.F.", fontsize=Z)
axs[1].set_ylabel("Cumulative\nNormalized R.D.F", fontsize=Z)
plt.tight_layout()
plt.savefig("rdf.png", format='png', dpi=300, bbox_inches="tight")
#plt.show()
plt.close()

XX = np.linspace(1,5,5)
fig, ax = plt.subplots(figsize=(6,6))
ax.errorbar(XX, widths, fmt='o-', color=(0.2,0.6,1.0), mec='k', mew=1.5, ms=10, lw=2, label="CG")
ax.errorbar(XX, experimental, fmt='^--', color=(1.0,0.0,0.5), mec='k', mew=1.5, ms=10, lw=2, label="Exp.")
ax.legend(fontsize=Z-4, loc='upper left')
ax.set_ylim(1,2.75)
ax.set_xlim(0,6)
#ax.set_xlabel("Ligand", fontsize=Z)
ax.set_xticks(XX)
ax.set_xticklabels(["LIG{}".format(nn) for nn in NP], rotation=30)
ax.set_ylabel("Monolayer width (nm)", fontsize=Z)
ax.tick_params(labelsize=Z)
plt.tight_layout()
plt.savefig("compare.png", format='png', dpi=300, bbox_inches="tight")
#plt.show()
plt.close()
