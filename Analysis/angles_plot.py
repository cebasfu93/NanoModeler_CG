import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
STD=True
Z = 20
cm1 = cm.viridis(np.linspace(0.2,0.8,3))
cm2 = cm.plasma(np.linspace(0.2,0.8,3))

data = np.genfromtxt("angles.sfu")
block1 = data[:9,1]
block1_std = data[:9,2]
block1 = np.reshape(block1, (3,3))
block1_std = np.reshape(block1_std, (3,3))
block2 = data[9:,1]
block2_std = data[9:,2]
block2 = np.reshape(block2, (4,3)).T
block2_std = np.reshape(block2_std, (4,3)).T

if not STD:
    block1_std = np.zeros_like(block1_std)
    block2_std = np.zeros_like(block2_std)

temps1 = [300, 600, 900]
carbons2 = [4, 8, 13, 20]
exp1 = [[48,38,34], [39,30,26], [37,25,21]]
exp2 = [[28,34.5,38.5,43], [25.5,29,36,42], [24.5,26.5,29.5,41]]

labels1 = ["D=3 nm", "D=5 nm", "D=7 nm"]
labels2 = ["T=300 K", "T=450 K", "T=600 K"]
labels = [labels1, labels2]

cms = [cm1, cm2]
blocks = [block1, block2]
blocks_stds = [block1_std, block2_std]
XX = [temps1, carbons2]
experimental = [exp1, exp2]
fig, axs = plt.subplots(figsize=(12,6), ncols=2, gridspec_kw={'wspace':0.3})
for ax, xx, exps, block, block_std, color, label in zip(axs, XX, experimental, blocks, blocks_stds, cms, labels):
    ax.tick_params(labelsize=Z)
    ax.set_ylabel("Tilt angle (deg)", fontsize=Z)
    for b, std, c, exp, lab in zip(block, block_std, color, exps, label):
        ax.errorbar(xx, b, yerr=std, fmt='o-', color=c, mec='k', mew=1.5, ms=10, lw=2, label=lab, capsize=4, capthick=2)
        ax.errorbar(xx, exp, fmt='^--', color=c, mec='k', mew=1.5, ms=10, lw=2)
    tmp1 = ax.errorbar([],[], fmt='-', color='k', mec='k', mew=1.5, ms=10, lw=2, capsize=4, capthick=2)
    tmp2 = ax.errorbar([],[], fmt='--', color='k', mec='k', mew=1.5, ms=10, lw=2)
    leg = plt.legend([tmp1, tmp2], ['CG', 'AA'], fontsize=Z, loc='upper center', bbox_to_anchor=(-0.2,1.18), ncol=2)
    plt.gca().add_artist(leg)
    ax.legend(fontsize=Z-4)

#axs[0].set_ylim(15,50)
axs[0].set_xlim(200,1000)
axs[0].set_xlabel("Temperature (K)", fontsize=Z)

#axs[1].set_ylim(20,45)
axs[1].set_xlim(2,22)
axs[1].set_xlabel("Number of carbons", fontsize=Z)

plt.tight_layout()
plt.savefig("angles.png", format='png', dpi=300, bbox_inches="tight")
plt.show()
plt.close()
