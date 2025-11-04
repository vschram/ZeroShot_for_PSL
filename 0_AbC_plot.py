import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif"
# })
fontsize = 16
fontsize_legend = fontsize-2
# C values (x-axis), on log scale
c = np.logspace(13, 23, 500)

# compute L values from the two equations
logL1 = -0.097 * np.log(c) + 5.32
logL2 = -0.029 * np.log(c) + 2.96

L1 = logL1
L2 = logL2

# create figure
plt.figure(figsize=(5, 5))

# plot both curves
plt.plot(c, L1, 'b:', label=r'Ground Truth' + '\n' + r'$L_{\log}^{SL} = -0.097 C_{\log} + 5.32$')
plt.plot(c, L2, 'b--', label=r'Predicted' + '\n' + r'$L_{\log}^{SL}  = -0.029 C_{\log} + 2.96$')

# fill between
plt.fill_between(c, L1, L2,  interpolate=True, color='lavender', label='Area Between Curves')

# log-log scale
plt.xscale('log')
#plt.yscale('log')
plt.xlim(1e13, 1e23)
plt.ylim(1e0, 3)

# labels, legend, grid
plt.xlabel(r'Compute', fontsize=fontsize)
plt.ylabel(r'Loss', fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# Adjust layout
plt.legend(fontsize=fontsize_legend)
plt.tight_layout()
#plt.savefig("ABC2.png")
#plt.savefig("ABC2.pdf")
plt.show()
