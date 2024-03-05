import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

# epsilon function when z = 0
def epsilon(x, y):
    value = np.min(
            [
                np.ones_like(x),
                np.max(
                    [
                        np.zeros_like(x),
                        np.max([3 * x, 3 * y], axis=0) - np.ones_like(x),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        )
    return value


x_vals = np.linspace(0.0, 1.0, 100)
y_vals = np.linspace(0.0, 1.0, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = epsilon(X,Y)
fig,ax=plt.subplots(1,1, subplot_kw=dict(aspect='equal'))
cp = ax.contourf(X, Y, Z, cmap='gray', extend='both', alpha=0.5, levels= 15)
ax.annotate(r"$\epsilon = 1$",
            xy=(0.75, 0.75), xycoords='data',
            fontsize=12)
ax.annotate(r"$\epsilon = 0$",
            xy=(0.15, 0.15), xycoords='data',
            fontsize=12)
ax.annotate(r"$\epsilon = x_1$",
            xy=(0.45, 0.15), xycoords='data',
            fontsize=12)
ax.annotate(r"$\epsilon = x_2$",
            xy=(0.15, 0.5), xycoords='data',
            fontsize=12)
plt.xticks([])
plt.yticks([])

plt.savefig("figures/figure_spatial_epsilon.pdf", bbox_inches='tight')

