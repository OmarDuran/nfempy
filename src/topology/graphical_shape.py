import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def draw_vertex_sequence(color_id, vertex_sequence, close_q=False):
    fig = plt.figure()
    color_names_raw = list(mcolors.CSS4_COLORS.keys())
    color_names = [color for color in color_names_raw if "white" not in color]
    color_names.reverse()
    ax = fig.add_subplot(111, projection="3d")
    color_rgba = mcolors.to_rgba(color_names[color_id])
    for sequence in vertex_sequence:
        points = []
        for vertex in sequence:
            points.append(vertex.point)
        points = np.vstack([points])
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        ax.plot(x, y, z, "o-", color=color_rgba)
        if close_q and len(sequence) > 2:
            ax.plot([x[-1], x[0]], [y[-1], y[0]], [z[-1], z[0]], "o-", color=color_rgba)

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
