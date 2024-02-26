import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as plt_markers
import numpy as np
from pathlib import Path

# Set the range of x-axis
p=Path()
ex_1_file_names = list(p.glob('output_example_1/*_error_ex_1.txt'))
file_name = str(ex_1_file_names[0])
rdata = np.genfromtxt(file_name, dtype=None, delimiter=',', skip_header=1)

fig = plt.figure()
h = rdata[:,np.array([1])]
error = rdata[:,np.array([2])]
plt.xlim(np.min(h)/1.1, np.max(h)*1.1)
plt.ylim(0.005, 20)
plt.loglog(h, error, label="Error", marker=plt_markers.CARETDOWNBASE)

aka = 0
# Create plots in 2x2 grid
for plot in range(4):
    # Create plots
    x = np.arange(0, 10, 0.1)
    y = np.random.randn(len(x))
    y2 = np.random.randn(len(x))
    ax = fig.add_subplot(2,2,plot+1)
    plt.loglog(x, y, label="y")
    plt.loglog(x, y2, label="y2")

aka = 0
# Create custom legend
blue_line = mlines.Line2D([], [], color='blue',markersize=15, label='Blue line')
green_line = mlines.Line2D([], [], color='green', markersize=15, label='Green line')
ax.legend(handles=[blue_line,green_line],bbox_to_anchor=(1.05, 0),  loc='lower left', borderaxespad=0.)
aka = 0