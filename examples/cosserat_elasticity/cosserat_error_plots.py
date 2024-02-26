import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.markers as plt_markers
import numpy as np
from pathlib import Path

# static maps
method_map = {'sc_rt': 'SC-RT', 'sc_bdm': 'SC-BDM', 'wc_rt': 'WC-RT', 'wc_bdm': 'WC-BDM'}
method_color_map = {'sc_rt': mcolors.TABLEAU_COLORS['tab:blue'], 'sc_bdm': mcolors.TABLEAU_COLORS['tab:orange'], 'wc_rt': mcolors.TABLEAU_COLORS['tab:green'], 'wc_bdm': mcolors.TABLEAU_COLORS['tab:red']}

mat_values_map = {'0.0001': '10^{-4}', '0.01': '10^{2}', '1.0': '10^{0}',
                  '100.0': '10^{2}', '10000.0': '10^{4}'}
markers_values_map = {'0.0001': "v", '0.01': "s", '1.0': "o",
                  '100.0': "s", '10000.0': "v"}

convergence_type_map = {'normal': np.array([2,3,4,5]), 'super': np.array([2,3,4,5])}

def filter_composer(method, m_lambda, m_eps, k, d):
    filter_0 = method
    filter_1 = '_lambda_' + str(m_lambda)
    filter_2 = '_gamma_' + str(m_eps)
    filter_3 = '_k' + str(k)
    filter_4 = '_' + str(d) + 'd'
    filter = filter_0 + filter_1 + filter_2 + filter_3 + filter_4
    return  filter

# Set the range of x-axis
p=Path()
ex_1_file_names = list(p.glob('output_example_1/*_error_ex_1.txt'))

dim = 2
aka = 0

conv_type = 'normal'
m_lambda = 1.0
k = 0
d = 2
mat_label = '\epsilon'

fig, ax = plt.subplots()

method = 'sc_rt'
m_eps = 1.0

filter = filter_composer(method=method, m_lambda=m_lambda, m_eps=m_eps, k=k, d=d)
result = [(idx, path.name) for idx, path in enumerate(ex_1_file_names) if (filter in path.name)]
assert len(result) == 1
label = r'$'+mat_label+' = '+ mat_values_map[str(m_eps)]+ '$'
marker = markers_values_map[str(m_eps)]
color = method_color_map[method]
file_name = str(ex_1_file_names[result[0][0]])
rdata = np.genfromtxt(file_name, dtype=None, delimiter=',', skip_header=1)
idxs = convergence_type_map[conv_type]

h = rdata[:,np.array([1])]
error = np.sum(rdata[:,idxs],axis=1)
plt.loglog(h, error, label=label, marker=marker,color=color)

plt.xlim(np.min(h)/1.1, np.max(h)*1.1)
plt.xlabel(r"$h$")
plt.ylabel("Error")
plt.ylim(0.005, 20)
plt.text(0.25, 1.0, r"$\mathbf{1}$")
plt.legend()
# plt.show()

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