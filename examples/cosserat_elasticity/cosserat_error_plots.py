import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.markers as plt_markers
plt.rcParams["text.usetex"] =True
import numpy as np
from pathlib import Path
from abc import ABC
# static maps

class painter(ABC):

    @property
    def figure_size(self):
        return (8, 8)

    @property
    def file_name(self):
        return self.name

    @file_name.setter
    def file_name(self, name):
        self.name = name


    @property
    def ordinate_range(self):
        return self.v_range

    @ordinate_range.setter
    def ordinate_range(self, range):
        self.v_range = range

    @property
    def method_map(self):
        map = {'sc_rt': 'SC-RT', 'sc_bdm': 'SC-BDM', 'wc_rt': 'WC-RT', 'wc_bdm': 'WC-BDM'}
        return map

    @property
    def method_color_map(self):
        map = {'sc_rt': mcolors.TABLEAU_COLORS['tab:blue'], 'sc_bdm': mcolors.TABLEAU_COLORS['tab:orange'], 'wc_rt': mcolors.TABLEAU_COLORS['tab:green'], 'wc_bdm': mcolors.TABLEAU_COLORS['tab:red']}
        return map

    @property
    def mat_values_map(self):
        map = {'0.0001': '10^{-4}', '0.01': '10^{-2}', '1.0': '10^{0}',
                  '100.0': '10^{2}', '10000.0': '10^{4}'}
        return map

    @property
    def markers_values_map(self):
        map = {'0.0001': "v", '0.01': "s", '1.0': "o",
                  '100.0': "s", '10000.0': "v"}
        return map

    @property
    def convergence_type_map(self):
        map = {'normal': np.array([2,3,4,5]), 'super': np.array([3,5])}
        return map

    @staticmethod
    def filter_composer(method, m_lambda, m_eps, k, d):
        filter_0 = method
        filter_1 = '_lambda_' + str(m_lambda)
        filter_2 = '_gamma_' + str(m_eps)
        filter_3 = '_k' + str(k)
        filter_4 = '_' + str(d) + 'd'
        filter = filter_0 + filter_1 + filter_2 + filter_3 + filter_4
        return  filter

class painter_first_kind(painter):

    @property
    def m_lambda(self):
        return 1.0

    @property
    def m_epsilon(self):
        return 1.0

    def color_canvas_with_variable_epsilon(self, k, d, methods, material_values, conv_type):

        mat_label = '\epsilon'
        fig, ax = plt.subplots(figsize=self.figure_size)

        for method in methods:
            # if conv_type is 'super' and method in ['wc_rt', 'wc_bdm']:
            #     continue;
            for m_value in material_values:

                filter = painter_first_kind.filter_composer(method=method, m_lambda=self.m_lambda, m_eps=m_value,
                                         k=k, d=d)
                result = [(idx, path.name) for idx, path in enumerate(ex_1_file_names) if
                          (filter in path.name)]
                assert len(result) == 1
                label = self.method_map[method] + ': ' + r'$' + mat_label + ' = ' + \
                        self.mat_values_map[str(m_value)] + '$'
                marker = self.markers_values_map[str(m_value)]
                color = self.method_color_map[method]
                file_name = str(ex_1_file_names[result[0][0]])
                rdata = np.genfromtxt(file_name, dtype=None, delimiter=',', skip_header=1)
                idxs = self.convergence_type_map[conv_type]

                h = rdata[:, np.array([1])]
                plt.xlim(np.min(h) / 1.1, np.max(h) * 1.1)

                error = np.sum(rdata[:, idxs], axis=1)
                plt.loglog(h, error, label=label, marker=marker, color=color)

        ax.grid(True, linestyle='-.', axis='both', which='both', color='black',
                alpha=0.25)
        ax.tick_params(which='both', labelcolor='black', labelsize='large', width=2)
        plt.xlabel(r"$h$")
        plt.ylabel("Error")
        plt.ylim(self.ordinate_range[0], self.ordinate_range[1])
        # plt.ylim(0.0001, 30)
        plt.text(0.25, 1.0, r"$\mathbf{1}$")
        plt.legend()
        # plt.show()
        plt.savefig(self.file_name, format='pdf')
        aka = 0

# Set the range of x-axis
p=Path()
ex_1_file_names = list(p.glob('output_example_1/*_error_ex_1.txt'))

dim = 2
aka = 0

conv_type = 'normal'
k = 0
d = 2
methods = ['sc_rt', 'sc_bdm', 'wc_rt', 'wc_bdm']
material_values = [1.0, 0.01, 0.0001]

painter_ex_1 = painter_first_kind()
painter_ex_1.ordinate_range = (0.001, 100)

painter_ex_1.file_name = 'normal_convergence_var_eps_k0_ex_1.pdf'
painter_ex_1.color_canvas_with_variable_epsilon(0, d, methods, material_values, conv_type)

painter_ex_1.file_name = 'normal_convergence_var_eps_k1_ex_1.pdf'
painter_ex_1.color_canvas_with_variable_epsilon(1, d, methods, material_values, conv_type)

# fig, ax = plt.subplots(figsize=(8, 8))
#
# for method in ['sc_rt', 'sc_bdm', 'wc_rt', 'wc_bdm']:
#     if conv_type is 'super' and method in ['wc_rt', 'wc_bdm']:
#         continue;
#     # method = 'sc_rt'
#     for m_eps in [1.0, 0.01, 0.0001]:
#     # m_eps = 1.0
#
#         filter = filter_composer(method=method, m_lambda=m_lambda, m_eps=m_eps, k=k, d=d)
#         result = [(idx, path.name) for idx, path in enumerate(ex_1_file_names) if (filter in path.name)]
#         assert len(result) == 1
#         label = method_map[method] + ': '+ r'$'+mat_label+' = '+ mat_values_map[str(m_eps)]+ '$'
#         marker = markers_values_map[str(m_eps)]
#         color = method_color_map[method]
#         file_name = str(ex_1_file_names[result[0][0]])
#         rdata = np.genfromtxt(file_name, dtype=None, delimiter=',', skip_header=1)
#         idxs = convergence_type_map[conv_type]
#
#         h = rdata[:,np.array([1])]
#         plt.xlim(np.min(h)/1.1, np.max(h)*1.1)
#
#         error = np.sum(rdata[:,idxs],axis=1)
#         plt.loglog(h, error, label=label, marker=marker,color=color)
#
# ax.grid(True, linestyle='-.', axis='both', which='both', color='black', alpha = 0.25)
# ax.tick_params(which='both',labelcolor='black', labelsize='large', width=2)
# plt.xlabel(r"$h$")
# plt.ylabel("Error")
# plt.ylim(0.0001, 30)
# plt.text(0.25, 1.0, r"$\mathbf{1}$")
# plt.legend()
# # plt.show()
#
# aka = 0
# # Create plots in 2x2 grid
# for plot in range(4):
#     # Create plots
#     x = np.arange(0, 10, 0.1)
#     y = np.random.randn(len(x))
#     y2 = np.random.randn(len(x))
#     ax = fig.add_subplot(2,2,plot+1)
#     plt.loglog(x, y, label="y")
#     plt.loglog(x, y2, label="y2")
#
# aka = 0
# # Create custom legend
# blue_line = mlines.Line2D([], [], color='blue',markersize=15, label='Blue line')
# green_line = mlines.Line2D([], [], color='green', markersize=15, label='Green line')
# ax.legend(handles=[blue_line,green_line],bbox_to_anchor=(1.05, 0),  loc='lower left', borderaxespad=0.)
# aka = 0