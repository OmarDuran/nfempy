import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.markers as plt_markers
plt.rcParams["text.usetex"] =True
import numpy as np
from pathlib import Path
from abc import ABC


class painter(ABC):

    @property
    def figure_size(self):
        return (8, 8)

    @property
    def file_pattern(self):
        return self._pattern

    @file_pattern.setter
    def file_pattern(self, pattern):
        self._pattern = pattern

    @property
    def file_name(self):
        return self._name

    @file_name.setter
    def file_name(self, name):
        self._name = name

    @property
    def ordinate_range(self):
        return self._v_range

    @ordinate_range.setter
    def ordinate_range(self, range):
        self._v_range = range

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

    @classmethod
    def create_directory(self):
        Path("figures").mkdir(parents=True, exist_ok=True)

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

    # @classmethod
    # def update_file_name(self):
    #     self._name = Path('figures') / Path(self._name)

    @property
    def m_lambda(self):
        return 1.0

    @property
    def m_epsilon(self):
        return 1.0

    def color_canvas_with_variable_epsilon(self, k, d, methods, material_values, conv_type):

        self.create_directory()

        p = Path()
        file_names = list(p.glob(self.file_pattern))
        mat_label = '\epsilon'
        fig, ax = plt.subplots(figsize=self.figure_size)

        for method in methods:
            if conv_type == 'super' and method in ['wc_rt', 'wc_bdm']:
                continue

            for m_value in material_values:

                filter = painter_first_kind.filter_composer(method=method, m_lambda=self.m_lambda, m_eps=m_value,
                                         k=k, d=d)
                result = [(idx, path.name) for idx, path in enumerate(file_names) if
                          (filter in path.name)]
                assert len(result) == 1
                label = self.method_map[method] + ': ' + r'$' + mat_label + ' = ' + \
                        self.mat_values_map[str(m_value)] + '$'
                marker = self.markers_values_map[str(m_value)]
                color = self.method_color_map[method]
                file_name = str(file_names[result[0][0]])
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
        plt.text(0.25, 1.0, r"$\mathbf{1}$")
        plt.legend()

        plt.savefig(Path('figures') / Path(self._name), format='pdf')

    def color_canvas_with_variable_lambda(self, k, d, methods, material_values, conv_type):

        self.create_directory()

        p = Path()
        file_names = list(p.glob(self.file_pattern))
        mat_label = '\lambda_{\sigma}'
        fig, ax = plt.subplots(figsize=self.figure_size)

        for method in methods:
            if conv_type == 'super' and method in ['wc_rt', 'wc_bdm']:
                continue

            for m_value in material_values:

                filter = painter_first_kind.filter_composer(method=method, m_lambda=m_value, m_eps=self.m_epsilon,
                                         k=k, d=d)
                result = [(idx, path.name) for idx, path in enumerate(file_names) if
                          (filter in path.name)]
                assert len(result) == 1
                label = self.method_map[method] + ': ' + r'$' + mat_label + ' = ' + \
                        self.mat_values_map[str(m_value)] + '$'
                marker = self.markers_values_map[str(m_value)]
                color = self.method_color_map[method]
                file_name = str(file_names[result[0][0]])
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
        plt.text(0.25, 1.0, r"$\mathbf{1}$")
        plt.legend()

        plt.savefig(Path('figures') / Path(self._name), format='pdf')

class painter_second_kind(painter):

    @property
    def markers_values_map(self):
        map = {'0': "o", '1': "s"}
        return map

    # @property
    # def method_color_map(self):
    #     map = {'0': mcolors.TABLEAU_COLORS['tab:blue'],  '1': mcolors.TABLEAU_COLORS['tab:orange']}
    #     return map

    @staticmethod
    def filter_composer(method, k, d):
        filter_0 = method
        filter_1 = '_k' + str(k)
        filter_2 = '_' + str(d) + 'd'
        filter = filter_0 + filter_1 + filter_2
        return  filter

    def color_canvas_with_variable_k(self, d, methods):

        self.create_directory()

        p = Path()
        file_names = list(p.glob(self.file_pattern))
        fig, ax = plt.subplots(figsize=self.figure_size)
        idxs = self.convergence_type_map['normal']

        for method in methods:

            for k in [0,1]:

                filter = painter_second_kind.filter_composer(method=method, k=k, d=d)
                result = [(idx, path.name) for idx, path in enumerate(file_names) if
                          (filter in path.name)]
                assert len(result) == 1
                label = self.method_map[method] + ': ' r'$ k = ' + str(k) + '$'
                marker = self.markers_values_map[str(k)]
                color = self.method_color_map[method]
                file_name = str(file_names[result[0][0]])
                rdata = np.genfromtxt(file_name, dtype=None, delimiter=',', skip_header=1)

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
        plt.text(0.25, 1.0, r"$\mathbf{1}$")
        plt.legend()
        
        plt.savefig(Path('figures') / Path(self._name), format='pdf')

def render_figures_example_1(d = 2):

    methods = ['sc_rt', 'sc_bdm', 'wc_rt', 'wc_bdm']
    file_pattern = 'output_example_1/*_error_ex_1.txt'

    painter_ex_1 = painter_first_kind()
    painter_ex_1.file_pattern = file_pattern

    material_values = [1.0, 0.01, 0.0001]
    painter_ex_1.ordinate_range = (0.001, 100)
    conv_type = 'normal'

    k = 0
    painter_ex_1.file_name = 'normal_convergence_var_eps_k0_ex_1.pdf'
    painter_ex_1.color_canvas_with_variable_epsilon(k, d, methods, material_values, conv_type)

    k = 1
    painter_ex_1.file_name = 'normal_convergence_var_eps_k1_ex_1.pdf'
    painter_ex_1.color_canvas_with_variable_epsilon(k, d, methods, material_values, conv_type)

    painter_ex_1.ordinate_range = (0.00001, 100)
    conv_type = 'super'

    k = 0
    painter_ex_1.file_name = 'super_convergence_var_eps_k0_ex_1.pdf'
    painter_ex_1.color_canvas_with_variable_epsilon(k, d, methods, material_values,
                                                    conv_type)

    k = 1
    painter_ex_1.file_name = 'super_convergence_var_eps_k1_ex_1.pdf'
    painter_ex_1.color_canvas_with_variable_epsilon(k, d, methods, material_values,
                                                    conv_type)


def render_figures_example_2(d = 2):
    methods = ['sc_rt', 'sc_bdm', 'wc_rt', 'wc_bdm']
    file_pattern = 'output_example_2/*_error_ex_2.txt'

    painter_ex_2 = painter_first_kind()
    painter_ex_2.file_pattern = file_pattern

    material_values = [1.0, 100.0, 10000.0]
    painter_ex_2.ordinate_range = (0.001, 100)
    conv_type = 'normal'

    k = 0
    painter_ex_2.file_name = 'normal_convergence_var_eps_k0_ex_2.pdf'
    painter_ex_2.color_canvas_with_variable_lambda(k, d, methods, material_values, conv_type)

    k = 1
    painter_ex_2.file_name = 'normal_convergence_var_eps_k1_ex_2.pdf'
    painter_ex_2.color_canvas_with_variable_lambda(k, d, methods, material_values, conv_type)

    painter_ex_2.ordinate_range = (0.00001, 100)
    conv_type = 'super'

    k = 0
    painter_ex_2.file_name = 'super_convergence_var_eps_k0_ex_2.pdf'
    painter_ex_2.color_canvas_with_variable_lambda(k, d, methods, material_values,
                                                    conv_type)

    k = 1
    painter_ex_2.file_name = 'super_convergence_var_eps_k1_ex_2.pdf'
    painter_ex_2.color_canvas_with_variable_lambda(k, d, methods, material_values,
                                                    conv_type)

def render_figures_example_3(d = 2):
    methods = ['wc_rt', 'wc_bdm']
    file_pattern = 'output_example_3/*_error_ex_3.txt'
    painter_ex_3 = painter_second_kind()
    painter_ex_3.file_pattern = file_pattern
    painter_ex_3.ordinate_range = (0.0001, 1)
    painter_ex_3.file_name = 'normal_convergence_var_k_ex_3.pdf'
    painter_ex_3.color_canvas_with_variable_k(d, methods)

render_figures_example_1()
render_figures_example_2()
render_figures_example_3()
