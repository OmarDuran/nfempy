from abc import ABC
from pathlib import Path

import matplotlib

font = {"family": "normal", "weight": "bold", "size": 20}
matplotlib.rc("font", **font)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams["text.usetex"] = True


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
        map = {
            "sc_rt": "SC-RT",
            "sc_bdm": "SC-BDM",
            "wc_rt": "WC-RT",
            "wc_bdm": "WC-BDM",
        }
        return map

    @property
    def method_color_map(self):
        map = {
            "sc_rt": mcolors.TABLEAU_COLORS["tab:blue"],
            "sc_bdm": mcolors.TABLEAU_COLORS["tab:orange"],
            "wc_rt": mcolors.TABLEAU_COLORS["tab:green"],
            "wc_bdm": mcolors.TABLEAU_COLORS["tab:red"],
        }
        return map

    @property
    def mat_values_map(self):
        map = {
            "0.0001": "10^{-4}",
            "0.01": "10^{-2}",
            "1.0": "10^{0}",
            "100.0": "10^{2}",
            "10000.0": "10^{4}",
        }
        return map

    @property
    def markers_values_map(self):
        map = {"0.0001": "v", "0.01": "s", "1.0": "o", "100.0": "s", "10000.0": "v"}
        return map

    @property
    def style_values_map(self):
        map = {
            "0.0001": "dashdot",
            "0.01": "dashed",
            "1.0": "-",
            "100.0": "dashed",
            "10000.0": "dashdot",
        }
        return map

    @classmethod
    def create_directory(self):
        Path("figures").mkdir(parents=True, exist_ok=True)

    def save_figure(self):
        plt.savefig(Path("figures") / Path(self._name), format="pdf")

    @staticmethod
    def filter_composer(method, m_lambda, m_eps, k, d, l):
        filter_0 = method
        filter_1 = "_lambda_" + str(m_lambda)
        filter_2 = "_gamma_" + str(m_eps)
        filter_3 = "_k" + str(k)
        filter_4 = "_l" + str(l)
        filter_5 = "_" + str(d) + "d"
        filter = filter_0 + filter_1 + filter_2 + filter_3 + filter_4 + filter_5
        return filter

    @staticmethod
    def ref_levels():
        return [0, 1, 2, 3, 5]

    @staticmethod
    def convergence_type_key(method, conv_type):
        composed_key = method + "_" + conv_type
        return composed_key

    def available_ref_levels(self, file_names):
        available_ref_levels = []
        for l in painter_first_kind.ref_levels():
            result = [
                (idx, path.name)
                for idx, path in enumerate(file_names)
                if ("_l" + str(l) in path.name)
            ]
            if len(result) != 0:
                available_ref_levels.append(l)

        return available_ref_levels


class painter_first_kind(painter):
    @property
    def m_lambda(self):
        return 1.0

    @property
    def m_epsilon(self):
        return 1.0

    def color_canvas_with_variable_epsilon(self, k, d, methods, material_values):
        self.create_directory()

        file_names = list(Path().glob(self.file_pattern))
        mat_label = "\epsilon"
        fig, ax = plt.subplots(figsize=self.figure_size)

        # check for refinements available
        available_ref_levels = self.available_ref_levels(file_names)
        label_methods = {}
        label_parameters = {}
        for method in methods:
            for m_value in material_values:
                min_res_iterations = []
                dofs = []
                for l in available_ref_levels:
                    filter = painter_first_kind.filter_composer(
                        method=method,
                        m_lambda=self.m_lambda,
                        m_eps=m_value,
                        k=k,
                        d=d,
                        l=l,
                    )
                    result = [
                        (idx, path.name)
                        for idx, path in enumerate(file_names)
                        if (filter in path.name)
                    ]
                    assert len(result) == 1
                    label_method = self.method_map[method]
                    label_parameter = (
                        r"$"
                        + mat_label
                        + " = "
                        + self.mat_values_map[str(m_value)]
                        + "$"
                    )
                    line_style = self.style_values_map[str(m_value)]
                    color = self.method_color_map[method]
                    file_name = str(file_names[result[0][0]])
                    rdata = np.genfromtxt(file_name, dtype=None, delimiter=",")
                    dofs.append(rdata[0])
                    min_res_iterations.append(rdata.shape[0] - 1)

                # levels = np.array(painter_first_kind.ref_levels())
                dofs = np.array(dofs)
                min_res_iterations = np.array(min_res_iterations)
                plt.xscale("log")
                plt.plot(
                    dofs,
                    min_res_iterations,
                    marker="o",
                    linestyle=line_style,
                    color=color,
                )
                label_methods[label_method] = color
                label_parameters[label_parameter] = line_style

        legend_elements = []
        for chunk in label_methods.items():
            legend_elements.append(
                Line2D([0], [0], lw=2, label=chunk[0], color=chunk[1])
            )
        for chunk in label_parameters.items():
            legend_elements.append(
                Line2D(
                    [0], [0], lw=2, label=chunk[0], linestyle=chunk[1], color="black"
                )
            )

        ax.grid(
            True, linestyle="-.", axis="both", which="both", color="black", alpha=0.25
        )
        ax.tick_params(which="both", labelcolor="black", labelsize="large", width=2)

        plt.xlabel("Number of DoF")
        plt.ylabel("Preconditioned minimal residual iterations")
        plt.ylim(self.ordinate_range[0], self.ordinate_range[1])
        plt.legend(
            handles=legend_elements,
            ncol=2,
            handleheight=1,
            handlelength=4.0,
            labelspacing=0.05,
        )

    def color_canvas_with_variable_lambda(self, k, d, methods, material_values):
        self.create_directory()

        file_names = list(Path().glob(self.file_pattern))
        mat_label = "\lambda_{\sigma}"
        fig, ax = plt.subplots(figsize=self.figure_size)

        # check for refinements available
        available_ref_levels = self.available_ref_levels(file_names)
        label_methods = {}
        label_parameters = {}
        for method in methods:
            for m_value in material_values:
                min_res_iterations = []
                dofs = []
                for l in available_ref_levels:
                    filter = painter_first_kind.filter_composer(
                        method=method,
                        m_lambda=m_value,
                        m_eps=self.m_epsilon,
                        k=k,
                        d=d,
                        l=l,
                    )
                    result = [
                        (idx, path.name)
                        for idx, path in enumerate(file_names)
                        if (filter in path.name)
                    ]
                    assert len(result) == 1
                    label_method = self.method_map[method]
                    label_parameter = (
                        r"$"
                        + mat_label
                        + " = "
                        + self.mat_values_map[str(m_value)]
                        + "$"
                    )
                    line_style = self.style_values_map[str(m_value)]
                    color = self.method_color_map[method]
                    file_name = str(file_names[result[0][0]])
                    rdata = np.genfromtxt(file_name, dtype=None, delimiter=",")
                    dofs.append(rdata[0])
                    min_res_iterations.append(rdata.shape[0] - 1)

                dofs = np.array(dofs)
                min_res_iterations = np.array(min_res_iterations)
                plt.xscale("log")
                plt.plot(
                    dofs,
                    min_res_iterations,
                    marker="o",
                    linestyle=line_style,
                    color=color,
                )
                label_methods[label_method] = color
                label_parameters[label_parameter] = line_style

        legend_elements = []
        for chunk in label_methods.items():
            legend_elements.append(
                Line2D([0], [0], lw=2, label=chunk[0], color=chunk[1])
            )
        for chunk in label_parameters.items():
            legend_elements.append(
                Line2D(
                    [0], [0], lw=2, label=chunk[0], linestyle=chunk[1], color="black"
                )
            )

        ax.grid(
            True, linestyle="-.", axis="both", which="both", color="black", alpha=0.25
        )
        ax.tick_params(which="both", labelcolor="black", labelsize="large", width=2)

        plt.xlabel("Number of DoF")
        plt.ylabel("Preconditioned minimal residual iterations")
        plt.ylim(self.ordinate_range[0], self.ordinate_range[1])
        plt.legend(
            handles=legend_elements,
            ncol=2,
            handleheight=1,
            handlelength=4.0,
            labelspacing=0.05,
        )


class painter_second_kind(painter):
    @property
    def markers_values_map(self):
        map = {"0": "o", "1": "s"}
        return map

    @property
    def style_values_map(self):
        map = {
            "0": "-",
            "1": "dashed",
        }
        return map

    @staticmethod
    def filter_composer(method, k, d, l):
        filter_0 = method
        filter_1 = "_k" + str(k)
        filter_2 = "_l" + str(l)
        filter_3 = "_" + str(d) + "d"
        filter = filter_0 + filter_1 + filter_2 + filter_3
        return filter

    def color_canvas_with_variable_k(self, d, methods):
        self.create_directory()

        p = Path()
        file_names = list(p.glob(self.file_pattern))
        fig, ax = plt.subplots(figsize=self.figure_size)
        available_ref_levels = self.available_ref_levels(file_names)
        label_methods = {}
        label_parameters = {}
        for method in methods:
            for k in [0, 1]:
                min_res_iterations = []
                dofs = []
                for l in available_ref_levels:
                    filter = painter_second_kind.filter_composer(
                        method=method, k=k, d=d, l=l
                    )
                    result = [
                        (idx, path.name)
                        for idx, path in enumerate(file_names)
                        if (filter in path.name)
                    ]
                    assert len(result) == 1
                    color = self.method_color_map[method]
                    file_name = str(file_names[result[0][0]])
                    rdata = np.genfromtxt(file_name, dtype=None, delimiter=",")
                    dofs.append(rdata[0])
                    min_res_iterations.append(rdata.shape[0] - 1)

                line_style = self.style_values_map[str(k)]
                color = self.method_color_map[method]
                label = self.method_map[method] + ": " r"$ k = " + str(k) + "$"
                label_method = self.method_map[method]
                label_parameter = r"$ k = " + str(k) + "$"
                dofs = np.array(dofs)
                min_res_iterations = np.array(min_res_iterations)
                plt.xscale("log")
                plt.plot(
                    dofs,
                    min_res_iterations,
                    label=label,
                    marker="o",
                    linestyle=line_style,
                    color=color,
                )
                label_methods[label_method] = color
                label_parameters[label_parameter] = line_style

        legend_elements = []
        for chunk in label_methods.items():
            legend_elements.append(
                Line2D([0], [0], lw=2, label=chunk[0], color=chunk[1])
            )
        for chunk in label_parameters.items():
            legend_elements.append(
                Line2D(
                    [0], [0], lw=2, label=chunk[0], linestyle=chunk[1], color="black"
                )
            )

        ax.grid(
            True, linestyle="-.", axis="both", which="both", color="black", alpha=0.25
        )
        ax.tick_params(which="both", labelcolor="black", labelsize="large", width=2)

        plt.xlabel("Number of DoF")
        plt.ylabel("Preconditioned minimal residual iterations")
        plt.ylim(self.ordinate_range[0], self.ordinate_range[1])
        plt.legend(
            handles=legend_elements,
            ncol=2,
            handleheight=1,
            handlelength=4.0,
            labelspacing=0.05,
        )


def render_figures_example_1(d=2):
    methods = ["sc_rt", "sc_bdm", "wc_rt", "wc_bdm"]
    file_pattern = "output_example_1/*_res_history_ex_1.txt"

    painter_ex_1 = painter_first_kind()
    painter_ex_1.file_pattern = file_pattern

    material_values = [1.0, 0.01, 0.0001]
    painter_ex_1.ordinate_range = (0, 350)

    k = 0
    painter_ex_1.file_name = "min_res_iterations_k0_example_1_" + str(d) + "d.pdf"
    painter_ex_1.color_canvas_with_variable_epsilon(k, d, methods, material_values)
    painter_ex_1.save_figure()

    k = 1
    painter_ex_1.file_name = "min_res_iterations_k1_example_1_" + str(d) + "d.pdf"
    painter_ex_1.color_canvas_with_variable_epsilon(k, d, methods, material_values)
    painter_ex_1.save_figure()


def render_figures_example_2(d=2):
    methods = ["sc_rt", "sc_bdm", "wc_rt", "wc_bdm"]
    file_pattern = "output_example_2/*_res_history_ex_2.txt"

    painter_ex_2 = painter_first_kind()
    painter_ex_2.file_pattern = file_pattern

    material_values = [1.0, 100.0, 10000.0]
    painter_ex_2.ordinate_range = (0, 350)

    k = 0
    painter_ex_2.file_name = "min_res_iterations_k0_example_2_" + str(d) + "d.pdf"
    painter_ex_2.color_canvas_with_variable_lambda(k, d, methods, material_values)
    painter_ex_2.save_figure()

    k = 1
    painter_ex_2.file_name = "min_res_iterations_k1_example_2_" + str(d) + "d.pdf"
    painter_ex_2.color_canvas_with_variable_lambda(k, d, methods, material_values)
    painter_ex_2.save_figure()


def render_figures_example_3(d=2):
    methods = ["wc_rt", "wc_bdm"]
    file_pattern = "output_example_3/*_res_history_ex_3.txt"
    painter_ex_3 = painter_second_kind()
    painter_ex_3.file_pattern = file_pattern
    painter_ex_3.ordinate_range = (0, 350)
    painter_ex_3.file_name = "min_res_iterations_example_3_" + str(d) + "d.pdf"
    painter_ex_3.color_canvas_with_variable_k(d, methods)
    painter_ex_3.save_figure()


dim = 3
render_figures_example_1(d=dim)
render_figures_example_2(d=dim)
render_figures_example_3(d=dim)
