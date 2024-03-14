from abc import ABC
from pathlib import Path

import matplotlib
import numpy as np

font = {"family": "normal", "weight": "bold", "size": 15}
matplotlib.rc("font", **font)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["text.usetex"] = True


class ConvergenceTriangle:
    def __init__(self, data, rate, h_shift, e_shift, mirror_q):
        if not self.validate_input(data, rate, h_shift, e_shift, mirror_q):
            raise ValueError("Not valid input(s).")
        self._data = np.log(data)
        self._rate = rate
        self._h_shift = h_shift
        self._e_shift = e_shift
        self._mirror_q = mirror_q
        self._triangle = None
        self._label_pos = None
        self._build_me()

    def validate_input(self, data, rate, h_shift, e_shift, mirror_q):
        data_ok = isinstance(data, np.ndarray)
        rate_ok = isinstance(rate, int)
        shifts_ok = isinstance(h_shift, float) and isinstance(e_shift, float)
        mirror_q_ok = isinstance(mirror_q, bool)

        if not data_ok:
            raise TypeError("Expected type for rate: ", type(rate))
        else:
            data_ok = data.shape[0] > 1 and data.shape[1] > 1
            if not data_ok:
                raise ValueError(
                    "Expected at least two points, data.shape is: ", data.shape
                )

        if not rate_ok:
            raise TypeError("Expected type for rate: ", type(rate))

        if not shifts_ok:
            raise TypeError("Expected type for h_shift or e_shift: ", type(h_shift))

        if not mirror_q_ok:
            raise TypeError("Expected type for mirror_q: ", type(mirror_q))

        valid_input_q = data_ok and rate_ok and shifts_ok and mirror_q_ok

        return valid_input_q

    def _build_me(self):
        p0, p2 = self._data[-2], self._data[-1]
        step = p0[0] - p2[0]
        if self._mirror_q:
            p1 = p0 - np.array([step, 0.0])
            p2 = p0 - np.array([step, self._rate * step])
        else:
            p1 = p0 - np.array([0.0, self._rate * step])
            p2 = p0 - np.array([step, self._rate * step])
        p0[0] += self._h_shift
        p1[0] += self._h_shift
        p2[0] += self._h_shift
        p0[1] += self._e_shift
        p1[1] += self._e_shift
        p2[1] += self._e_shift
        self._triangle = np.exp(np.vstack((p0, p1, p2, p0)))

        xc = np.mean(np.vstack((p0, p1, p2)), axis=0)
        if self._mirror_q:
            dirh = xc - np.mean(np.vstack((p2, p1)), axis=0)
            dire = xc - np.mean(np.vstack((p0, p1)), axis=0)
            dirh[1] = 0.0
            dire[0] = 0.0
            dirh = dirh / np.linalg.norm(dirh)
            dire = dire / np.linalg.norm(dire)
            step_pos = np.exp(-0.35 * dire + np.mean(np.vstack((p0, p1)), axis=0))
            rate_pos = np.exp(-0.04 * dirh + np.mean(np.vstack((p2, p1)), axis=0))
            self._label_pos = (step_pos, rate_pos)
        else:
            dirh = xc - np.mean(np.vstack((p0, p1)), axis=0)
            dire = xc - np.mean(np.vstack((p2, p1)), axis=0)
            dirh[1] = 0.0
            dire[0] = 0.0
            dirh = dirh / np.linalg.norm(dirh)
            dire = dire / np.linalg.norm(dire)
            step_pos = np.exp(-0.35 * dire + np.mean(np.vstack((p2, p1)), axis=0))
            rate_pos = np.exp(-0.04 * dirh + np.mean(np.vstack((p0, p1)), axis=0))
            self._label_pos = (step_pos, rate_pos)

    def inset_me(self):
        plt.fill(
            self._triangle[:, 0],
            self._triangle[:, 1],
            linestyle="-",
            facecolor="lightgray",
            edgecolor="gray",
            linewidth=1,
        )
        plt.text(
            self._label_pos[0][0],
            self._label_pos[0][1],
            r"$\mathbf{1}$",
            color="gray",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.text(
            self._label_pos[1][0],
            self._label_pos[1][1],
            r"$\mathbf{" + str(self._rate) + "}$",
            color="gray",
            horizontalalignment="center",
            verticalalignment="center",
        )


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

    @property
    def convergence_type_map(self):
        map = {
            "sc_rt_normal": np.array([3, 4, 9, 10]),
            "sc_rt_super": np.array([4, 6, 11, 12]),
            "sc_bdm_normal": np.array([3, 4, 9, 10]),
            "sc_bdm_super": np.array([4, 6, 11, 12]),
            "wc_rt_normal": np.array([3, 4, 9, 10]),
            "wc_rt_super": np.array([4, 6, 11, 12]),
            "wc_bdm_normal": np.array([3, 4, 9, 10]),
            "wc_bdm_super": np.array([4, 6, 11, 12]),
        }
        return map

    @classmethod
    def create_directory(self):
        Path("figures").mkdir(parents=True, exist_ok=True)

    def save_figure(self):
        plt.savefig(Path("figures") / Path(self._name), format="pdf")

    @staticmethod
    def filter_composer(method, m_lambda, m_eps, k, d):
        filter_0 = method
        filter_1 = "_lambda_" + str(m_lambda)
        filter_2 = "_gamma_" + str(m_eps)
        filter_3 = "_k" + str(k)
        filter_4 = "_" + str(d) + "d"
        filter = filter_0 + filter_1 + filter_2 + filter_3 + filter_4
        return filter

    @staticmethod
    def convergence_type_key(method, conv_type):
        composed_key = method + "_" + conv_type
        return composed_key


class painter_ex_1(painter):
    @property
    def m_lambda(self):
        return 1.0

    @property
    def m_epsilon(self):
        return 1.0

    @property
    def convergence_type_map(self):
        map = {
            "sc_rt_normal": np.array([3, 4, 9, 10]),
            "sc_rt_super": np.array([4, 6, 11]),
            "sc_bdm_normal": np.array([3, 4, 9, 10]),
            "sc_bdm_super": np.array([4, 5, 10, 11]),
            "wc_rt_normal": np.array([3, 4, 9, 10]),
            "wc_rt_super": np.array([11]),
            "wc_bdm_normal": np.array([3, 4, 9, 10]),
            "wc_bdm_super": np.array([11]),
        }
        return map

    def color_canvas_with_variable_epsilon(
        self, k, d, methods, material_values, conv_type
    ):
        self.create_directory()

        file_names = list(Path().glob(self.file_pattern))
        mat_label = "\epsilon"
        fig, ax = plt.subplots(figsize=self.figure_size)
        label_methods = {}
        label_parameters = {}
        for method in methods:
            for m_value in material_values:
                filter = painter_ex_1.filter_composer(
                    method=method, m_lambda=self.m_lambda, m_eps=m_value, k=k, d=d
                )
                result = [
                    (idx, path.name)
                    for idx, path in enumerate(file_names)
                    if (filter in path.name)
                ]
                assert len(result) == 1
                label_method = self.method_map[method]
                label_parameter = (
                    r"$" + mat_label + " = " + self.mat_values_map[str(m_value)] + "$"
                )
                line_style = self.style_values_map[str(m_value)]
                color = self.method_color_map[method]
                file_name = str(file_names[result[0][0]])
                rdata = np.genfromtxt(
                    file_name, dtype=None, delimiter=",", skip_header=1
                )
                conv_type_key = painter_ex_1.convergence_type_key(method, conv_type)
                idxs = self.convergence_type_map[conv_type_key]

                h = rdata[:, np.array([2])]
                plt.xlim(np.min(h) / 1.1, np.max(h) * 1.1)

                error = np.sum(rdata[:, idxs], axis=1)
                plt.loglog(h, error, linestyle=line_style, color=color)
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
        plt.xlabel(r"$h$")
        plt.ylabel("Error")
        plt.ylim(self.ordinate_range[0], self.ordinate_range[1])
        plt.legend(
            handles=legend_elements,
            ncol=2,
            handleheight=1,
            handlelength=4.0,
            labelspacing=0.05,
        )

    def build_inset_var_epsilon(
        self, k, d, method, m_value, conv_type, rate, h_shift, e_shift, mirror_q=False
    ):
        file_names = list(Path().glob(self.file_pattern))
        filter = painter_ex_1.filter_composer(
            method=method, m_lambda=self.m_lambda, m_eps=m_value, k=k, d=d
        )
        result = [
            (idx, path.name)
            for idx, path in enumerate(file_names)
            if (filter in path.name)
        ]
        assert len(result) == 1
        file_name = str(file_names[result[0][0]])
        rdata = np.genfromtxt(file_name, dtype=None, delimiter=",", skip_header=1)
        conv_type_key = painter_ex_1.convergence_type_key(method, conv_type)
        idxs = self.convergence_type_map[conv_type_key]
        ldata = np.vstack((rdata[:, 2], np.sum(rdata[:, idxs], axis=1))).T
        conv_triangle = ConvergenceTriangle(ldata, rate, h_shift, e_shift, mirror_q)
        conv_triangle.inset_me()


class painter_ex_2(painter):
    @property
    def m_lambda(self):
        return 1.0

    @property
    def m_epsilon(self):
        return 1.0

    @property
    def convergence_type_map(self):
        map = {
            "sc_rt_normal": np.array([3, 4, 9, 10]),
            "sc_rt_super": np.array([4, 6, 11]),
            "sc_bdm_normal": np.array([3, 4, 9, 10]),
            "sc_bdm_super": np.array([4, 5, 10, 11]),
            "wc_rt_normal": np.array([3, 4, 9, 10]),
            "wc_rt_super": np.array([5, 10, 11, 12]),
            "wc_bdm_normal": np.array([3, 4, 9, 10]),
            "wc_bdm_super": np.array([5, 6, 11, 12]),
        }
        return map

    def color_canvas_with_variable_lambda(
        self, k, d, methods, material_values, conv_type
    ):
        self.create_directory()

        file_names = list(Path().glob(self.file_pattern))
        mat_label = "\lambda_{\sigma}"
        fig, ax = plt.subplots(figsize=self.figure_size)
        label_methods = {}
        label_parameters = {}
        for method in methods:
            for m_value in material_values:
                filter = painter_ex_2.filter_composer(
                    method=method, m_lambda=m_value, m_eps=self.m_epsilon, k=k, d=d
                )
                result = [
                    (idx, path.name)
                    for idx, path in enumerate(file_names)
                    if (filter in path.name)
                ]
                assert len(result) == 1
                label_method = self.method_map[method]
                label_parameter = (
                    r"$" + mat_label + " = " + self.mat_values_map[str(m_value)] + "$"
                )
                line_style = self.style_values_map[str(m_value)]
                color = self.method_color_map[method]
                file_name = str(file_names[result[0][0]])
                rdata = np.genfromtxt(
                    file_name, dtype=None, delimiter=",", skip_header=1
                )
                conv_type_key = painter_ex_2.convergence_type_key(method, conv_type)
                idxs = self.convergence_type_map[conv_type_key]

                h = rdata[:, np.array([2])]
                plt.xlim(np.min(h) / 1.1, np.max(h) * 1.1)

                if conv_type_key == "wc_rt_" + "super":  # scaling in equation (5.8)
                    rdata[:, idxs[1]] = np.sqrt(h[:, 0]) * rdata[:, idxs[1]]

                error = np.sum(rdata[:, idxs], axis=1)
                plt.loglog(h, error, linestyle=line_style, color=color)
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
        plt.xlabel(r"$h$")
        plt.ylabel("Error")
        plt.ylim(self.ordinate_range[0], self.ordinate_range[1])
        plt.legend(
            handles=legend_elements,
            ncol=2,
            handleheight=1,
            handlelength=4.0,
            labelspacing=0.05,
        )

    def build_inset_var_lambda(
        self, k, d, method, m_value, conv_type, rate, h_shift, e_shift, mirror_q=False
    ):
        file_names = list(Path().glob(self.file_pattern))
        filter = painter_ex_2.filter_composer(
            method=method, m_lambda=m_value, m_eps=self.m_epsilon, k=k, d=d
        )
        result = [
            (idx, path.name)
            for idx, path in enumerate(file_names)
            if (filter in path.name)
        ]
        assert len(result) == 1
        file_name = str(file_names[result[0][0]])
        rdata = np.genfromtxt(file_name, dtype=None, delimiter=",", skip_header=1)
        conv_type_key = painter_ex_2.convergence_type_key(method, conv_type)
        idxs = self.convergence_type_map[conv_type_key]
        ldata = np.vstack((rdata[:, 2], np.sum(rdata[:, idxs], axis=1))).T
        conv_triangle = ConvergenceTriangle(ldata, rate, h_shift, e_shift, mirror_q)
        conv_triangle.inset_me()


class painter_ex_3(painter):
    @property
    def markers_values_map(self):
        map = {"0": "o", "1": "s"}
        return map

    @staticmethod
    def filter_composer(method, k, d):
        filter_0 = method
        filter_1 = "_k" + str(k)
        filter_2 = "_" + str(d) + "d"
        filter = filter_0 + filter_1 + filter_2
        return filter

    @property
    def convergence_type_map(self):
        map = {
            "sc_rt_normal": np.array([3, 4, 9, 10]),
            "sc_rt_super": np.array([4, 6, 11]),
            "sc_bdm_normal": np.array([3, 4, 9, 10]),
            "sc_bdm_super": np.array([4, 5, 10, 11]),
            "wc_rt_normal": np.array([3, 4, 9, 10]),
            "wc_rt_super": np.array([11]),
            "wc_bdm_normal": np.array([3, 4, 9, 10]),
            "wc_bdm_super": np.array([11]),
        }
        return map

    def color_canvas_with_variable_k(self, d, methods, conv_type):
        self.create_directory()

        p = Path()
        file_names = list(p.glob(self.file_pattern))
        fig, ax = plt.subplots(figsize=self.figure_size)

        for method in methods:
            for k in [0, 1]:
                filter = painter_ex_3.filter_composer(method=method, k=k, d=d)
                result = [
                    (idx, path.name)
                    for idx, path in enumerate(file_names)
                    if (filter in path.name)
                ]
                assert len(result) == 1
                conv_type_key = painter_ex_3.convergence_type_key(method, conv_type)
                idxs = self.convergence_type_map[conv_type_key]
                label = self.method_map[method] + ": " r"$ k = " + str(k) + "$"
                marker = self.markers_values_map[str(k)]
                color = self.method_color_map[method]
                file_name = str(file_names[result[0][0]])
                rdata = np.genfromtxt(
                    file_name, dtype=None, delimiter=",", skip_header=1
                )

                h = rdata[:, np.array([2])]
                plt.xlim(np.min(h) / 1.1, np.max(h) * 1.1)

                error = np.sum(rdata[:, idxs], axis=1)
                plt.loglog(h, error, label=label, marker=marker, color=color)

        ax.grid(
            True, linestyle="-.", axis="both", which="both", color="black", alpha=0.25
        )
        ax.tick_params(which="both", labelcolor="black", labelsize="large", width=2)
        plt.xlabel(r"$h$")
        plt.ylabel("Error")
        plt.ylim(self.ordinate_range[0], self.ordinate_range[1])
        plt.legend()

    def build_inset_var_k_order(
        self, k, d, method, conv_type, rate, h_shift, e_shift, mirror_q=False
    ):
        file_names = list(Path().glob(self.file_pattern))
        filter = painter_ex_3.filter_composer(method=method, k=k, d=d)
        result = [
            (idx, path.name)
            for idx, path in enumerate(file_names)
            if (filter in path.name)
        ]
        assert len(result) == 1
        file_name = str(file_names[result[0][0]])
        rdata = np.genfromtxt(file_name, dtype=None, delimiter=",", skip_header=1)
        conv_type_key = painter_ex_3.convergence_type_key(method, conv_type)
        idxs = self.convergence_type_map[conv_type_key]
        ldata = np.vstack((rdata[:, 2], np.sum(rdata[:, idxs], axis=1))).T
        conv_triangle = ConvergenceTriangle(ldata, rate, h_shift, e_shift, mirror_q)
        conv_triangle.inset_me()


def render_figures_example_1(d=2):
    methods = ["sc_rt", "sc_bdm", "wc_rt", "wc_bdm"]
    file_pattern = "output_example_1/*_error_ex_1.txt"

    painter = painter_ex_1()
    painter.file_pattern = file_pattern

    material_values = [1.0, 0.01, 0.0001]
    painter.ordinate_range = (0.005, 50)
    conv_type = "normal"

    k = 0
    rate = k + 1
    painter.file_name = "convergence_k0_example_1_" + str(d) + "d.pdf"
    painter.color_canvas_with_variable_epsilon(
        k, d, methods, material_values, conv_type
    )
    painter.build_inset_var_epsilon(
        k, d, methods[3], material_values[2], conv_type, rate, 0.0, -0.2
    )
    painter.save_figure()

    k = 1
    rate = k + 1
    painter.file_name = "convergence_k1_example_1_" + str(d) + "d.pdf"
    painter.color_canvas_with_variable_epsilon(
        k, d, methods, material_values, conv_type
    )
    painter.build_inset_var_epsilon(
        k, d, methods[3], material_values[2], conv_type, rate, 0.0, -0.2
    )
    painter.save_figure()

    painter.ordinate_range = (0.000001, 50)
    conv_type = "super"

    k = 0
    rate = k + 2
    painter.file_name = "superconvergence_k0_example_1_" + str(d) + "d.pdf"
    painter.color_canvas_with_variable_epsilon(
        k, d, methods, material_values, conv_type
    )
    painter.build_inset_var_epsilon(
        k, d, methods[3], material_values[1], conv_type, rate, 0.0, -0.5
    )
    painter.save_figure()

    k = 1
    rate = k + 2
    painter.file_name = "superconvergence_k1_example_1_" + str(d) + "d.pdf"
    painter.color_canvas_with_variable_epsilon(
        k, d, methods, material_values, conv_type
    )
    painter.build_inset_var_epsilon(
        k, d, methods[3], material_values[1], conv_type, rate, 0.0, -0.4
    )
    painter.save_figure()


def render_figures_example_2(d=2):
    methods = ["sc_rt", "sc_bdm", "wc_rt", "wc_bdm"]
    file_pattern = "output_example_2/*_error_ex_2.txt"

    painter = painter_ex_2()
    painter.file_pattern = file_pattern

    material_values = [1.0, 100.0, 10000.0]
    painter.ordinate_range = (1.0, 300.0)
    conv_type = "normal"

    k = 0
    painter.file_name = "convergence_k0_example_2_" + str(d) + "d.pdf"
    painter.color_canvas_with_variable_lambda(k, d, methods, material_values, conv_type)
    rate = k + 1
    painter.build_inset_var_lambda(
        k, d, methods[0], material_values[0], conv_type, rate, 0.0, -0.2
    )
    painter.save_figure()

    k = 1
    painter.file_name = "convergence_k1_example_2_" + str(d) + "d.pdf"
    painter.color_canvas_with_variable_lambda(k, d, methods, material_values, conv_type)
    rate = k + 1
    painter.build_inset_var_lambda(
        k, d, methods[0], material_values[0], conv_type, rate, 0.0, -0.2
    )
    painter.save_figure()

    painter.ordinate_range = (0.00005, 50)
    conv_type = "super"

    k = 0
    painter.file_name = "superconvergence_k0_example_2_" + str(d) + "d.pdf"
    painter.color_canvas_with_variable_lambda(k, d, methods, material_values, conv_type)
    rate = k + 2
    painter.build_inset_var_lambda(
        k, d, methods[0], material_values[0], conv_type, rate, 0.0, -0.2
    )
    painter.save_figure()

    k = 1
    painter.file_name = "superconvergence_k1_example_2_" + str(d) + "d.pdf"
    painter.color_canvas_with_variable_lambda(k, d, methods, material_values, conv_type)
    rate = k + 2
    painter.build_inset_var_lambda(
        k, d, methods[0], material_values[0], conv_type, rate, 0.0, -0.2
    )
    painter.save_figure()


def render_figures_example_3(d=2):
    methods = ["wc_rt", "wc_bdm"]
    file_pattern = "output_example_3/*_error_ex_3.txt"
    painter = painter_ex_3()
    painter.file_pattern = file_pattern
    painter.ordinate_range = (0.001, 20)
    conv_type = "normal"
    painter.file_name = "convergence_example_3_" + str(d) + "d.pdf"
    painter.color_canvas_with_variable_k(d, methods, conv_type)
    k = 0
    rate = k + 1
    painter.build_inset_var_k_order(k, d, methods[1], conv_type, rate, 0.0, -0.2)
    k = 1
    rate = k + 1
    painter.build_inset_var_k_order(k, d, methods[1], conv_type, rate, 0.0, -0.2)
    painter.save_figure()

    painter.ordinate_range = (0.000001, 0.01)
    conv_type = "super"
    painter.file_name = "superconvergence_example_3_" + str(d) + "d.pdf"
    painter.color_canvas_with_variable_k(d, methods, conv_type)
    k = 0
    rate = k + 2
    painter.build_inset_var_k_order(k, d, methods[1], conv_type, rate, 0.0, -0.2)
    k = 1
    rate = k + 2
    painter.build_inset_var_k_order(k, d, methods[1], conv_type, rate, 0.0, -0.2)
    painter.save_figure()


# Only figure range is adjusted for 3d.
dim = 3
render_figures_example_1(d=dim)
render_figures_example_2(d=dim)
render_figures_example_3(d=dim)
