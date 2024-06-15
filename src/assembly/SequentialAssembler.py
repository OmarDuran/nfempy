import time
from functools import partial
from assembly.ScatterFormData import (
    scatter_form_data,
    scatter_bc_form_data,
    scatter_time_dependent_form_data,
    scatter_time_dependent_bc_form_data,
    scatter_interface_form_data,
    scatter_bc_interface_form_data,
)


class SequentialAssembler:
    def __init__(self, space, jacobian, residual):
        self.space = space
        self.jacobian = jacobian
        self.residual = residual

    @property
    def form_type_to_callable_map(self):
        map = {
            "form": scatter_form_data,
            "bc_form": scatter_bc_form_data,
            "interface_form": scatter_interface_form_data,
            "bc_interface_form": scatter_bc_interface_form_data,
            "time_dependent_form": scatter_time_dependent_form_data,
            "time_dependent_bc_form": scatter_time_dependent_bc_form_data,
        }
        return map

    @property
    def form_to_input_list(self):
        if hasattr(self, "_form_to_input_list"):
            return self._form_to_input_list
        else:
            return None

    @form_to_input_list.setter
    def form_to_input_list(self, form_to_input_list):
        self._form_to_input_list = form_to_input_list

    def __time_dependent_scatter_form(self, input_list):
        form_type, sequence, form, alphas, time_value = input_list
        alpha_n, alpha = alphas
        scatter_callable = self.form_type_to_callable_map[form_type]
        scatter_function = partial(
            scatter_callable,
            weak_form=form,
            res_g=self.residual,
            jac_g=self.jacobian,
            alpha_n=alpha_n,
            alpha=alpha,
            t=time_value,
        )
        list(map(scatter_function, sequence))

    def __scatter_form(self, input_list):
        form_type, sequence, form, alpha = input_list
        scatter_callable = self.form_type_to_callable_map[form_type]
        scatter_function = partial(
            scatter_callable,
            weak_form=form,
            res_g=self.residual,
            jac_g=self.jacobian,
            alpha=alpha,
        )
        list(map(scatter_function, sequence))

    def scatter_forms(self, measure_time_q=False):
        for item in self.form_to_input_list.items():
            if measure_time_q:
                st = time.time()

            form_name, input_list = item
            form_type = input_list[0]
            is_time_dependent_q = "time_dependent" in form_type
            if is_time_dependent_q:
                self.__time_dependent_scatter_form(input_list)
            else:
                self.__scatter_form(input_list)

            if measure_time_q:
                et = time.time()
                elapsed_time = et - st
                print("SequentialAssembler:: Weak form: ", form_name)
                print("SequentialAssembler:: Scatter time:", elapsed_time, "seconds")
