
class Assembler:

    def __init__(self, space, jacobian, residual):
        self.space = space
        self.jacobian = jacobian
        self.residual = residual

    # @property
    # def generator_to_forms_map(self):
    #     if hasattr(self, "_generator_to_forms_map"):
    #         return self._generator_to_forms_map
    #     else:
    #         return None
    #
    # @generator_to_forms_map.setter
    # def generator_to_forms_map(self, generator_to_forms_map):
    #     self._generator_to_forms_map = generator_to_forms_map
    #
    # def scatter_forms(self):
    #     for item in self.generator_to_forms_map:
    #         generator, weakform

