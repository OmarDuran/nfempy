from abc import ABC, abstractmethod

from spaces.product_space import ProductSpace


class WeakForm(ABC):
    def __init__(self, space: ProductSpace):
        self.space = space
        self._functions = None

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space):
        self._space = space

    @property
    def functions(self):
        return self._functions

    @functions.setter
    def functions(self, functions):
        self._functions = functions

    @abstractmethod
    def evaluate_form_at(self, element_index, alpha):
        pass
