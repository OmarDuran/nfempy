from abc import ABC, abstractmethod
import numpy as np


class Shape(ABC):
    def __init__(self):
        self._active = True
        self._tag = None
        self._physical_tag = None
        self._dimension = None
        self._composite = False
        self._boundary_shapes = np.array([], dtype=Shape)
        self._immersed_shapes = np.array([], dtype=Shape)

    @staticmethod
    def shape_dimension_by_type(name):
        dimension_by_type = {
            "Vertex": 0,
            "Edge": 1,
            "Wire": 1,
            "Face": 2,
            "Shell": 2,
            "Solid": 3,
            "CompositeSolid": 3,
        }
        return dimension_by_type[name]

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, tag):
        self._active = tag

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, tag):
        self._tag = tag

    @property
    def physical_tag(self):
        return self._physical_tag

    @physical_tag.setter
    def physical_tag(self, physical_tag):
        self._physical_tag = physical_tag

    @property
    def dimension(self):
        return self._dimension

    @property
    def composite(self):
        return self._composite

    @composite.setter
    def composite(self, composite):
        self._composite = composite

    @property
    def boundary_shapes(self):
        return self._boundary_shapes

    @boundary_shapes.setter
    def boundary_shapes(self, shapes):
        # make shape unique
        perm = np.argsort([shape.tag for shape in shapes])
        shapes = shapes[perm]

        check_shapes = np.array(
            [shape.dimension in self.admissible_dimensions() for shape in shapes]
        )
        if np.all(check_shapes):
            self._boundary_shapes = shapes
        else:
            raise ValueError(
                "This shape can only contain these dimensions: ",
                self.admissible_dimensions(),
            )

    @property
    def immersed_shapes(self):
        return self._immersed_shapes

    @immersed_shapes.setter
    def immersed_shapes(self, shapes):
        check_shapes = np.array(
            [shape.dimension in self.admissible_dimensions() for shape in shapes]
        )
        if np.all(check_shapes):
            self._immersed_shapes = shapes
        else:
            raise ValueError(
                "This shape can only contain these dimensions: ",
                self.admissible_dimensions(),
            )

    @abstractmethod
    def admissible_dimensions(self):
        pass

    def __eq__(self, other):
        return (self.dimension, self.tag) == (other.dimension, other.tag)

    def __contains__(self, other):
        other_id = (other.dimension, other.tag)
        boundary_ids = [(shape.dimension, shape.tag) for shape in self.boundary_shapes]
        immersed_ids = [(shape.dimension, shape.tag) for shape in self.immersed_shapes]
        return (other_id in boundary_ids) or (other_id in immersed_ids)
