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
        # TODO: Remane _immersed_shapes to _embed_shapes
        # https://math.stackexchange.com/questions/68254/what-is-the-difference-between-immersion-and-embedding
        # self._embedded_shapes = np.array([], dtype=Shape)

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

    @dimension.setter
    def dimension(self, dimension):
        self._dimension

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

    def hash(self):
        return hash((self.dimension, self.tag))

    def __eq__(self, other):
        equality_by_dimension_and_tag = (self.dimension, self.tag) == (
            other.dimension,
            other.tag,
        )

        n_boundary_shapes_q = self.boundary_shapes.size == other.boundary_shapes.size
        n_immersed_shapes_q = self.immersed_shapes.size == other.immersed_shapes.size

        equality_by_boundary_shapes = True
        if self.boundary_shapes.size != 0 and n_boundary_shapes_q:
            equality_by_boundary_shapes = np.all(
                np.array(
                    [
                        shape == other_shape
                        for shape, other_shape in zip(
                            self.boundary_shapes, other.boundary_shapes
                        )
                    ]
                )
            )

        equality_by_immersed_shapes = True
        if self.immersed_shapes.size != 0 and n_immersed_shapes_q:
            equality_by_boundary_shapes = np.all(
                np.array(
                    [
                        shape == other_shape
                        for shape, other_shape in zip(
                            self.immersed_shapes, other.immersed_shapes
                        )
                    ]
                )
            )
        equality_q = (
            equality_by_dimension_and_tag
            and n_boundary_shapes_q
            and n_immersed_shapes_q
            and equality_by_boundary_shapes
            and equality_by_immersed_shapes
        )
        return equality_q

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, other):
        other_id = (other.dimension, other.tag)
        boundary_ids = [(shape.dimension, shape.tag) for shape in self.boundary_shapes]
        immersed_ids = [(shape.dimension, shape.tag) for shape in self.immersed_shapes]
        boundary_q = other_id in boundary_ids
        immersed_q = other_id in immersed_ids
        if boundary_q:
            boundary_q = (
                len([shape for shape in self.boundary_shapes if shape == other]) == 1
            )
        if immersed_q:
            immersed_q = (
                len([shape for shape in self.immersed_shapes if shape == other]) == 1
            )
        return boundary_q or immersed_q

    def shape_assignment(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Cannot assign shapes with differing dimensions.")
        if self.composite != other.composite:
            raise ValueError(
                "Cannot assign composite shapes to a non-composite shape, or vice versa."
            )

        self.active = other.active
        self.tag = other.tag
        self.physical_tag = other.physical_tag
        self.dimension = other.dimension
        self.composite = other.composite
        self.boundary_shapes = other.boundary_shapes
        self.immersed_shapes = other.immersed_shapes
