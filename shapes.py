import numpy as np


class Shape:
    """
    This class represents a shape (rectangle in 2D or cuboid in 3D) and provides
    methods to analyze its properties.
    """

    def __init__(self, array):
        self.m = array.shape[0] - 1
        self.vertices = array[0 : self.m]
        self.X_point = array[self.m]
        self.orth_vectors = np.zeros((array.shape[1], array.shape[1]))
        self.index = None

    def check_orthogonality(self) -> bool:
        """This method is performed on given points in an array self.vertices
        to check if vectors passing through those points are orthogonal and
        form a 2D square shape or 3D rectangle shape.

        Two vectors are orthogonal if theri dot product is equal to 0

        If vectors are orthogonal the method assigns those vectors into a varialbe
        self.orth_vectors needed for further operations.

        Futhermore, it also sets a variable self.index that is an index of a vertex
        which is common to all vectors of the base.

        Returns:
            bool: True if vectors are orthogonal, False otherwise
        """

        for i in range(self.m):
            vectors = [
                self.vertices[(i) % self.m] - self.vertices[(k + i + 1) % self.m]
                for k in range(self.m - 1)
            ]

            if all(
                np.dot(vectors[j % len(vectors)], vectors[(j + 1) % len(vectors)]) == 0
                for j in range(self.m - 1)
            ):
                self.index = i
                self.orth_vectors = np.array(vectors)
                return True
        else:
            return False

    def diagonal_length(self) -> float:
        """
        The method that calculates an L2 norm of all vectors to get a
        length of the diagonal line

        Returns:
            float: L2 norm - length of the diagonal line
        """
        return np.linalg.norm(np.sum(self.orth_vectors, axis=0, keepdims=True))

    def x_in_shape(self) -> bool:
        """
        If a point is a subset of an n-dimensional shape it must satisfy the following:
            it's orthogonal projection on each vector that are forming a shape must
            be >= 0 and <= length of each vector forming a shape in space.

        Returns:
            bool: True if X is a subset od the shape, False otherwise
        """
        dot_products = [
            (
                np.dot(
                    self.orth_vectors[i],
                    self.X_point - self.vertices[(self.index + i + 1) % self.m],
                ),
                np.dot(self.orth_vectors[i], self.orth_vectors[i]),
            )
            for i in range(self.m - 1)
        ]

        if all(
            dot_products[j][0] <= dot_products[j][1] and dot_products[j][0] >= 0
            for j in range(len(dot_products))
        ):
            return True
        else:
            return False

    def shape_type(self):
        if self.vertices.shape[1] == 2:
            return "Rectagle"
        elif self.vertices.shape[1] == 3:
            return "Cuboid"
