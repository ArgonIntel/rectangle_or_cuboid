import numpy as np
from shapes import Shape


def read_points_from_file():
    """Read points data from a file

    This function reads points data from a text file named "data.txt" and
    returns a NumPy array containing the points.

    Returns:
        npumpy.array: points from file converted into numpy.array for
        easy manipulation.
    """
    points_list = []
    try:
        with open("data_3d.txt", "r", encoding="utf-8") as file:
            points = file.readlines()
            for point in points:
                point = [float(x) for x in point.strip().split(",")]
                points_list.append(point)
            return np.array(points_list)
    except Exception as e:
        print(f"An error has occured while reading the file: {e}")


def main():
    points_matrix = read_points_from_file()

    if points_matrix.shape in [(4, 2), (5, 3)]:
        shape = Shape(points_matrix)
        if shape.check_orthogonality():
            shape_type = shape.shape_type()
            print(
                "Point X with coordinates "
                + str(shape.X_point)
                + f" is placed inside the {shape_type}: "
                + str(shape.x_in_shape())
            )
            print(
                f"The length of a diagonal of a {shape_type} is {shape.diagonal_length()}"
            )
            print(f"The shape is {shape_type}")
        else:
            print("Provided points are not vertices of a Rectangle or Cuboid")
    else:
        print("Points provided in a file are invalid")


if __name__ == "__main__":
    main()
