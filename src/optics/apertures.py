import numpy as np

class Aperture:
    """ Apertures are limits in local (x, y) coordinates that can be used to filter rays
    A derived class should implement the contains function, which checks if a point is inside the aperture,
    and the mesh function, which generates a mesh of points that are inside the aperture"""
    def contains(self, points):
        """ Check if a point is inside the aperture
        Args:
          points: a N x 2 numpy array representing the point in 2D (surface coordinates) space
        Returns:
          a N x 1 numpy array of booleans representing if the points are inside the aperture"""
        raise NotImplementedError("contains() is not implemented")

    def mesh(self):
        """ Generate a mesh of points that are inside the aperture. Used to display the surfaces
        Returns:
          a N x 2 numpy array representing the points inside the aperture, generated from any parameter space.
          best is to use the numpy mgrid function for this."""
        raise NotImplementedError("mesh() is not implemented")

class CircularAperture(Aperture):
    """ A CircularAperture is an aperture that is a circle"""
    def __init__(self, radius):
        """ Create a CircularAperture object
        Args:
          radius: the radius of the aperture"""
        self.radius = radius

    def contains(self, points):
        """ Check if a point is inside the aperture
        Args:
          points: a N x 2 numpy array representing the point in 2D space
        Returns:
          a N x 1 numpy array of booleans representing if the points are inside the aperture"""
        return np.norm(points, axis=1) < self.radius

    def mesh(self):
        """ Generate a mesh of points that are inside the aperture. Used to display the surfaces.
        Returns:
          mesh_x: The x coordinates of the mesh points, as a 2D array
          mesh_y: The y coordinates of the mesh points, as a 2D array"""
        thetas = np.linspace(0, 2 * np.pi, 30)
        rs = np.linspace(0, self.radius, 15)
        thetagrid, rgrid = np.meshgrid(thetas, rs)
        mesh_x = rgrid * np.cos(thetagrid)
        mesh_y = rgrid * np.sin(thetagrid)
        return mesh_x, mesh_y

class RectangularAperture(Aperture):
    """ A RectanularAperture is an aperture that is a rectangle"""
    def __init__(self, size_x, size_y):
        """ Create a RectangularAperture object
        Args:
          size_x: the x extent of the aperture
          size_y: the y extent of the aperture"""
        self.size_x = size_x
        self.size_y = size_y

    def contains(self, points):
        """ Check if a point is inside the aperture
        Args:
          points: a N x 2 numpy array representing the point in 2D space
        Returns:
          a N x 1 numpy array of booleans representing if the points are inside the aperture"""
        return np.abs(points[:, 0]) < self.size_x / 2 and np.abs(points[:, 1]) < self.size_y / 2

    def mesh(self):
        """ Generate a mesh of points that are inside the aperture. Used to display the surfaces.
        Returns:
          mesh_x: The x coordinates of the mesh points, as a 2D array
          mesh_y: The y coordinates of the mesh points, as a 2D array"""
        xs = np.linspace(-self.size_x / 2, self.size_x / 2, 30)
        ys = np.linspace(-self.size_y / 2, self.size_y / 2, 30)
        mesh_x, mesh_y = np.meshgrid(xs, ys)
        return mesh_x, mesh_y
