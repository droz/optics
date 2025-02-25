import numpy as np

class Surface:
    """ Surfaces are objects that can propagate RayBundles. They are the basic building blocks of optical systems.
    A derived class should implement the sag function, which calculates the sag of the surface at a given point."""
    def __init__(self, origin, rotation, aperture, index1, index2):
        """ Create a Surface object
        Args:
          origin: 1 x 3 numpy array representing the origin of the surface in 3D space
          rotation: 3 x 3 numpy array representing the rotation matrix of the surface in 3D space
                    this matrix transforms local surface coordinates to global coordinates
          aperture: an Aperture object representing the aperture of the surface
          index1: the index of refraction of the medium the light is coming from
          index2: the index of refraction of the medium the light is going to. If 0, this is a mirror."""
        self.origin = origin
        self.rotation = rotation
        self.aperture = aperture
        self.index1 = index1
        self.index2 = index2

    def sag(self, points):
        """ Calculate the sag of the surface for an aray of 2D points
        Args:
          points: a N x 2 numpy array representing the point in 2D (surface coordinates) space
        Returns:
          a N x 1 numpy array representing the sag of the surface at these points"""
        raise NotImplementedError("sag() is not implemented by the base class")

    def normals(self, points):
        """ Calculate the normal of the surface at given points
        Args:
          points: a N x 2 numpy array representing the point in 2D (surface coordinates) space
        Returns:
          a N x 3 numpy array representing the normal (in surface coordinates) of the surface at these points"""
        # We are going to use the gradient of the sag function to calculate the normal.
        # For this we need to choose a good epsilon in x and y, not too small to avoid numerical errors, but not too large
        # to avoid sampling errors for a high frequency sag function. Choosing something close to the wavelength of light
        # is a good compromise.
        epsilon = 100e-9
        sag = self.sag(points)
        dsag_dx = (self.sag(points + np.array([epsilon, 0])) - sag)
        dsag_dy = (self.sag(points + np.array([0, epsilon])) - sag)
        # The two tangent vectors are (epsilon, 0, dsag_dx) and (0, epsilon, dsag_dy)
        # The normal is the cross product of these two vectors
        # The coordinates of the crossproduct end up being:
        # (- epsilon * dsag_dx, - epsilon * dsag_dy, epsilon^2)
        # but because we are going to normalize the vector anyway, we can just use:
        # (-dsag_dx, -dsag_dy, epsilon)
        norm = np.sqrt(dsag_dx**2 + dsag_dy**2 + epsilon**2)
        nx = -dsag_dx / norm
        ny = -dsag_dy / norm
        nz = epsilon / norm
        normals = np.array([nx, ny, nz]).T

        return normals

    def intersections(self, ray_bundle):
        """ Calculate the intersection of a RayBundle with the surface
        Args:
          ray_bundle: a RayBundle object
        Returns:
          a 2D numpy array representing the intersection of the rays with the surface in 2D (surface coordinates) space"""
        # First we transform the ray bundle 
        # We are going to use a Newton-Raphson method to find the intersection of the rays with the surface
        # We start by assuming that the intersection is at the origin of the surface
        origins = ray_bundle.origins
        directions = ray_bundle.directions
        origin = self.origin
        normal = self.normal


        # Calculate the intersection point of the rays with the surface
        t = np.sum((origin - origins) * normal, axis=1) / np.sum(directions * normal, axis=1)
        intersection_points = origins + t[:, np.newaxis] * directions
        return intersection_points


    def refract(self, ray_bundle):
        """ Propagate a RayBundle going through the surface
        Args:
          ray_bundle: a RayBundle object
        Returns:
          a RayBundle object"""
        
        raise NotImplementedError("propagate() is not implemented")

    def mesh(self):
        """ Generate a mesh representing the surface. Used to display the surfaces
        Returns:
          mesh_x: The x coordinates of the mesh points, as a 2D array
          mesh_y: The y coordinates of the mesh points, as a 2D array
          mesh_z: The z coordinates of the mesh points, as a 2D array"""
        mesh_x, mesh_y = self.aperture.mesh()
        xs = mesh_x.flatten()
        ys = mesh_y.flatten()
        points2d = np.array([xs, ys]).T
        zs = self.sag(points2d)
        mesh_z = np.reshape(zs, mesh_x.shape)
        return mesh_x, mesh_y, mesh_z

class SphericalSurface(Surface):
    """ A SphericalSurface is a Surface that has a spherical sag function"""
    def __init__(self, origin, rotation, aperture, index1, index2, radius):
        """ Create a SphericalSurface object
        Args:
          origin: 1 x 3 numpy array representing the origin of the surface in 3D space
          rotation: 3 x 3 numpy array representing the rotation matrix of the surface in 3D space
          aperture: an Aperture object representing the aperture of the surface
          index1: the index of refraction of the medium the light is coming from
          index2: the index of refraction of the medium the light is going to. If 0, this is a mirror.
          radius: the radius of the sphere"""
        super().__init__(origin, rotation, aperture, index1, index2)
        self.radius = radius

    def sag(self, points):
        """ Calculate the sag of the surface for an aray of 2D points
        Args:
          points: a N x 2 numpy array representing the point in 2D (surface coordinates) space
        Returns:
          a N x 1 numpy array representing the sag of the surface at these points"""
        r2 = points[:, 0]**2 + points[:, 1]**2
        if self.radius < 0:
            return self.radius + np.sqrt(self.radius**2 - r2)
        else:
            return self.radius - np.sqrt(self.radius**2 - r2)

