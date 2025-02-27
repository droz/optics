from .apertures import Aperture
from .rays import RayBundle
import numpy as np

class Screen:
    """ A Screen is an object that captures rays for analysis"""
    def __init__(self,
                 origin: np.ndarray,
                 rotation: np.ndarray,
                 aperture: Aperture):
        """ Create a Screen object
        Args:
            origin: 1 x 3 numpy array representing the origin of the screen in 3D space
            rotation: 3 x 3 numpy array representing the rotation matrix of the screen in 3D space
                    this matrix transforms local screen coordinates to global coordinates
            aperture: an Aperture object that represents the screen
            stop_rays: a boolean that indicates if the rays should stop at the screen or continue"""
        self.origin = np.array(origin)
        self.rotation = np.array(rotation)
        self.aperture = aperture

    def reset(self):
        """ Reset the screen to its initial state"""
        pass

    def capture(self, ray_bundle: RayBundle):
        """ Capture the rays on the screen
        Args:
            ray_bundle: a RayBundle object"""
        # Start by computing the coordinates of the rays in the screen frame
        local_origins, local_directions = ray_bundle.globalToLocal(self.origin, self.rotation)
        # Compute the intersection of the rays with the screen. In the local frame,
        # the screen is the plane at z = 0
        # Filter all the rays that are parralel to the screen
        idx = np.where(np.abs(local_directions[:, 2]) > 1e-6)[0]
        t = -local_origins[idx, 2] / local_directions[idx, 2]
        intersection_points = local_origins + t[idx, np.newaxis] * local_directions

    def mesh(self):
        """ Generate a mesh representing the screen, in global coordinates.
            Used to display the screen
        Returns:
          mesh_x: The x coordinates of the mesh points, as a 2D array
          mesh_y: The y coordinates of the mesh points, as a 2D array
          mesh_z: The z coordinates of the mesh points, as a 2D array"""
        mesh_x, mesh_y = self.aperture.mesh()
        xs = mesh_x.flatten()
        ys = mesh_y.flatten()
        zs = np.zeros_like(xs)
        # Transform the mesh to global coordinates
        points_local = np.array([xs, ys, zs]).T
        points_global = np.matmul(points_local, self.rotation.T) + self.origin
        mesh_x = np.reshape(points_global[:, 0], mesh_x.shape)
        mesh_y = np.reshape(points_global[:, 1], mesh_x.shape)
        mesh_z = np.reshape(points_global[:, 2], mesh_x.shape)
        return mesh_x, mesh_y, mesh_z
