from dataclasses import dataclass
import numpy as np

@dataclass
class RayBundle():
    """ A RayBundle is a collection of rays."""
    # The origins of the rays
    origins: np.ndarray
    # The directions of the rays (unit vectors)
    directions: np.ndarray
    # The lengths of the rays. zero length means the ray is infinite
    lengths: np.ndarray
    # The wavelengths of the rays, in meters
    wavelengths_m: np.ndarray
    # The powers of each ray, in watts
    powers_w: np.ndarray
    # The indices of the "display" rays. These are the rays that are used to display the bundle and will
    # not be used to compute the total power or other quantities.
    display_rays: list

    def normalize_directions(self):
        """ Normalize the directions of the rays"""
        self.directions /= np.linalg.norm(self.directions, axis=1)[:, np.newaxis]
        