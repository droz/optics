from optics.rays import RayBundle
import numpy as np

class Source:
    """Sources are objects that generate ray bundles.
    A derived class should implement the generate function, which generates a ray bundle."""
    def __init__(self, origin, direction):
        """ Create a Source object
        Args:
          origin: 1 x 3 numpy array representing the origin of the source in 3D space
          direction: 1 x 3 numpy array representing the direction of the source in 3D space"""
        self.origin = origin
        self.direction = direction

    def generate(self, n_rays):
        """ Generate a ray bundle
        Args:
            n_rays: the number of rays to generate
        Returns:
            a RayBundle object"""
        raise NotImplementedError("generate() is not implemented by the base class")
    
class GaussianBeamSource(Source):
    """ A GaussianBeamSource is a Source that generates a Gaussian beam"""
    def __init__(self, origin, direction, waist_x, waist_y, power_w, wavelength_m):
        """ Create a GaussianBeamSource object
        Args:
            origin: 1 x 3 numpy array representing the origin of the source in 3D space
            direction: 1 x 3 numpy array representing the direction of the source in 3D space
            waist_x: the waist of the beam in the x direction
            waist_y: the waist of the beam in the y direction
            power_w: the power of the source, in watts
            wavelength_m: the wavelength of the source in meters"""
        super().__init__(origin, direction)
        self.waist_x = waist_x
        self.waist_y = waist_y
        self.wavelength_m = wavelength_m
        self.power_w = power_w

    def generate(self, n_rays):
        """ Generate a ray bundle
        Args:
            n_rays: The number of rays to generate.
        Returns:
            a RayBundle object"""
        # We need to allocate the number of rays to the four different parameters:
        # x, y, theta_x, theta_y. For now we do that evenly (as many rays for each parameter)
        n_rays_per_parameter = n_rays ** (1/4)
        n_rays_per_parameter = int(np.ceil(n_rays_per_parameter))
        # Always make sure that there is a center ray
        if n_rays_per_parameter % 2 == 0:
            n_rays_per_parameter += 1
        # Compute the beam size at 1m of the waist (see https://en.wikipedia.org/wiki/Gaussian_beam)
        zr_x = np.pi * self.waist_x ** 2 / self.wavelength_m
        zr_y = np.pi * self.waist_y ** 2 / self.wavelength_m
        size_1m_x = self.waist_x * np.sqrt(1 + (1 / zr_x) ** 2)
        size_1m_y = self.waist_y * np.sqrt(1 + (1 / zr_y) ** 2)
        # Generate the rays. We start with the center ray and the two rays at the waist. These will be display rays.
        # Center ray
        origins = [[0, 0, 0]]
        directions = [[0, 0, 1]]
        # Rays on the ellipse
        num_ellipse_rays = 12
        for theta in np.linspace(0, 2 * np.pi, num_ellipse_rays, endpoint=False):
            ctheta = np.cos(theta)
            stheta = np.sin(theta)
            orig_x = self.waist_x * ctheta
            orig_y = self.waist_y * stheta
            dir_x = size_1m_x * ctheta - orig_x
            dir_y = size_1m_y * stheta - orig_y
            origins.append([orig_x, orig_y, 0])
            directions.append([dir_x, dir_y, 1])
        rays = RayBundle(
            origins = np.array(origins),
            directions = np.array(directions),
            lengths = np.full(len(origins), 0),
            wavelengths_m = np.full(len(origins), self.wavelength_m),
            powers_w = np.full(len(origins), 0),
            display_rays = range(1 + num_ellipse_rays))
        
        return rays