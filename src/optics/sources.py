from optics.rays import RayBundle
import numpy as np

class Source:
    """Sources are objects that generate ray bundles.
    A derived class should implement the generate function, which generates a ray bundle."""
    def __init__(self, origin, rotation):
        """ Create a Source object
        Args:
            origin: 1 x 3 numpy array representing the origin of the source in 3D space
            rotation: 3 x 3 numpy array representing the rotation matrix of the surface in 3D space
                    this matrix transforms local surface coordinates to global coordinates"""
        self.origin = origin
        self.rotation = rotation

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
        # Compute the beam size at 1m of the waist (see https://en.wikipedia.org/wiki/Gaussian_beam)
        zr_x = np.pi * self.waist_x ** 2 / self.wavelength_m
        zr_y = np.pi * self.waist_y ** 2 / self.wavelength_m
        size_1m_x = self.waist_x * np.sqrt(1 + (1 / zr_x) ** 2)
        size_1m_y = self.waist_y * np.sqrt(1 + (1 / zr_y) ** 2)
        # Generate the rays. We start with the display rays.
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
        origins = np.array(origins)
        directions = np.array(directions)
        powers = np.full(len(origins), 0)
        
        # Now we generate the rest of the rays. We pick them at random in the beam
        # Nearly 100% of the power should be contained within a circle of radius 2 * waist
        orig_x = np.random.normal(0, self.waist_x / 2.0, n_rays)
        orig_y = np.random.normal(0, self.waist_y / 2.0, n_rays)
        starts = np.array([orig_x, orig_y, np.zeros(n_rays)]).T
        end_x = np.random.normal(0, size_1m_x / 2.0, n_rays)
        end_y = np.random.normal(0, size_1m_y / 2.0, n_rays)
        ends = np.array([end_x, end_y, np.ones(n_rays)]).T
        origins = np.vstack((origins, starts))
        directions = np.vstack((directions, ends - starts))
        powers = np.concatenate((powers, np.full(n_rays, self.power_w / n_rays)))

        rays = RayBundle(
            origins = np.array(origins),
            directions = np.array(directions),
            lengths = np.full(len(origins), 0),
            wavelengths_m = np.full(len(origins), self.wavelength_m),
            powers_w = powers,
            display_rays = range(1 + num_ellipse_rays))

        rays.normalizeDirections()
        rays.transformFromLocalToGlobal(self.origin, self.rotation)

        return rays