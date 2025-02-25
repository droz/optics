from optics.surfaces import SphericalSurface
from optics.apertures import CircularAperture, RectangularAperture
from optics.display import display
from optics.sources import GaussianBeamSource
import numpy as np

# Create a spherical surface with a circular aperture
origin = np.array([0, 0, 0])
rotation = np.eye(3)
aperture = CircularAperture(1)
#aperture = RectangularAperture(1, 1)
index1 = 1
index2 = 1.5
radius = -1.5
surface = SphericalSurface(origin, rotation, aperture, index1, index2, radius)

points = np.array([[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]])

rotation = [[-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]]
source = GaussianBeamSource(np.array([0, 0, 0]), rotation, 1e-6, 2e-6, 1, 850e-9)
rays = source.generate(100000)

display(rays, show_all_rays=True)

#display(surface, show_normals=True)