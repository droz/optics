from optics.surfaces import SphericalSurface
from optics.apertures import CircularAperture, RectangularAperture
from optics.display import display
from optics.sources import GaussianBeamSource
from optics.screen import Screen
from optics.system import System
import numpy as np

# Create a test optical system.
# The system will be lined up with the x axis, this will allow us to test the rotation of the surfaces
system = System()
rotation = [[0,  0,  1],
            [0, -1,  0],
            [1,  0,  0]]

# Create a gaussian beam source
source_origin = np.array([-0.1, 0, 0])
source = GaussianBeamSource(origin=source_origin,
                            rotation=rotation,
                            waist_x=1e-6,
                            waist_y=2e-6,
                            power_w=1,
                            wavelength_m=850e-9)
system.setSource(source)

# Create a spherical surface with a circular aperture
origin = np.array([0.1, 0, 0])
surface = SphericalSurface(origin, rotation, CircularAperture(0.05), 1, 1.5, -0.2)
system.addSurface(surface)

# Add a screen to visualize what is going on
screen = Screen(np.array([0.03, 0, 0]), rotation, CircularAperture(0.05))
system.addScreen(screen)

# Generate and propagate rays
system.propagate(1000)

# Display the results
display(system)