from .sources import Source

class System:
    """A system of optical elements."""
    def __init__(self):
        """Create a System object."""
        self.source = None
        self.surfaces = []
        self.screens = []
        self.rays = []

    def setSource(self, source : Source):
        """Set the ray source for the system (for now we only allow a single source)
        Args:
            source: a Source object"""
        self.source = source
    
    def addSurface(self, surface):
        """Add a surface to the system
        Args:
            surface: a Surface object"""
        self.surfaces.append(surface)

    def addScreen(self, screen):
        """Add a screen to the system
        Args:
            screen: a Screen object"""
        self.screens.append(screen)

    def propagate(self, n_rays: int):
        """Generate rays from the sources and propagate them through the system
        Args:
            n_rays: the number of rays to generate
        """
        # Start by generating the rays from the source
        self.rays = [self.source.generate(n_rays)]



        # Then go over all the rays and intersect with all the screens
        for screen in self.screens:
            screen.reset()
            for rays in self.rays:
                screen.capture(rays)