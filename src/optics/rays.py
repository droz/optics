class RayBundle:
    def __init__(self, origins, directions, powers, index):
        """ Create a RayBundle object
        Args:
          origins: a N x 3 numpy array representing the origins of the rays
          directions: a N x 3 numpy array representing the directions of the rays (unit vectors)
          powers: a N x 1 numpy array representing the power of the rays"""
        self.origins = origins
        self.directions = directions
        self.powers = powers
        self.index = index