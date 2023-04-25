import pyro
import pyro.distributions as dist


class Belief(object):
    """
    Base class of every Objects in environment
    """
    def __init__(self, size):
        self._type = "belief"
        self._base_type = "belief"
        self.samples = None
        self.dist = None
        self.size = size


class UniformBelief(Belief):
    def __init__(self, shape, low, high):
        super().__init__(int(shape))
        self.dist = dist.Uniform(float(low), float(high))  # hard-coded


# used for updates (for now: just updates a sample)
def belief_constructor(samples, size):
    class UpdatedBelief(Belief):
        def __init__(self):
            super().__init__(size)
            self.samples = samples

    return UpdatedBelief
