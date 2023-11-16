import pyro
import pyro.distributions as distros
import opentamp.core.util_classes.matrix as matrix
import numpy as np


class Belief(object):
    """
    Base class of every Belief in environment
    """
    def __init__(self, size, num_samples):
        self._type = "belief"
        self._base_type = "belief"
        self.size = size
        self.num_samples = num_samples


# NOTE: works terribly with MCMC
class UniformBelief(Belief):
    def __init__(self, args):
        self.dist = distros.Uniform(float(args[2]), float(args[3]))  # hard-coded
        self._type = "unif_belief"
        super().__init__(int(args[0]), int(args[1]))
        tensor_samples = self.dist.sample_n(self.num_samples * self.size)
        self.samples = tensor_samples.view(self.num_samples, self.size) # sample from prior


class IsotropicGaussianBelief(Belief):
    def __init__(self, args):
        self.dist = distros.Normal(float(args[2]), float(args[3]))  # hard-coded
        self._type = "gaussian_belief"
        super().__init__(int(args[0]), int(args[1]))
        tensor_samples = self.dist.sample_n(self.num_samples * self.size)
        self.samples = tensor_samples.view(self.num_samples, self.size, 1)  # sample from prior



# used for updates (for now: just updates a sample)
def belief_constructor(samples=None, size=1):
    class UpdatedBelief(Belief):
        def __init__(self, size, samples):
            super().__init__(size, samples.shape[0])
            self.samples = samples

    return UpdatedBelief(size, samples)
