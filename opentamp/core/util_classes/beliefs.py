import pyro
import pyro.distributions as distros
import torch
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
        self.dist = None  ## must override


# NOTE: works terribly with Hamiltonian Monte Carlo (so **DON'T USE** for inference)
class UniformPrior(Belief):
    def __init__(self, args):
        super().__init__(2, int(args[0]))
        self.dist = distros.Uniform(torch.tensor([float(args[1]), float(args[2])]), torch.tensor([float(args[3]), float(args[4])]))   # hard-coded
        self._type = "unif_belief"
        tensor_samples = self.dist.sample_n(self.num_samples)
        self.samples = tensor_samples.view(self.num_samples, self.size, 1) # sample from prior


## spherical Gaussians
class Isotropic2DGaussianPrior(Belief):
    def __init__(self, args):
        super().__init__(int(args[0]), int(args[1]))
        self.dist = distros.MultivariateNormal(torch.tensor([float(args[2]), float(args[3])]), float(args[4]) * torch.eye(2)) # hard-coded
        self._type = "isotropic_gaussian_prior"
        tensor_samples = self.dist.sample_n(self.num_samples)
        self.samples = tensor_samples.view(self.num_samples, self.size, 1)  # sample from prior

## mixture of Spherical Gaussians
class MixedIsotropic2DGaussianPrior(Belief):
    def __init__(self, args):
        super().__init__(int(args[0]), int(args[1]))
        weights = torch.tensor([float(args[2]), float(args[3])])
        locs = torch.tensor([[float(args[4]), float(args[5])],
                             [float(args[7]), float(args[8])]])
        scales = torch.tensor([float(args[6]), float(args[9])])
        cat_dist = distros.Categorical(probs=weights)
        stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(2, 1, 1))
        stack_scale = torch.tile(scales.unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
        cov_tensor = stack_eye * stack_scale
        batched_multivar = distros.MultivariateNormal(loc=locs, covariance_matrix=cov_tensor)
        self.dist =  distros.MixtureSameFamily(cat_dist, batched_multivar)
        self._type = "mixed_isotropic_gaussian_prior"
        tensor_samples = self.dist.sample_n(self.num_samples)
        self.samples = tensor_samples.view(self.num_samples, self.size, 1)  # sample from prior


# # used for updates (for now: just updates a sample)
# def belief_constructor(samples=None, size=1):
#     class UpdatedBelief(Belief):
#         def __init__(self, size, samples):
#             super().__init__(size, samples.shape[0])
#             self.samples = samples

#     return UpdatedBelief(size, samples)
