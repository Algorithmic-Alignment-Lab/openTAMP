import pyro
from pyro.distributions import Distribution
import torch
import torch.distributions.constraints as constraints


class CustomDist(Distribution):
    def __init__(self, sample_data):
        self.sample_data = sample_data
        self.support = constraints.real_vector

    def sample(self):
        rand_idx = torch.randint(high=self.sample_data.shape[0], size=(1,))
        return self.sample_data[rand_idx]

    def log_prob(self, x, *args, **kwargs):
        print(x)
        print(x-self.sample_data)
        print(x.requires_grad)
        print(self.sample_data.requires_grad)
        equalities = torch.norm(x - self.sample_data, p=2) < 0.001
        print(equalities.requires_grad)
        return equalities.sum().log()