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
        return torch.tensor(self.sample_data[rand_idx])

    def log_prob(self, x, *args, **kwargs):
        print(x)
        print(x-self.sample_data)
        equalities = torch.norm(x - self.sample_data, p=2) < 0.001
        if equalities.any():  # if is a sample
            return -torch.log(torch.tensor(self.sample_data.shape[0]))
        else:
            return -torch.inf