import numpy as np
import torch

PRIOR_CHOICES = {} # distributions
MAX_PRIOR_CHOICES = {} # max-likelihoods for those priors


class Belief:
    """
    Belief tracks a likelihood for belief-state planning (tracking happens elsewhere), stores a likelihood filter
    """
    def __init__(self, prior_choice_str):
        self.likelihood = PRIOR_CHOICES[prior_choice_str]
        self.max_likelihood = MAX_PRIOR_CHOICES[prior_choice_str]
        self.prior_choice_str = prior_choice_str

    def __str__(self):
        return self.prior_choice_str

    def filter_likelihood(self, forward_likelihood):
        pass

    def gen_sample(self):
        pass
