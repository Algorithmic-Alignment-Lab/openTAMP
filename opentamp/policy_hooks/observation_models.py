## set of methods taking forward samples for observation
import torch
import pyro
import pyro.distributions as dist
from opentamp.core.util_classes.custom_dist import CustomDist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS
import numpy as np


def pointer_observation(params, active_ts, true_goal=None):    
    ray_width = np.pi / 16

    def is_in_ray(a_pose, target):
        return np.abs(np.arctan(target[1]/target[0]) - a_pose) <= ray_width

    # b_global_samp = pyro.sample('belief_global', dist.MultivariateNormal(belief_samp.float(), torch.eye(2)))  # samples and adds Gaussian noise, for the sake of probabilistic model
    
    belief_tensor = params['target1'].belief.samples[:, :, -1]  # creating a dataset 
    
    # Laplace approximation, in order to avoid ugliness in internal discrete sampling
    b_global_samp = pyro.sample('belief_global', dist.MultivariateNormal(belief_tensor.mean(axis=0), belief_tensor.T.cov()))

    if is_in_ray(params['pr2'].pose[0,active_ts[1]], b_global_samp.detach()):
        # obs['obs'+str(ts)] = pyro.sample('obs'+str(ts), dist.MultivariateNormal(belief_samp.float(), torch.eye(2)))
        pyro.sample('target1', dist.MultivariateNormal(b_global_samp.float(), 0.1 * torch.eye(2)))
    else:
        # no detection
        pyro.sample('target1', dist.MultivariateNormal(-3 * torch.ones(2), 0.5*torch.eye(2)))

    ml_obs = {}

    if true_goal is not None:
        ml_obs['target1'] = torch.tensor(true_goal)
    else:
        ml_obs['target1'] = belief_tensor[0,:]  # return randonly sampled tensor from the belief-tensor for now

    return ml_obs


## dummy observation model
def dummy_obs(plan, active_ts):
    # start observations in the first action todo: loop this over actions in the plan
    pyro.sample('belief_global', dist.Normal(0, 1))

    # return torch.tensor([0, 0])  # return observation
