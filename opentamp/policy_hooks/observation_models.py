## set of methods taking forward samples for observation
import torch
import pyro
import pyro.distributions as dist
from opentamp.core.util_classes.custom_dist import CustomDist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS

def toy_observation(rs_params, belief_mean, belief_cov):
    # LaPlace estimate: todo SVI if needed
    b_global = pyro.sample('belief_global', dist.Normal(belief_mean, belief_cov))

    if rs_params is None:
        return b_global

    # start observations in the first action todo: loop this over actions in the plan
    obs = torch.torch.empty(rs_params[0].pose.shape[1]-1)
    print(obs.shape)
    for a in rs_params:
        for i in range(1, rs_params[0].pose.shape[1]):
            # differentially take conditional depending on the ray
            # 1.10714871779
            if is_in_ray(a.pose[0][i], b_global.item()):
                obs[i - 1] = pyro.sample('obs'+str(i), dist.Uniform(b_global-torch.tensor(0.001), b_global+torch.tensor(0.001)))
            else:
                obs[i - 1] = pyro.sample('obs'+str(i), dist.Uniform(b_global-torch.tensor(1), b_global+torch.tensor(1)))  # no marginal information gotten

    return obs

## dummy_obs commands

def dummy_obs(rs_params, belief_mean, belief_cov):
    # start observations in the first action todo: loop this over actions in the plan
    b_global = pyro.sample('belief_global', dist.Normal(belief_mean, belief_cov))

    return b_global
