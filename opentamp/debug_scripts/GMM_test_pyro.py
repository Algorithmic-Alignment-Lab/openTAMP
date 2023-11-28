import torch
import pyro
import pyro.distributions as dist
from opentamp.core.util_classes.custom_dist import CustomDist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, SVI, TraceEnum_ELBO, Trace_ELBO, config_enumerate
from pyro.infer.autoguide import AutoDelta
import numpy as np

pyro.clear_param_store()
 
# def init_loc_fn(site):
#     K=2
#     data=params['target1'].belief.samples[:, :, -1]
#     if site["name"] == "weights":
#         # Initialize weights to uniform.
#         return torch.ones(2) / 2
#     if site["name"] == "scales":
#         return torch.ones(2)
#     if site["name"] == "locs":
#         return torch.tensor([[3., 3.], [3., -3.]])
#     raise ValueError(site["name"])


# def initialize(seed):
#     global global_guide, svi
#     data=params['target1'].belief.samples[:, :, -1]
#     pyro.clear_param_store()
#     global_guide = AutoDelta(
#         poutine.block(self.approx_model, expose=["weights", "locs", "scales"]),
#         init_loc_fn=init_loc_fn,
#     )
#     adam_params = {"lr": 0.1, "betas": [0.8, 0.99]}
#     optimizer = pyro.optim.Adam(adam_params)

#     svi = SVI(self.approx_model, global_guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))
#     return svi.loss(self.approx_model, global_guide, data)


## make simulated data to test Pyro VI approx with
cat_dist = dist.Categorical(probs=torch.tensor([0.75, 0.25]))
stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(2, 1, 1))
stack_scale = torch.tile(torch.tensor([1, 1]).unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
cov_tensor = stack_eye * stack_scale
batched_multivar = dist.MultivariateNormal(loc=torch.tensor([[4., 2.],[3., 0.]]), covariance_matrix=cov_tensor)
mix_dist = dist.MixtureSameFamily(cat_dist, batched_multivar)

data = mix_dist.sample_n(100)

def init_loc_fn(site):
    if site["name"] == "weights":
        # Initialize weights to uniform.
        return torch.ones(2) / 2
    if site["name"] == "scales":
        return torch.ones(2) / 2
    if site["name"] == "locs":
        return torch.tensor([[3., 3.], [3., -3.]])
    raise ValueError(site["name"])

## Global variable (weight on either cluster)
@config_enumerate
def model(data):
    weights = pyro.sample("weights", dist.Dirichlet(20.0 * torch.ones(2)))

    ## Different Locs and Scales for each
    with pyro.plate("components", 2):
        ## Uninformative prior on locations
        locs = pyro.sample("locs", dist.MultivariateNormal(torch.tensor([3.0, 0.0]), 20*torch.eye(2)))
        scales = pyro.sample("scales", dist.LogNormal(0.0, 10.0))

    with pyro.plate("data", len(data)):
        ## Local variables
        assignment = pyro.sample("mode_assignment", dist.Categorical(weights))
        stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(100, 1, 1))
        stack_scale = torch.tile(scales[assignment].unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
        cov_tensor = stack_eye * stack_scale
        pyro.sample("belief_global", dist.MultivariateNormal(locs[assignment], cov_tensor), obs=data)


# Choose the best among 100 random initializations.
# loss, seed = min((initialize(seed), seed) for seed in range(100))
# initialize(seed)
# print(f"seed = {seed}, initial_loss = {loss}")

adam_params = {"lr": 0.001, "betas": [0.8, 0.99]}
optimizer = pyro.optim.Adam(adam_params)
guide = AutoDelta(
        poutine.block(model, expose=["weights", "locs", "scales"]),
        init_loc_fn=init_loc_fn,
    )

svi = SVI(model, guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))

## setup the inference algorithm
nsteps = 1000  ## NOTE: causes strange bugs when run too long (weights concentrate to 1)

## do gradient steps, TODO update with genreal belief signature 
for _ in range(nsteps):
    loss = svi.step(data)
    # print(global_guide(params['target1'].belief.samples[:, :, -1]))

print(guide(data))
