## set of methods taking forward samples for observation
import torch
import pyro
import pyro.distributions as dist
from opentamp.core.util_classes.custom_dist import CustomDist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, SVI, TraceEnum_ELBO, config_enumerate
from pyro.infer.autoguide import AutoDelta
import numpy as np



class ObservationModel(object):
    approx_params : None  ## parameters into the parametric approximation for the current belief state
    active_planned_observations : None  ## dict of parameter names, mapping to the current observed values

    ## get the observation that is currently active / being planned over
    def get_active_planned_observations(self):
        return self.active_planned_observations
    
    ## get the current parameters for training the (variational) approximation to a given belief state
    def get_approx_params(self):
        return self.approx_params
    
    ## get the observation that is currently active / being planned over
    def set_active_planned_observations(self, planned_obs):
        self.active_planned_observations = planned_obs

    ## sample an observation, optionally conditioned on true state, and set active_planned observation
    ## can override with different schemes (max-likelihood observation, average, etc...)
    # def resample_planned_observation(self, params):
    #     ## by default, just samples from the prior of each belief state         
    #     planned_obs = {}

    #     ## resample the prior for target1
    #     for param_key in params:
    #         if hasattr(params[param_key], 'belief'):
    #             planned_obs[param_key] = params[param_key].belief.dist.sample().detach()  ## sample the prior 

    #     self.active_planned_observations = planned_obs

    ## a callable giving a parametric form for the approximation to the belief posterior, used subsequently in MCMC
    def approx_model(self, data):
        raise NotImplementedError
    
    ## a callable fiting the approx_params dict on a set of samples
    def fit_approximation(self, params):
        raise NotImplementedError

    ## a callable representing the forward model
    def forward_model(self, params, active_ts, provided_state=None):
        raise NotImplementedError

class PointerObservationModel(ObservationModel):
    def __init__(self):
        # uninitialized parameters
        self.approx_params = {'weights': None, 'locs': None, 'scales': None}
        self.active_planned_observations = {'target1': torch.empty((2,)).detach()}
    
    @config_enumerate
    def approx_model(self, data):
        ## Global variable (weight on either cluster)
        weights = pyro.sample("weights", dist.Dirichlet(5 * torch.ones(2)))

        ## Different Locs and Scales for each
        with pyro.plate("components", 2):
            ## Uninformative prior on locations
            locs = pyro.sample("locs", dist.MultivariateNormal(torch.tensor([3.0, 0.0]), 20.0 * torch.eye(2)))
            scales = pyro.sample("scales", dist.LogNormal(0.0, 10.0))

        with pyro.plate("data", len(data)):
            ## Local variables
            assignment = pyro.sample("mode_assignment", dist.Categorical(weights))
            stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(100, 1, 1))
            stack_scale = torch.tile(scales[assignment].unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
            cov_tensor = stack_eye * stack_scale
            pyro.sample("belief_global", dist.MultivariateNormal(locs[assignment], cov_tensor), obs=data)

    def forward_model(self, params, active_ts, provided_state=None):        
        ray_width = np.pi / 4  ## has 45-degree field of view on either side

        def is_in_ray(a_pose, target):
            if target[0] >= 0:
                return np.abs(np.arctan(target[1]/target[0]) - a_pose) <= ray_width
            else:
                return np.abs(np.arctan(target[1]/target[0]) - (a_pose - np.pi)) <= ray_width
        
        ## construct Gaussian mixture model using the current approximation
        cat_dist = dist.Categorical(probs=self.approx_params['weights'])
        stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(2, 1, 1))
        stack_scale = torch.tile(self.approx_params['scales'].unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
        cov_tensor = stack_eye * stack_scale
        batched_multivar = dist.MultivariateNormal(loc=self.approx_params['locs'], covariance_matrix=cov_tensor)
        mix_dist = dist.MixtureSameFamily(cat_dist, batched_multivar)

        if provided_state is not None:
            ## overrides the current belief sample with a true state
            b_global_samp = provided_state['target1']
        else:
            ## sample from current Gaussian mixture model
            b_global_samp = pyro.sample('belief_global', mix_dist)
        
        samps = {}

        if is_in_ray(params['pr2'].pose[0,active_ts[1]-1], b_global_samp.detach()):
            ## sample around the true belief, with extremely low variation / error
            samps['target1'] = pyro.sample('target1', dist.MultivariateNormal(b_global_samp.float(), 0.01 * torch.eye(2)))
        else:
            ## sample from prior dist -- have no additional knowledge, don't read it
            samps['target1'] = pyro.sample('target1', dist.MultivariateNormal(torch.zeros((2,)), 0.01 * torch.eye(2)))

        return samps

    def fit_approximation(self, params):        
        ## clear Pyro optimization context, for 
        pyro.clear_param_store()
 
        def init_loc_fn(site):
            K=2
            data=params['target1'].belief.samples[:, :, -1]
            if site["name"] == "weights":
                # Initialize weights to uniform.
                return torch.ones(2) / 2
            if site["name"] == "scales":
                return torch.ones(2)
            if site["name"] == "locs":
                return torch.tensor([[3., 3.], [3., -3.]])
            raise ValueError(site["name"])


        def initialize():
            # global global_guide, svi
            # data=params['target1'].belief.samples[:, :, -1]
            pyro.clear_param_store()

        # Choose the best among 100 random initializations.
        # loss, seed = min((initialize(seed), seed) for seed in range(100))
        initialize()
        # print(f"seed = {seed}, initial_loss = {loss}")

        global_guide = AutoDelta(
            poutine.block(self.approx_model, expose=["weights", "locs", "scales"]),
            init_loc_fn=init_loc_fn,
        )
        adam_params = {"lr": 0.1, "betas": [0.8, 0.99]}
        optimizer = pyro.optim.Adam(adam_params)

        svi = SVI(self.approx_model, global_guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))

        ## setup the inference algorithm
        nsteps = 1000  ## NOTE: causes strange bugs when run too long (weights concentrate to 1)

        ## do gradient steps, TODO update with genreal belief signature 
        for _ in range(nsteps):
            loss = svi.step(params['target1'].belief.samples[:, :, -1])
            # print(global_guide(params['target1'].belief.samples[:, :, -1]))

        pars = global_guide(params['target1'].belief.samples[:, :, -1])
        
        ## need to detach for observation_model to be serializable
        for p in pars:
            pars[p] = pars[p].detach()

        self.approx_params = pars


## TODO implement one with just clustering, not even using GMM
class NoVIPointerObservationModel(ObservationModel):
    def __init__(self):
        # uninitialized parameters
        self.approx_params = {'loc': None, 'cov': None}
        self.active_planned_observations = {'target1': torch.empty((2,)).detach()}

    def forward_model(self, params, active_ts, provided_state=None):        
        ray_width = np.pi / 4  ## has 45-degree field of view on either side

        def is_in_ray(a_pose, target):
            return np.abs(np.arctan(target[1]/target[0]) - a_pose) <= ray_width
        
        ## construct Gaussian mixture model using the current approximation

        # cat_dist = dist.Categorical(probs=self.approx_params['weights'])
        # stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(2, 1, 1))
        # stack_scale = torch.tile(self.approx_params['scales'].unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
        # cov_tensor = stack_eye * stack_scale
        # batched_multivar = dist.MultivariateNormal(loc=self.approx_params['locs'], covariance_matrix=cov_tensor)
        # mix_dist = dist.MixtureSameFamily(cat_dist, batched_multivar)

        multiv_gauss = dist.MultivariateNormal(loc=self.approx_params['loc'], covariance_matrix=self.approx_params['cov'])

        if provided_state is not None:
            ## overrides the current belief sample with a true state
            b_global_samp = provided_state['target1']
        else:
            ## sample from current Gaussian mixture model
            b_global_samp = pyro.sample('belief_global', multiv_gauss)
        
        samps = {}

        if is_in_ray(params['pr2'].pose[0,active_ts[1]], b_global_samp.detach()):
            ## sample around the true belief, with extremely low variation / error
            samps['target1'] = pyro.sample('target1', dist.MultivariateNormal(b_global_samp.float(), 0.01 * torch.eye(2)))
        else:
            ## sample from prior belief (*NOT* from the current mixture)
            samps['target1'] = pyro.sample('target1', params['target1'].belief.dist)

        return samps
    
    ## override resample logic by simply observing the mean
    # def resample_planned_observation(self, params):
    #     ## by default, just samples from the prior of each belief state         
    #     planned_obs = {}

    #     ## resample the prior for target1
    #     for param_key in params:
    #         if hasattr(params[param_key], 'belief'):
    #             planned_obs[param_key] = params[param_key].belief.samples[:,:,-1].mean(axis=0) ## return the mean

    #     self.active_planned_observations = planned_obs


    ## LaPlace approximation -- Gaussian param
    def fit_approximation(self, params):
        belief_data = params['target1'].belief.samples[:, :, -1]
        self.approx_params = {'loc': belief_data.mean(axis=0), 'cov': belief_data.T.cov()}



## bivariate Gaussian guide to perform SVI
# def basic_gaussian_mix_guide(data):
#     # registering var
#     mean_1 = pyro.param("mean_1", torch.tensor([3.0, 3.0]).unsqueeze(dim=0))
#     mean_2 = pyro.param("mean_2", torch.tensor([3.0, 3.0]).unsqueeze(dim=0))
    
#     var_1 = pyro.param("var_1", torch.tensor(1.0), constraint=dist.constraints.positive)
#     var_2 = pyro.param("var_2", torch.tensor(1.0), constraint=dist.constraints.positive)

#     cluster_1_prob = pyro.param("cluster_1_prob", torch.tensor([0.5]), constraint=dist.constraints.interval(0.0, 1.0))
    
#     ## constructing stack for Gaussian Mixture
#     prob_vec = torch.cat([cluster_1_prob, 1 - cluster_1_prob])
#     mean_tensor = torch.cat((mean_1, mean_2), dim=0)
#     var_tensor = torch.cat(((var_1 *torch.eye(2)).unsqueeze(dim=0), (var_2*torch.eye(2)).unsqueeze(dim=0)), dim=0)

#     # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
#     pyro.sample("belief_samp", dist.MixtureSameFamily(dist.Categorical(probs=prob_vec), dist.MultivariateNormal(mean_tensor, var_tensor), validate_args=None))



# ## dummy observation model
# def dummy_obs(plan, active_ts):
#     # start observations in the first action todo: loop this over actions in the plan
#     pyro.sample('belief_global', dist.Normal(0, 1))

#     # return torch.tensor([0, 0])  # return observation
