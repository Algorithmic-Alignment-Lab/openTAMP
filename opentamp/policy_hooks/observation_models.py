## set of methods taking forward samples for observation
import torch
import pyro
import pyro.distributions as dist
from opentamp.core.util_classes.custom_dist import CustomDist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, SVI, TraceEnum_ELBO, config_enumerate
from pyro.infer.autoguide import AutoDelta
import numpy as np
import os

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
# print('USED DEVICE FOR MCMC INFERENCE IS: ', DEVICE)

class ObservationModel(object):
    approx_params : None  ## parameters into the parametric approximation for the current belief state
    active_planned_observations : None  ## dict of parameter names, mapping to the current observed values
    past_obs : {} ## dict of past observations to condition on, indexed {timestep: observation}

    ## get the observation that is currently active / being planned over
    def get_active_planned_observations(self):
        return self.active_planned_observations
    
    ## get the current parameters for training the (variational) approximation to a given belief state
    def get_approx_params(self):
        return self.approx_params
    
    ## get the observation that is currently active / being planned over
    def set_active_planned_observations(self, planned_obs):
        self.active_planned_observations = planned_obs

    def get_obs_likelihood(self, obs):
        raise NotImplementedError

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
        self.approx_params = {'weights'+str(os.getpid()): None, 'locs'+str(os.getpid()): None, 'scales'+str(os.getpid()): None}
        self.active_planned_observations = {'target1': torch.empty((2,)).detach()}
    
    @config_enumerate
    def approx_model(self, data):
        ## Global variable (weight on either cluster)
        weights = pyro.sample("weights"+str(os.getpid()), dist.Dirichlet(5 * torch.ones(2)))

        ## Different Locs and Scales for each
        with pyro.plate("components"+str(os.getpid()), 2):
            ## Uninformative prior on locations
            locs = pyro.sample("locs"+str(os.getpid()), dist.MultivariateNormal(torch.tensor([3.0, 0.0]), 20.0 * torch.eye(2)))
            scales = pyro.sample("scales"+str(os.getpid()), dist.LogNormal(0.0, 10.0))

        with pyro.plate("data"+str(os.getpid()), len(data)):
            ## Local variables
            assignment = pyro.sample("mode_assignment"+str(os.getpid()), dist.Categorical(weights))
            stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(100, 1, 1))
            stack_scale = torch.tile(scales[assignment].unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
            cov_tensor = (stack_eye * stack_scale)
            pyro.sample("belief_global"+str(os.getpid()), dist.MultivariateNormal(locs[assignment], cov_tensor), obs=data)

    def forward_model(self, params, active_ts, provided_state=None, past_obs={}):        
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

        # return tensors on CPU for compatib
        for samp in samps:
            samps[samp].to('cpu')

        return samps

    def fit_approximation(self, params):        
        def init_loc_fn(site):
            K=2
            data=params['target1'].belief.samples[:, :, -1]
            if site["name"] == "weights"+str(os.getpid()):
                # Initialize weights to uniform.
                return (torch.ones(2) / 2)
            if site["name"] == "scales"+str(os.getpid()):
                return torch.ones(2)
            if site["name"] == "locs"+str(os.getpid()):
                return torch.tensor([[3., 3.], [3., -3.]])
            raise ValueError(site["name"])


        def initialize():
            ## clear Pyro optimization context, for 
            pyro.clear_param_store()

        # Choose the best among 100 random initializations.
        # loss, seed = min((initialize(seed), seed) for seed in range(100))
        initialize()
        # print(f"seed = {seed}, initial_loss = {loss}")

        global_guide = AutoDelta(
            poutine.block(self.approx_model, expose=["weights"+str(os.getpid()), "locs"+str(os.getpid()), "scales"+str(os.getpid())]),
            init_loc_fn=init_loc_fn,
        )
        adam_params = {"lr": 0.01, "betas": [0.99, 0.3]}
        optimizer = pyro.optim.Adam(adam_params)

        svi = SVI(self.approx_model, global_guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))

        ## setup the inference algorithm
        nsteps = 200  ## NOTE: causes strange bugs when run too long (weights concentrate to 1)

        ## do gradient steps, TODO update with genreal belief signature 
        for i in range(nsteps):
            loss = svi.step(params['target1'].belief.samples[:, :, -1])
            # print(global_guide(params['target1'].belief.samples[:, :, -1]))

        pars = global_guide(params['target1'].belief.samples[:, :, -1])
        
        new_p = {}

        ## need to detach for observation_model to be serializable
        for p in ['weights', 'locs', 'scales']:
            new_p[p] = pars[p+str(os.getpid())].detach()

        self.approx_params = new_p


class NoVIPointerObservationModel(ObservationModel):
    def __init__(self):
        # uninitialized parameters
        self.approx_params = {'weights'+str(os.getpid()): None, 'locs'+str(os.getpid()): None, 'scales'+str(os.getpid()): None}
        self.active_planned_observations = {'target1': torch.empty((2,)).detach()}
    
    def is_in_ray(self, a_pose, target):
        ray_width = np.pi / 4  ## has 45-degree field of view on either side

        if target[0] >= 0:
            return np.abs(np.arctan(target[1]/target[0]) - a_pose) <= ray_width
        else:
            return np.abs(np.arctan(target[1]/target[0]) - (a_pose - np.pi)) <= ray_width


    def approx_model(self, data):
        pass

    def get_unnorm_obs_log_likelihood(self, params, prefix_obs, fail_ts):
        log_likelihood = torch.zeros((params['target1'].belief.samples.shape[0], ))

        for idx in range(params['target1'].belief.samples.shape[0]):
            ## initialize log_likelihood to prior probability

            log_likelihood[idx] = params['target1'].belief.dist.log_prob(params['target1'].belief.samples[idx, :, fail_ts]).sum().item()

            ## add in terms for the forward model
            for obs_active_ts in prefix_obs:
                if self.is_in_ray(params['pr2'].pose[0,obs_active_ts[1]-1], params['target1'].belief.samples[idx,:,fail_ts]):
                    ## sample around the true belief, with extremely low variation / error
                    log_likelihood[idx] += dist.MultivariateNormal(params['target1'].belief.samples[idx,:,fail_ts], (0.01 * torch.eye(2))).log_prob(prefix_obs[obs_active_ts]['target1'])
                else:
                    ## sample from prior dist -- have no additional knowledge, don't read it
                    log_likelihood[idx] += dist.MultivariateNormal(torch.zeros((2,)).to(DEVICE), 0.01 * torch.eye(2)).log_prob(prefix_obs[obs_active_ts]['target1'])

        return log_likelihood

    def forward_model(self, params, active_ts, provided_state=None, past_obs={}):        
        if provided_state is not None:
            ## overrides the current belief sample with a true state
            b_global_samp = provided_state['target1'].to(DEVICE)
        else:
            ## sample from current Gaussian mixture model
            b_global_samp = pyro.sample('belief_global', params['target1'].belief.dist).to(DEVICE)
        
        ## sample through strict prefix of current obs
        for obs_active_ts in past_obs:
            if self.is_in_ray(params['pr2'].pose[0,obs_active_ts[1]-1], b_global_samp.detach().to('cpu')):
                ## sample around the true belief, with extremely low variation / error
                pyro.sample('target1.'+str(obs_active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), (0.01 * torch.eye(2)).to(DEVICE)))
            else:
                ## sample from prior dist -- have no additional knowledge, don't read it
                pyro.sample('target1.'+str(obs_active_ts[0]), dist.MultivariateNormal(torch.zeros((2,)).to(DEVICE), 0.01 * torch.eye(2).to(DEVICE)))

        
        ## get sample for current timestep, record and return
        samps = {}

        if self.is_in_ray(params['pr2'].pose[0,active_ts[1]-1], b_global_samp.detach().to('cpu')):
            ## sample around the true belief, with extremely low variation / error
            samps['target1'] = pyro.sample('target1.'+str(active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), 0.01 * torch.eye(2).to(DEVICE)))
        else:
            ## sample from prior dist -- have no additional knowledge, don't read it
            samps['target1'] = pyro.sample('target1.'+str(active_ts[0]), dist.MultivariateNormal(torch.zeros((2,)).to(DEVICE), 0.01 * torch.eye(2).to(DEVICE)))

        # return tensors on CPU for compatib
        for samp in samps:
            samps[samp].to('cpu')

        return samps

    ## no VI in the pointer observation
    def fit_approximation(self, params):        
        pass



class NoVIObstacleObservationModel(ObservationModel):
    def __init__(self):
        # uninitialized parameters
        self.approx_params = {'weights'+str(os.getpid()): None, 'locs'+str(os.getpid()): None, 'scales'+str(os.getpid()): None}
        self.active_planned_observations = {'obs1': torch.empty((2,)).detach()}
    
    def is_in_ray(self, a_pose, target):
        ray_width = np.pi / 4  ## has 45-degree field of view on either side

        if target[0] >= 0:
            return np.abs(np.arctan(target[1]/target[0]) - a_pose) <= ray_width
        else:
            return np.abs(np.arctan(target[1]/target[0]) - (a_pose - np.pi)) <= ray_width


    def approx_model(self, data):
        pass

    def get_unnorm_obs_log_likelihood(self, params, prefix_obs, fail_ts):
        log_likelihood = torch.zeros((params['obs1'].belief.samples.shape[0], ))

        for idx in range(params['obs1'].belief.samples.shape[0]):
            ## initialize log_likelihood to prior probability

            log_likelihood[idx] = params['obs1'].belief.dist.log_prob(params['obs1'].belief.samples[idx, :, fail_ts]).sum().item()

            ## add in terms for the forward model
            for obs_active_ts in prefix_obs:
                if self.is_in_ray(params['pr2'].pose[0,obs_active_ts[1]-1], params['obs1'].belief.samples[idx,:,fail_ts]):
                    ## sample around the true belief, with extremely low variation / error
                    log_likelihood[idx] += dist.MultivariateNormal(params['obs1'].belief.samples[idx,:,fail_ts], (0.01 * torch.eye(2))).log_prob(prefix_obs[obs_active_ts]['obs1'])
                else:
                    ## sample from prior dist -- have no additional knowledge, don't read it
                    log_likelihood[idx] += dist.MultivariateNormal(torch.zeros((2,)).to(DEVICE), 0.01 * torch.eye(2)).log_prob(prefix_obs[obs_active_ts]['obs1'])

        return log_likelihood

    def forward_model(self, params, active_ts, provided_state=None, past_obs={}):        
        if provided_state is not None:
            ## overrides the current belief sample with a true state
            b_global_samp = provided_state['obs1'].to(DEVICE)
        else:
            ## sample from current Gaussian mixture model
            b_global_samp = pyro.sample('belief_global', params['obs1'].belief.dist).to(DEVICE)
        
        ## sample through strict prefix of current obs
        for obs_active_ts in past_obs:
            if self.is_in_ray(params['pr2'].pose[0,obs_active_ts[1]-1], b_global_samp.detach().to('cpu')):
                ## sample around the true belief, with extremely low variation / error
                pyro.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), (0.01 * torch.eye(2)).to(DEVICE)))
            else:
                ## sample from prior dist -- have no additional knowledge, don't read it
                pyro.sample('obs1.'+str(obs_active_ts[0]), dist.MultivariateNormal(torch.zeros((2,)).to(DEVICE), 0.01 * torch.eye(2).to(DEVICE)))

        
        ## get sample for current timestep, record and return
        samps = {}

        if self.is_in_ray(params['pr2'].pose[0,active_ts[1]-1], b_global_samp.detach().to('cpu')):
            ## sample around the true belief, with extremely low variation / error
            samps['obs1'] = pyro.sample('obs1.'+str(active_ts[0]), dist.MultivariateNormal(b_global_samp.float().to(DEVICE), 0.01 * torch.eye(2).to(DEVICE)))
        else:
            ## sample from prior dist -- have no additional knowledge, don't read it
            samps['obs1'] = pyro.sample('obs1.'+str(active_ts[0]), dist.MultivariateNormal(torch.zeros((2,)).to(DEVICE), 0.01 * torch.eye(2).to(DEVICE)))

        # return tensors on CPU for compatib
        for samp in samps:
            samps[samp].to('cpu')

        return samps

    ## no VI in the pointer observation
    def fit_approximation(self, params):        
        pass