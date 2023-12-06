from gym import Env
from gym import spaces
import random
import numpy as np
import torch
import pyro.distributions as distros


class BlankEnv(Env):    
    def __init__(self):
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype='float32')
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype='float32')
        self.curr_state = np.array([0.0]*1)
        self.curr_obs = np.array([0.0]*2)
        self.dist = self.assemble_dist()

    def assemble_dist(self):
        weights = torch.tensor([0.5,0.5])
        locs = torch.tensor([[3., 3.],
                             [3., 3.]])
        scales = torch.tensor([0.5, 0.5])
        cat_dist = distros.Categorical(probs=weights)
        stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(2, 1, 1))
        stack_scale = torch.tile(scales.unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
        cov_tensor = stack_eye * stack_scale
        batched_multivar = distros.MultivariateNormal(loc=locs, covariance_matrix=cov_tensor)
        dist =  distros.MixtureSameFamily(cat_dist, batched_multivar)
        return dist

    def step(self, action):
        self.curr_state = action

        if self.is_in_ray(action, self.belief_true['target1'].detach().numpy()):
            ## sample around the true belief, with extremely low variation / error
            self.curr_obs = distros.MultivariateNormal(self.belief_true['target1'], 0.01 * torch.eye(2)).sample().numpy()
        else:
            ## reject this observation, give zero reading
            self.curr_obs = distros.MultivariateNormal(torch.zeros((2,)), 0.01 * torch.eye(2)).sample().numpy()

        return self.curr_obs, 1.0, False, {}

    def reset(self):
        self.curr_state = np.array([0.0]*1)
        self.curr_obs = np.array([0.0]*2)
        return self.curr_obs
    
    def render(self, mode='rgb_array'):
        pass

    def is_in_ray(self, a_pose, target):
        return np.abs(np.arctan(target[1]/target[0]) - a_pose) <= np.pi / 4

    ## get random sample to initialize uncertain problem
    def resample_belief_true(self):
        self.belief_true = {'target1': self.dist.sample()}

        # self.belief_true = {'target1': torch.tensor([3., 3.])}

    

class BlankEnvWrapper(BlankEnv):
    def reset_to_state(self, state):
        self.curr_state = state
        self.curr_obs = np.array([0.0]*2)
        return self.curr_obs

    def get_vector(self):
        state_vector_include = {
            'pr2': ['pose']
        }
        
        action_vector_include = {
            'pr2': ['pose']
        }

        target_vector_include = {
            'target': ['pose']
        }
        
        return state_vector_include, action_vector_include, target_vector_include


    # reset without affecting the simulator
    def get_random_init_state(self):
        return np.array([0.0]*1)

    # determine whether or not a given state satisfies a goal condition
    def assess_goal(self, condition, state, targets=None, cont=None):
        return 0.0 ## always succeeds for now
