from gym import Env
from gym import spaces
import random
import numpy as np
import torch
import pyro.distributions as distros

from opentamp.policy_hooks.utils.policy_solver_utils import *

from opentamp.envs.blank_gym_env import BlankEnv, BlankEnvWrapper

class BlankEnvBimodal(BlankEnv):
    def __init__(self):
        super().__init__()

    def assemble_dist(self):
        weights = torch.tensor([0.6,0.4])
        locs = torch.tensor([[3., 3.],
                             [3., -3.]])
        scales = torch.tensor([0.5, 0.5])
        cat_dist = distros.Categorical(probs=weights)
        stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(2, 1, 1))
        stack_scale = torch.tile(scales.unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
        cov_tensor = stack_eye * stack_scale
        batched_multivar = distros.MultivariateNormal(loc=locs, covariance_matrix=cov_tensor)
        dist =  distros.MixtureSameFamily(cat_dist, batched_multivar)
        return dist
    

class BlankEnvWrapperBimodal(BlankEnvBimodal):
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
        init_pose = random.random() * np.pi/2  # give random initial state between 0 and 90 degrees
        return np.array([init_pose])

    # determine whether or not a given state satisfies a goal condition
    def assess_goal(self, condition, state, targets=None, cont=None):
        item_loc = self.belief_true['target1']
        # if pointing directly at the object
        if np.abs(np.arctan(item_loc[1]/item_loc[0]) - state) <= 0.1:
            return 0.0
        else:
            return 1.0

        # return 0.0 ## always succeeds for now

    def assess_constraint_viol(self):
        return 0.0
