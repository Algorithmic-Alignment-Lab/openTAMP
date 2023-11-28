from gym import Env
from gym import spaces
import random
import numpy as np

class BlankEnv(Env):    
    def __init__(self):
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype='float32')
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype='float32')
        # self.belief_global_inds = np.array([5, 6])

    def step(self, action):
        return np.array([0.0]*1), 1.0, False, {}

    def reset(self):
        return np.array([0.0]*1)
    
    def render(self, mode='rgb_array'):
        pass


class BlankEnvWrapper(BlankEnv):
    def reset_to_state(self, state):
        pass

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
        r = random.random()
        return 1.0 if r < 0.3 else 0.0
