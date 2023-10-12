from gym import spaces
from gym.core import Env

class BlankClass(Env):
    def __init__(self):
        self.action_space = spaces.Box(0.0, 1.0)
        self.observation_space = spaces.Box(0.0, 1.0)
        self.reward_range = spaces.Box(0.0, 1.0)

    def step(self):
        return self.observation_space.sample(), self.reward_range.sample(), False, {}

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass
