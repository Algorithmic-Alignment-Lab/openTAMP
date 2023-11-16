from opentamp.policy_hooks.tamp_agent import TAMPAgent
import opentamp.policy_hooks.utils.policy_solver_utils as utils

from opentamp.policy_hooks.gym_prob import *

class GymAgent(TAMPAgent):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.gym_env = hyperparams['gym_env_type']()
        self.done = False
        self.cur_state = self.gym_env.reset()

    def reset(self, m):
        self.cur_state = self.gym_env.reset()

    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        sample.set(utils.MJC_SENSOR_ENUM, self.cur_state, t)
        sample.set(utils.STATE_ENUM, self.cur_state, t)

    def run_policy_step(self, U_full, cur_state):
        self.cur_state, _, _, _ = self.gym_env.step(U_full)  # left to internal logic
        return True, 0

    def goal_f(self, condition, state, targets=None, cont=False):
        return self.gym_env.assess_goal(condition, state, targets=targets, cont=cont)

    def goal(self, cond, targets=None):
        # TODO read off from problem file, don't use this one
        return None

    def reset_to_state(self, x):
        return self.gym_env.reset_to_state(x)
    
    def get_state(self):
        return self.cur_state

    # TODO look up what this actually does
    # have since removed all instances of this
    def set_symbols(self, plan, task, anum=0, cond=0, targets=None, st=0):
        pass
