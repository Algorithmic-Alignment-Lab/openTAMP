from opentamp.policy_hooks.tamp_agent import TAMPAgent
import opentamp.policy_hooks.utils.policy_solver_utils as utils


import numpy as np
# from opentamp.policy_hooks.gym_prob import *

class GymAgent(TAMPAgent):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.gym_env = hyperparams['gym_env_type']()
        self.done = False
        self.curr_obs = self.gym_env.reset()
        self.curr_state = self.gym_env.curr_state
        self.num_tasks = 0
        self.curr_targ = np.array([0.])
        self.past_targ = np.array([0.])
        self.past_point = 0
        self.past_val = 0
        self.past_task = -1.0

    def reset(self, m):
        self.curr_obs = self.gym_env.reset()
        self.curr_state = self.gym_env.curr_state

    ## subclass as needed for different kinds of sample populations (e.g. RNN stuff)
    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        sample.set(utils.MJC_SENSOR_ENUM, self.curr_obs, t)
        sample.set(utils.STATE_ENUM, self.curr_state, t)
        sample.set(utils.PAST_COUNT_ENUM, np.array([self.num_tasks]), t)
        sample.set(utils.ANG_ENUM, self.curr_targ, t=t)
        sample.set(utils.PAST_ANG_ENUM, self.past_targ, t=t)
        sample.set(utils.PAST_POINT_ENUM, np.array([self.past_point]), t=t)
        sample.set(utils.PAST_VAL_ENUM, np.array([self.past_val]), t=t)
        sample.set(utils.PAST_TASK_ENUM, np.array([self.past_task]), t=t)

    def run_policy_step(self, U_full, curr_state):
        self.curr_obs, _, _, _ = self.gym_env.step(U_full)  # left to internal logic
        self.curr_state = self.gym_env.curr_state
        return True, 0

    def goal_f(self, condition, state, targets=None, cont=False):
        return self.gym_env.assess_goal(condition, state, targets=targets, cont=cont)

    def goal(self, cond, targets=None):
        # TODO read off from problem file, don't use this one
        return None

    def reset_to_state(self, x):
        self.curr_obs = self.gym_env.reset_to_state(x)
        self.curr_state = self.gym_env.curr_state

    def reset(self, m):
        self.curr_obs = self.gym_env.reset()
        self.curr_state = self.gym_env.curr_state
    
    def get_state(self):
        return self.curr_state

    ## TODO look up what this actually does
    ## NOTE have since removed all instances of this
    def set_symbols(self, plan, task, anum=0, cond=0, targets=None, st=0):
        pass
    
    ## annotates image at the start of a task
    def get_annotated_image(self, s, t, cam_id=None):
        if cam_id is None: cam_id = self.camera_id
        x = s.get_X(t=t)
        # task = s.get(FACTOREDTASK_ENUM, t=t)
        # u = s.get(ACTION_ENUM, t=t)
        u = str(u.round(2))[1:-1]
        # pos = s.get(END_POSE_ENUM, t=t)
        # pos = str(pos.round(2))[1:-1]
        # textover1 = self.mjc_env.get_text_overlay(body='Task: {0}'.format(task))
        # textover2 = self.mjc_env.get_text_overlay(body='{0}; {1}'.format(u, pos), position='bottom left')
        self.reset_to_state(x)
        # im = self.mjc_env.render(camera_id=cam_id, height=self.image_height, width=self.image_width, view=False, overlays=(textover1, textover2))
        im = self.gym_env.render()
        return im

    ## gets the image for this agent
    def get_image(self, x, depth=False, cam_id=None):
        self.reset_to_state(x)
        if cam_id is None: cam_id = self.camera_id
        im = self.gym_env.render()
        return im
    
    def get_annotated_image(self, s, t, cam_id=None):
        base_im = self.get_image(s.get_X()[t,:], depth=False, cam_id=cam_id)
        im = self.gym_env.postproc_im(base_im, s, t, cam_id)
        return im

    ## TODO SHUNT THESE INTO API, builds hist_info samples
    def store_hist_info(self, hist_info):
        self.num_tasks = hist_info[0] 
        self.past_targ = hist_info[1]
        self.past_point = hist_info[2]
        self.past_val = hist_info[3]
        self.past_task = hist_info[4]

    def store_aux_info(self, aux_info):
        self.curr_targ = aux_info

    def get_inv_cov(self):
        return np.eye(self.dU)