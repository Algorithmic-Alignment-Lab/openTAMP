from gym import Env
from gym import spaces
import random
import numpy as np
import torch
import pyro.distributions as distros

from opentamp.policy_hooks.TAMPAgent import *

from opentamp.policy_hooks.utils.policy_solver_utils import *

from opentamp.envs.gym_env_nav_belief import GymEnvNav

class GymEnvNavTheta(GymEnvNav):
    def step(self, action):
        # make single step in direction of target
        # self.curr_state[:2] += action[:2]  # move by action
        reg_val = action[:2] / ACTION_SCALE
        xy_vals = np.array([reg_val[0] * np.cos(reg_val[1]), reg_val[0] * np.sin(reg_val[1])])
        self.curr_state[:2] += xy_vals
        self.curr_state[2] = action[2] # set angle explicitly
        goal_rel_pos = (self.curr_state[3:5] - self.curr_state[:2]) * 1  ## return relative position
        obstacle_rel_pos = (self.curr_state[7:] - self.curr_state[:2]) * 1 
        # obstacle_abs_angle = np.arctan(obstacle_rel_pos[1]/obstacle_rel_pos[0]) if np.abs(obstacle_rel_pos[0]) > 0.001 else (np.pi/2 if obstacle_rel_pos[1]*obstacle_rel_pos[0]>0 else -np.pi/2)
        obstacle_rel_distance = np.linalg.norm(obstacle_rel_pos, ord=2)
        # spot_abs_angle = np.arctan(action[1]/action[0]) if action∂[0] > 0.001 else (np.pi/2 if action[1]>0 else -np.pi/2)
        
        # making formula globally true at all theta (correcting for angle readings behind)
        obstacle_angle = self.compute_angle(obstacle_rel_pos)
        # spot_angle = spot_abs_angle if action[0] >= 0 else (spot_abs_angle + np.pi if -np.pi/2 <= spot_abs_angle < 0 else spot_abs_angle - np.pi)
        
        # relative angle of obstacle with respect to spot camera
        obstacle_rel_angle = obstacle_angle - self.curr_state[2]

        ## rotate the relative pose to be in the frame of the SPOT
        # rot_matrix = np.array([[np.cos(spot_angle),np.sin(spot_angle)],[-np.sin(spot_angle),np.cos(spot_angle)]])
        # obstacle_rel_pos_spot_frame = np.dot(rot_matrix, obstacle_rel_pos)

        # lidar_obs = np.array([8.0] * 8)
        # lidar_list = [(np.arange(-np.pi/4, np.pi/4, np.pi/16)[i], np.arange(-np.pi/4, np.pi/2, np.pi/16)[i+1]) for i in range(8)]
        
        # # formulas only valid on -pi/2 to pi/2
        # for detect_idx, theta_thresh in enumerate(lidar_list):
        #     if theta_thresh[0] <= obstacle_rel_angle < theta_thresh[1] or \
        #         theta_thresh[0] <= obstacle_rel_angle + 2*np.pi < theta_thresh[1] or \
        #         theta_thresh[0] <= obstacle_rel_angle - 2*np.pi < theta_thresh[1]:
        #         lidar_obs[detect_idx] = obstacle_rel_distance

        cam_angle = (self.curr_state[2]+np.pi)%(2*np.pi) - np.pi
        # if np.abs(cam_angle - obstacle_angle) <= np.pi/4 and np.linalg.norm(obstacle_rel_pos) <= 6.0:
        obs_view =  obstacle_rel_pos
        # else:
        #     obs_view = np.array([-10.0, -10.0])

        target_rel_pos = (self.curr_state[3:5] - self.curr_state[:2]) * 1 
        # target_abs_angle = np.arctan(target_rel_pos[1]/target_rel_pos[0]) if np.abs(target_rel_pos[0]) > 0.001 else (np.pi/2 if target_rel_pos[1]*target_rel_pos[0]>0 else -np.pi/2)
        target_angle = self.compute_angle(target_rel_pos)
        # if np.abs(cam_angle - target_angle) <= np.pi/4 and np.linalg.norm(target_rel_pos) <= 6.0:
        targ_view = target_rel_pos
        # else:
        #     targ_view = np.array([-10.0, -10.0])
        target_rel_distance = np.linalg.norm(target_rel_pos, ord=2)

        self.curr_obs = np.concatenate([self.curr_state[1:3], np.array([target_angle]), np.array([target_rel_distance]), np.array([obstacle_angle]), np.array([obstacle_rel_distance])])

        # if too close to object, indicate that the current trajectory violated a safety constraint
        if obstacle_rel_distance <= 1.5:
            self.constraint_viol = True

        # self.curr_obs = np.concatenate((self.curr_obs)) ## add norm of destination as proxy for speed

        # ## TODO: integrate noise in the flag
        # if self.is_in_ray(action, self.belief_true['target1'].detach().numpy()):
        #     ## sample around the true belief, with extremely low variation / error
        #     # noisy_obs = distros.MultivariateNormal(self.belief_true['target1'], 0.01 * torch.eye(2)).sample().numpy()
        #     no_noisy_obs = self.belief_true['target1'].detach().numpy()

        #     if no_noisy_obs[0] < 0.001:
        #         nan_ang = np.pi/2 if no_noisy_obs[1] >= 0.0 else -np.pi/2
        #         self.curr_obs = np.array([nan_ang, 1.0])
        #     else:
        #         self.curr_obs = np.array([np.arctan(no_noisy_obs[1]/no_noisy_obs[0]), 1.0])
        # else:
        #     ## reject this observation, give zero reading
        #     # noisy_obs = distros.MultivariateNormal(torch.zeros((2,)), 0.01 * torch.eye(2)).sample().numpy()
        #     no_noisy_obs = np.zeros((2,))
        #     self.curr_obs = np.array([0.0, 0.0])

        return self.curr_obs, 1.0, False, {}

    

class GymEnvNavWrapper(GymEnvNavTheta):
    def reset_to_state(self, state):
        # self.curr_state[:3] = state[:3]
        # self.curr_state[5:7] = state[5:7]
        self.curr_state = state
        self.curr_obs = np.array([0.0]*6)
        self.constraint_viol = False
        return self.curr_obs

    def get_vector(self):
        state_vector_include = {
            'pr2': ['pose', 'theta'],
            'target1': ['value'],
            'softtarget1': ['value'],
            'obs1': ['value']
        }
        
        action_vector_include = {
            'pr2': ['pose', 'theta']
        }

        target_vector_include = {
            'target': ['pose']
        }
        
        return state_vector_include, action_vector_include, target_vector_include


    # reset without affecting the simulator
    def get_random_init_state(self):
        # # init_pose = random.random() * np.pi/2  # give random initial state between 0 and 90 degrees
        # init_pose = np.array([0.0,0.0,0.0])
        # # init_theta = np.array([random.random() * 2 * np.pi - np.pi]) ## random on -np.pi to np.pi
        # # init_vel = np.array([0.0])

        # is_valid = False
        # while not is_valid:
        #     proposal_targ = self.dist.sample().detach().numpy()

        #     if np.linalg.norm(proposal_targ) <= 4.0:
        #         continue

        #     rand = random.random()

        #     avg_val = torch.tensor(proposal_targ * rand) 

        #     obstacle_dist = distros.Uniform(avg_val - torch.tensor([1.0, 1.0]), 
        #                                     avg_val + torch.tensor([1.0, 1.0]))
        #     proposal_obs = obstacle_dist.sample().detach().numpy()

        #     if np.linalg.norm(proposal_targ-proposal_obs) < 1.5 or np.linalg.norm(proposal_obs) < 1.5 :
        #         continue
                
        #     is_valid = True
        
        # # by default, point at the obstacle at spawn
        # obstacle_abs_angle = np.arctan(proposal_obs[1]/proposal_obs[0]) if np.abs(proposal_obs[0]) > 0.001 else (np.pi/2 if proposal_obs[1]*proposal_obs[0]>0 else -np.pi/2)
        # obstacle_angle = obstacle_abs_angle if proposal_obs[0] >= 0  else (obstacle_abs_angle + np.pi if -np.pi/2 <= obstacle_abs_angle < 0 else obstacle_abs_angle - np.pi)
        # init_pose[2] = obstacle_angle

        rand_side_init = random.randrange(4)
        # rand_side_goal = random.randrange(4)
        # while rand_side_goal == rand_side_init:
        #     rand_side_goal = random.randrange(4)

        init_coords = random.random() * 6 - 3
        # goal_coords = random.random() * 6 - 3

        init_pos = np.array([-1, -1])
        # goal_pos = np.array([-1, -1])

        if rand_side_init == 0:
            init_pos = np.array([-3., init_coords])
        if rand_side_init == 1:
            init_pos = np.array([init_coords, -3.])
        if rand_side_init == 2:
            init_pos = np.array([3., init_coords])
        if rand_side_init == 3:
            init_pos = np.array([init_coords, 3.])

        # if rand_side_goal == 0:
        #     goal_pos = np.array([-3., goal_coords])
        # if rand_side_goal == 1:
        #     goal_pos = np.array([goal_coords, -3.])
        # if rand_side_goal == 2:
        #     goal_pos = np.array([3., goal_coords])
        # if rand_side_goal == 3:
        #     goal_pos = np.array([goal_coords, 3.])

        goal_pos = -init_pos
        soft_goal_pos = init_pos * 2/3
        proposal_obs = goal_pos
        obstacle_abs_angle = np.arctan(proposal_obs[1]/proposal_obs[0]) if np.abs(proposal_obs[0]) > 0.001 else (np.pi/2 if proposal_obs[1]*proposal_obs[0]>0 else -np.pi/2)
        obstacle_angle = obstacle_abs_angle if proposal_obs[0] >= 0  else (obstacle_abs_angle + np.pi if -np.pi/2 <= obstacle_abs_angle < 0 else obstacle_abs_angle - np.pi)

        obs = self.obs_dist.sample().detach().numpy()
        goal_pos = self.target_dist.sample().detach().numpy()

        self.curr_state[7:] = obs
        self.curr_state[3:5] = goal_pos

        ## initalize to center
        return np.concatenate((np.array([0.0, 0.0]), np.array([obstacle_angle]), goal_pos, np.array([8.0,0.]), obs))

    # determine whether or not a given state satisfies a goal condition
    def assess_goal(self, condition, state, targets=None, cont=None):
        angle = self.curr_state[2]
        goal_rel_pose = self.curr_state[3:5] - self.curr_state[:2]
        # if pointing directly at the object

        if np.linalg.norm(goal_rel_pose) <= 1.0:
            return 0.0
        else:
            return 1.0

        # return 0.0 ## always succeeds for now

    # determine whether constraints have been violated since last reset
    def assess_constraint_viol(self):
        if self.constraint_viol:
            return 1.0
        else:
            return 0.0
