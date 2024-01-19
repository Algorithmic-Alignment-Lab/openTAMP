from gym import Env
from gym import spaces
import random
import numpy as np
import torch
import pyro.distributions as distros

from opentamp.policy_hooks.utils.policy_solver_utils import *

class GymEnvNav(Env):    
    def __init__(self):
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype='float32')
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype='float32')
        self.curr_state = np.array([0.0]*2)
        self.curr_obs = np.array([0.0]*10)
        self.dist = self.assemble_dist()
        self.belief_true = {'obs1': torch.tensor([0.0, 0.0])}
        self.constraint_viol = False

    def assemble_dist(self):
        # weights = torch.tensor([0.6,0.4])
        # locs = torch.tensor([[3., 3.],
        #                      [3., -3.]])
        # scales = torch.tensor([0.5, 0.5])
        # cat_dist = distros.Categorical(probs=weights)
        # stack_eye = torch.tile(torch.eye().unsqueeze(dim=0), dims=(2, 1, 1))
        # stack_scale = torch.tile(scales.unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
        # cov_tensor = stack_eye * stack_scale
        # batched_multivar = distros.MultivariateNormal(loc=locs, covariance_matrix=cov_tensor)
        dist = distros.Uniform(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))
        return dist

    def step(self, action):
        # make single step in direction of target
        self.curr_state += action  # move by action
        goal_rel_pos = (np.array([3.0, 3.0]) - self.curr_state) * 1  ## return relative position
        obstacle_rel_pos = (self.belief_true['obs1'].detach().numpy() - self.curr_state) * 1 
        obstacle_abs_angle = np.arctan(obstacle_rel_pos[1]/obstacle_rel_pos[0]) if obstacle_rel_pos[0] > 0.001 else (np.pi/2 if obstacle_rel_pos[1]>0 else -np.pi/2)
        obstacle_rel_distance = np.linalg.norm(obstacle_rel_pos, ord=2)
        spot_abs_angle = np.arctan(action[1]/action[0]) if action[0] > 0.001 else (np.pi/2 if action[1]>0 else -np.pi/2)
        
        # making formula globally true at all theta (correcting for angle readings behind)
        obstacle_angle = obstacle_abs_angle if obstacle_rel_pos[0] >= 0  else (obstacle_abs_angle + np.pi if -np.pi/2 <= obstacle_abs_angle < 0 else obstacle_abs_angle - np.pi)
        spot_angle = spot_abs_angle if action[0] >= 0 else (spot_abs_angle + np.pi if -np.pi/2 <= spot_abs_angle < 0 else spot_abs_angle - np.pi)
        
        # relative angle of obstacle with respect to spot,
        obstacle_rel_angle = obstacle_angle - spot_angle

        ## rotate the relative pose to be in the frame of the SPOT
        # rot_matrix = np.array([[np.cos(spot_angle),np.sin(spot_angle)],[-np.sin(spot_angle),np.cos(spot_angle)]])
        # obstacle_rel_pos_spot_frame = np.dot(rot_matrix, obstacle_rel_pos)

        lidar_obs = np.array([8.0] * 8)
        lidar_list = [(np.arange(-np.pi, np.pi, np.pi/4)[i], np.arange(-np.pi, 2 * np.pi, np.pi/4)[i+1]) for i in range(8)]
        
        # formulas only valid on -pi/2 to pi/2
        for detect_idx, theta_thresh in enumerate(lidar_list):
            if theta_thresh[0] <= obstacle_rel_angle < theta_thresh[1]:
                lidar_obs[detect_idx] = obstacle_rel_distance

        self.curr_obs = np.concatenate([goal_rel_pos, lidar_obs])

        # if too close to object, indicate that the current trajectory violated a safety constraint
        if obstacle_rel_distance <= 1.0:
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

    def reset(self):
        self.curr_state = np.array([-3.0, -3.0])
        self.curr_obs = np.array([0.0]*10)
        self.constraint_viol = False
        return self.curr_obs
    

    ## NOTE: only rgb_array mode supported, ignores keyword
    def render(self, mode='rgb_array'):
        def is_in_ray_vectorized(a_pose, x_coord, y_coord, ray_ang):
            return np.where(x_coord > 0, 
                            np.abs(np.arctan(y_coord/x_coord) - a_pose) <= ray_ang,
                            np.abs(np.arctan(y_coord/x_coord) - (a_pose - np.pi)) <= ray_ang)
            
        def is_close_to_obj_vectorized(true_loc, x_coord, y_coord):
            return np.linalg.norm(np.stack((x_coord, y_coord)) - np.tile(true_loc.reshape(-1, 1, 1, 1), (1, 256, 256, 3)), axis=0) <= 0.2

        color_arr = np.ones((256, 256, 3), dtype=np.uint8) * 255
        
        ## initializing vectorized x_coord and y_coord arrays
        x_coords = np.stack([np.tile(np.arange(-5, 5, 5./128.).reshape(-1, 1), (1, 256))]*3, axis=2)
        y_coords = np.stack([x_coords[:,:,0].copy().T]*3, axis=2)

        red = np.stack((np.ones((256, 256), dtype=np.uint8)*255, np.zeros((256, 256), dtype=np.uint8), np.zeros((256, 256), dtype=np.uint8)), axis=2)
        green = np.stack((np.zeros((256, 256), dtype=np.uint8), np.ones((256, 256), dtype=np.uint8)*255, np.zeros((256, 256), np.uint8)), axis=2)
        blue = np.stack((np.zeros((256, 256), np.uint8), np.zeros((256, 256), np.uint8), np.ones((256, 256), dtype=np.uint8)*255), axis=2)
        white = np.ones((256, 256, 3), dtype=np.uint8) * 255

        ## coloring in the robot location
        color_arr = np.where(is_close_to_obj_vectorized(self.curr_state, x_coords, y_coords),
                                            red, 
                                            white)

        # ## coloring in precise ray within pointer
        # color_arr = np.where(is_in_ray_vectorized(self.curr_state, x_coords, y_coords, 0.1),
        #                                     green+red, 
        #                                     color_arr)

        ## coloring in the obstacle
        color_arr = np.where(is_close_to_obj_vectorized(self.belief_true['obs1'], x_coords, y_coords),
                                    blue, 
                                    color_arr)
        
        ## coloring in the target
        color_arr = np.where(is_close_to_obj_vectorized(np.array([3.0, 3.0]), x_coords, y_coords),
                                    green, 
                                    color_arr)

        return color_arr
    
    def postproc_im(self, base_im, s, t, cam_id):
        # def is_in_ray_vectorized(a_pose, x_coord, y_coord, ray_ang):
        #     return np.where(x_coord > 0, 
        #         np.abs(np.arctan(y_coord/x_coord) - a_pose) <= ray_ang,
        #         np.abs(np.arctan(y_coord/x_coord) - (a_pose - np.pi)) <= ray_ang)

        # def is_close_to_obj_vectorized(true_loc, x_coord, y_coord):
        #     return np.linalg.norm(np.stack((x_coord, y_coord)) - np.tile(true_loc.reshape(-1, 1, 1), (1, 256, 256)), axis=0) <= 0.2

        # def in_corner(x_coord, y_coord):
        #     return np.logical_and(x_coord <= -3.0, y_coord <= -3.0)


        # im = base_im.copy()

        # ## initializing vectorized x_coord and y_coord arrays
        # x_coords = np.tile(np.arange(-5, 5, 5./128.).reshape(-1, 1), (1, 256))
        # y_coords = x_coords.copy().T

        # ## adding the current target location as an observation for rendering
        # im[:,:,0] = np.where(is_in_ray_vectorized(s.get(ANG_ENUM)[t,:], x_coords, y_coords, 0.01),
        #                                     np.zeros((256, 256), dtype=np.uint8), 
        #                                     im[:,:,0])
                
        # im[:,:,1] = np.where(is_in_ray_vectorized(s.get(ANG_ENUM)[t,:], x_coords, y_coords, 0.01),
        #                             np.ones((256, 256), dtype=np.uint8) * 255, 
        #                             im[:,:,1])
        
        # im[:,:,2] = np.where(is_in_ray_vectorized(s.get(ANG_ENUM)[t,:], x_coords, y_coords, 0.01),
        #                             np.zeros((256, 256), dtype=np.uint8), 
        #                             im[:,:,2])

        # ## adding in preliminary task stuff as an enum
        # im[:,:,0] = np.where(in_corner(x_coords, y_coords),
        #                                 np.ones((256, 256), dtype=np.uint8)*255 if s.task[0]==0 else np.zeros((256, 256), dtype=np.uint8), 
        #                                 im[:,:,0])
            
        # im[:,:,1] = np.where(in_corner(x_coords, y_coords),
        #                                 np.ones((256, 256), dtype=np.uint8)*255 if s.task[0]==1 else np.zeros((256, 256), dtype=np.uint8), 
        #                                 im[:,:,1])
        
        # im[:,:,2] = np.where(in_corner(x_coords, y_coords),
        #                                 np.ones((256, 256), dtype=np.uint8)*255 if s.task[0]==2 else np.zeros((256, 256), dtype=np.uint8), 
        #                                 im[:,:,2])

        return base_im


    def is_in_ray(self, a_pose, target):
        return np.abs(np.arctan(target[1]/target[0]) - a_pose) <= np.pi / 4

    ## get random sample to initialize uncertain problem
    def sample_belief_true(self):
        return {'obs1': self.dist.sample()}
        # return {'obs1': torch.tensor([0.0, 0.0])}
        # rand = random.random() * 8
        # if rand < 1.0:
        #     return {'target1': torch.tensor([3.0, 3.0])}
        # elif rand < 2.0:
        #     return {'target1': torch.tensor([3.0, -3.0])}
        # elif rand < 3.0:
        #     return {'target1': torch.tensor([-3.0, 3.0])}
        # elif rand < 4.0:
        #     return {'target1': torch.tensor([-3.0, -3.0])}
        # elif rand < 5.0:
        #     return {'target1': torch.tensor([4.2426, 0])}
        # elif rand < 6.0:
        #     return {'target1': torch.tensor([0, 4.2426])}
        # elif rand < 7.0:
        #     return {'target1': torch.tensor([-4.2426, 0])}
        # else:
        #     return {'target1': torch.tensor([0, -4.2426])}

    def set_belief_true(self, belief_dict):
        self.belief_true = belief_dict
    

class GymEnvNavWrapper(GymEnvNav):
    def reset_to_state(self, state):
        self.curr_state = state
        self.curr_obs = np.array([0.0]*10)
        self.constraint_viol = False
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
        # init_pose = random.random() * np.pi/2  # give random initial state between 0 and 90 degrees
        return np.array([-3.0, -3.0]) ## targets are randomized, initial pose fixed

    # determine whether or not a given state satisfies a goal condition
    def assess_goal(self, condition, state, targets=None, cont=None):
        item_loc = np.array([3.0, 3.0])
        # if pointing directly at the object

        if np.linalg.norm(item_loc - state, ord=2) <= 0.4:
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
