from gym import Env
from gym import spaces
import random
import numpy as np
import torch
import pyro.distributions as distros

import opentamp
import json

from opentamp.policy_hooks.utils.policy_solver_utils import *
import sys
sys.path.append('./drake_gym_sim/src') #need to run from opentamp/ directory
print(sys.path)


from drake_gym_sim.imtGym.imtCommon import RobotState, RobotAction, RobotActionReturnParameter, RobotActionParameter, OBJECT_NAME_BANANA
from drake_gym_sim.imtGym.imtSim import ImtSim
from drake_gym_sim.imtGym import table_specs
from drake_gym_sim.imtGym.hardcoded_cameras import (
    get_base_positions_for_hardcoded_cameras,
    get_cam_poses_nested_array,
)
from drake_gym_sim.imtGym.move_object import RepeatingTimer, move_box
from os import system, name, path
from pydrake.all import Meshcat, MeshcatParams

import logging, logging.handlers

prob = opentamp.__path__._path[0] + '/new_specs/drake_nav_domain_deterministic/namo_purenav_prob.json'



_sim = None

"""def get_vector(filename):
    with open(filename) as f:
        d = json.load(f)

    init_objs = d["init_objs"]
    
    target_pose = []
    robot_pose = []
    obstacle_pose = []
    for init_obj in init_objs:
        print(init_obj)
        if init_obj['type'] == "Target":
            target_pose = init_obj["value"]

        elif init_obj['type'] == "Robot":
            robot_pose = init_obj["pose"]
        elif init_obj['type'] == "Obstacle":
            obstacle_pose = init_obj["value"]

    return target_pose, robot_pose, obstacle_pose"""


def drake_start_sim():
        # set host to 0.0.0.0 to enable port forwarding on remote server

        meshcat = Meshcat(MeshcatParams(host="0.0.0.0"))

        rng = np.random.default_rng(145)  # this is for python

        logging.root.setLevel(logging.INFO)
        # logging.root.setLevel(logging.DEBUG)  # Uncomment to enable debug logging.

        fsm_logger = logging.getLogger("imt.fsm")
        fsm_logger.setLevel(logging.INFO)

        bs_logger = logging.getLogger("imt.perception")
        bs_logger.setLevel(logging.INFO)

        simulation_time = -1
        # simulation_time = 1
        add_debug_logger = True
        add_fixed_cameras = False
        use_teleop = False
        plot_camera_input = False
        #target_pose, robot_pose, obstacle_pose = get_vector(prob)
        robot_pose = [0,0]
        obstacle_pose = [1, 1]
        target_pose = [3, 3]
        print(f"{target_pose=}, {robot_pose=}, {obstacle_pose=}")
        #obstacle_pose = [1, 1, 0]
        global _sim
        _sim = ImtSim()
        _sim.create_and_run_simulation(meshcat,
            rng,
            number_people=1,
            add_debug_logger=add_debug_logger,
            simulation_time=simulation_time,
            starting_position = np.array([robot_pose[0], robot_pose[1], -1.57]),
            add_fixed_cameras=add_fixed_cameras,
            obstacle_position= np.array([obstacle_pose[0], obstacle_pose[1], 0]), 
            target_position= np.array([target_pose[0], target_pose[1], 0]),
            use_teleop=use_teleop,
            plot_camera_input=plot_camera_input,
            table_specs=table_specs.mix_rooms,
            use_naive_fsm=True,
            use_dummy_spotter=True)

        _sim.start_simulation()

drake_start_sim()


class DrakeGymEnvNav(Env): 
       
    def __init__(self):
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype='float32')
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype='float32')
        self.curr_state = np.array([0.0]*6)
        self.curr_obs = np.array([0.0]*10)
        self.dist = self.assemble_dist()
        self.belief_true = {}
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
        dist = distros.Uniform(torch.tensor([-3.0, -3.0]), torch.tensor([3.0, 3.0]))
        return dist

    def step(self, action):
        # make single step in direction of target
        self.curr_state[:2] += action  # move by action
        goal_rel_pos = (self.curr_state[2:4] - self.curr_state[:2]) * 1  ## return relative position
        obstacle_rel_pos = (self.curr_state[4:] - self.curr_state[:2]) * 1 
        obstacle_abs_angle = np.arctan(obstacle_rel_pos[1]/obstacle_rel_pos[0]) if np.abs(obstacle_rel_pos[0]) > 0.001 else (np.pi/2 if obstacle_rel_pos[1]*obstacle_rel_pos[0]>0 else -np.pi/2)
        obstacle_rel_distance = np.linalg.norm(obstacle_rel_pos, ord=2)
        # spot_abs_angle = np.arctan(action[1]/action[0]) if action[0] > 0.001 else (np.pi/2 if action[1]>0 else -np.pi/2)
        
        # making formula globally true at all theta (correcting for angle readings behind)
        obstacle_angle = obstacle_abs_angle if obstacle_rel_pos[0] >= 0  else (obstacle_abs_angle + np.pi if -np.pi/2 <= obstacle_abs_angle < 0 else obstacle_abs_angle - np.pi)
        # spot_angle = spot_abs_angle if action[0] >= 0 else (spot_abs_angle + np.pi if -np.pi/2 <= spot_abs_angle < 0 else spot_abs_angle - np.pi)
        
        # relative angle of obstacle with respect to spot,
        # obstacle_rel_angle = obstacle_angle - spot_angle

        ## rotate the relative pose to be in the frame of the SPOT
        # rot_matrix = np.array([[np.cos(spot_angle),np.sin(spot_angle)],[-np.sin(spot_angle),np.cos(spot_angle)]])
        # obstacle_rel_pos_spot_frame = np.dot(rot_matrix, obstacle_rel_pos)

        lidar_obs = np.array([8.0] * 8)
        lidar_list = [(np.arange(-np.pi, np.pi, np.pi/4)[i], np.arange(-np.pi, 2 * np.pi, np.pi/4)[i+1]) for i in range(8)]
        
        # formulas only valid on -pi/2 to pi/2
        for detect_idx, theta_thresh in enumerate(lidar_list):
            if theta_thresh[0] <= obstacle_angle < theta_thresh[1]:
                lidar_obs[detect_idx] = obstacle_rel_distance

        self.curr_obs = np.concatenate([goal_rel_pos, lidar_obs])

        # if too close to object, indicate that the current trajectory violated a safety constraint
        if obstacle_rel_distance <= 0.5:
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
        self.curr_state = np.array([0.0]*6)
        self.curr_obs = np.array([0.0]*10)
        self.constraint_viol = False
        return self.curr_obs
    

    ## NOTE: only rgb_array mode supported, ignores keyword
    def render(self, mode='rgb_array'):
        print("+++++++++++++++++++ DrakeGymEnvNav.render()")
        def is_in_ray_vectorized(a_pose, x_coord, y_coord, ray_ang):
            return np.where(x_coord > 0, 
                            np.abs(np.arctan(y_coord/x_coord) - a_pose) <= ray_ang % 2*np.pi,
                            np.abs(np.arctan(y_coord/x_coord) - (a_pose - np.pi)) <= ray_ang % np.pi)
            
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
        color_arr = np.where(is_close_to_obj_vectorized(self.curr_state[:2], x_coords, y_coords),
                                            red, 
                                            white)

        # ## coloring in precise ray within pointer
        # color_arr = np.where(is_in_ray_vectorized(self.curr_state, x_coords, y_coords, 0.1),
        #                                     green+red, 
        #                                     color_arr)

        ## coloring in the obstacle
        color_arr = np.where(is_close_to_obj_vectorized(self.curr_state[4:], x_coords, y_coords),
                                    blue, 
                                    color_arr)
        
        ## coloring in the target
        color_arr = np.where(is_close_to_obj_vectorized(self.curr_state[2:4], x_coords, y_coords),
                                    green, 
                                    color_arr)
        
        print(f"{self.curr_state=}")
        self.drake_spot_move(self.curr_state[:2][0], self.curr_state[:2][1])

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




    #######################FROM DRAKE GYM SIM MAIN FUNCTIONS #######################
    
    
    def drake_spot_move(self, x, y):
        action = RobotAction.MOVE
        para = RobotActionParameter()
        para.move_position = np.array([x, y, 0])
        #pos_idx = random.randint(0, len(base_pos)-1)
        #para.move_position = base_pos[pos_idx]
        #para.move_position = base_pos[8]
        ret : RobotActionReturnParameter = _sim.do_action(action, para)
        print(f"---------------------------------------------------------------")
        print(f"---------------------------------------------------------------")
        print(f"----------move-action: done, {ret=}----------------------------")
        print(f"---------------------------------------------------------------")
        print(f"---------------------------------------------------------------")




class DrakeGymEnvNavWrapper(DrakeGymEnvNav):

    def reset_to_state(self, state):
        self.curr_state = state
        self.curr_obs = np.array([0.0]*10)
        self.constraint_viol = False
        return self.curr_obs

    def get_vector(self):
        state_vector_include = {
            'pr2': ['pose'],
            'target1': ['value'],
            'obs1': ['value']
        }
        
        action_vector_include = {
            'pr2': ['pose']
        }

        target_vector_include = {
            'target': ['pose']
        }
        
        return state_vector_include, action_vector_include, target_vector_include


    def render(self, mode='rgb_array'):
        super(DrakeGymEnvNavWrapper, self).render(mode)
        print("!!!!!!!!!!!!!!!!!!! DrakeGymEnvNavWrapper.render()")

    # reset without affecting the simulator
    def get_random_init_state(self):
        # init_pose = random.random() * np.pi/2  # give random initial state between 0 and 90 degrees
        init_pose = np.array([0.0,0.0])
        # init_theta = np.array([random.random() * 2 * np.pi - np.pi]) ## random on -np.pi to np.pi
        # init_vel = np.array([0.0])

        is_valid = False
        while not is_valid:
            proposal_targ = self.dist.sample().detach().numpy()

            if np.linalg.norm(proposal_targ) <= 4.0:
                continue

            rand = random.random()

            avg_val = torch.tensor(proposal_targ * rand) 

            obstacle_dist = distros.Uniform(avg_val - torch.tensor([1.0, 1.0]), 
                                            avg_val + torch.tensor([1.0, 1.0]))
            proposal_obs = obstacle_dist.sample().detach().numpy()

            if np.linalg.norm(proposal_targ-proposal_obs) < 1.5 or np.linalg.norm(proposal_obs) < 1.5 :
                continue
                
            is_valid = True
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!GET_RANDOM_INIT_STATE: {init_pose=},{proposal_targ=}, {proposal_obs=}")
        return np.concatenate((init_pose,proposal_targ, proposal_obs))

    # determine whether or not a given state satisfies a goal condition
    def assess_goal(self, condition, state, targets=None, cont=None):
        item_loc = self.curr_state[2:4]
        pose = self.curr_state[:2]
        # if pointing directly at the object

        if np.linalg.norm(item_loc - pose, ord=2) <= 0.5:
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
