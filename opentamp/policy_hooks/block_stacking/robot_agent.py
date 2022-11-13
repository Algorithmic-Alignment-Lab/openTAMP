import copy
import ctypes
import pickle as pickle
import sys
from threading import Thread
import time
from tkinter import TclError
import traceback
import xml.etree.ElementTree as xml

import matplotlib
try:
    matplotlib.use("TkAgg")
except:
    pass
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import scipy.interpolate
from scipy.spatial.transform import Rotation

from dm_control import mujoco
from dm_control.mujoco import TextOverlay
import pybullet as P
import robodesk

import opentamp.pma.backtrack_ll_solver_OSQP as bt_ll
import opentamp.core.util_classes.common_constants as const
import opentamp.core.util_classes.items as items
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
from opentamp.core.util_classes.viewer import OpenRAVEViewer
import opentamp.core.util_classes.transform_utils as T
from opentamp.policy_hooks.sample import Sample
from opentamp.policy_hooks.utils.policy_solver_utils import *
import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.policy_hooks.tamp_agent import TAMPAgent


PANDA_XML = opentamp.__path__._path[0] + "/robot_info/robodesk/franka_panda.xml"
HEADER_XML = opentamp.__path__._path[0] + "/robot_info/robodesk/franka_panda_headers.xml"

const.NEAR_GRIP_COEFF = 4e-2 # 2.2e-2 # 1.8e-2 # 2e-2
const.NEAR_GRIP_ROT_COEFF = 7e-3
const.NEAR_APPROACH_COEFF = 7e-3 # 8e-3
const.NEAR_RETREAT_COEFF = 8e-3 # 1.2e-2
const.NEAR_APPROACH_ROT_COEFF = 1e-3
const.GRASP_DIST = 0.12 # 0.13
const.PLACE_DIST = 0.12 # 0.13
const.APPROACH_DIST = 0.01 # 0.01
const.RETREAT_DIST = 0.01 # 0.01
const.QUICK_APPROACH_DIST = 0.015 # 0.02
const.QUICK_RETREAT_DIST = 0.015 # 0.02
const.EEREACHABLE_COEFF = 3e-1 # 2e-1 # 9e-2 # 1e-1 # 3e-2 # 2e-2
const.WEAK_EEREACHABLE_COEFF = 1e-2
const.EEREACHABLE_ROT_COEFF = 1e-2 # 8e-3
const.EEREACHABLE_STEPS = 5
const.EEATXY_COEFF = 5e-2 # 8e-2
const.RCOLLIDES_COEFF = 2e-2 # 2e-2
const.OBSTRUCTS_COEFF = 2.5e-2

bt_ll.INIT_TRAJ_COEFF = 3e-1
bt_ll.TRAJOPT_COEFF = 1e3
bt_ll.RS_COEFF = 1e2
# bt_ll.RESAMPLE_ALL = True
# bt_ll.INIT_TRUST_REGION_SIZE = 1e-1 # 1e-2
# bt_ll.MAX_MERIT_INCR = 100
bt_ll.OSQP_MAX_ITER = int(4e03)
bt_ll.INIT_TRUST_REGION_SIZE = 1e1


STACK_OFFSET = 0.08
SHELF_Y = 0.86 # 0.87

STEP = 0.1
NEAR_TOL = 0.05
LOCAL_NEAR_TOL = 0.12 # 0.3

def theta_error(cur_quat, next_quat):
    sign1 = np.sign(cur_quat[np.argmax(np.abs(cur_quat))])
    sign2 = np.sign(next_quat[np.argmax(np.abs(next_quat))])
    next_quat = np.array(next_quat)
    cur_quat = np.array(cur_quat)
    angle = -(sign1 * sign2) * robo_T.get_orientation_error(sign1 * next_quat, sign2 * cur_quat)
    return angle

class optimal_pol:
    def __init__(self, dU, action_inds, state_inds, opt_traj):
        self.dU = dU
        self.action_inds = action_inds
        self.state_inds = state_inds
        self.opt_traj = opt_traj

    def act(self, X, O, t, noise=None):
        u = np.zeros(self.dU)
        if t < len(self.opt_traj):
            for param, attr in self.action_inds:
                cur_val = X[self.state_inds[param, attr]] if (param, attr) in self.state_inds else None
                if attr.find('grip') >= 0:
                    #val = self.opt_traj[min(t+1, len(self.opt_traj)-1), self.state_inds[param, attr]][0]
                    val = self.opt_traj[min(t, len(self.opt_traj)-1), self.state_inds[param, attr]][0]
                    val = 0.045 if val > 0.01 else -0.005
                    u[self.action_inds[param, attr]] = val 
                elif attr.find('ee_pos') >= 0:
                    cur_ee = cur_val if cur_val is not None else self.opt_traj[t, self.state_inds[param, attr]]
                    next_ee = self.opt_traj[t, self.state_inds[param, attr]]
                    u[self.action_inds[param, attr]] = next_ee - cur_ee
                elif attr.find('ee_rot') >= 0:
                    cur_ee = cur_val
                    cur_quat = np.array(T.euler_to_quaternion(cur_ee, 'xyzw'))
                    next_ee = self.opt_traj[t, self.state_inds[param, attr]]
                    next_quat = np.array(T.euler_to_quaternion(next_ee, 'xyzw'))
                    currot = Rotation.from_quat(cur_quat)
                    targrot = Rotation.from_quat(next_quat)
                    act = (targrot * currot.inv()).as_rotvec()
                    u[self.action_inds[param, attr]] = act
                else:
                    cur_attr = cur_val if cur_val is not None else self.opt_traj[t, self.state_inds[param, attr]]
                    next_attr = self.opt_traj[t, self.state_inds[param, attr]]
                    u[self.action_inds[param, attr]] = next_attr - cur_attr
        else:
            for param, attr in self.action_inds:
                if attr.find('grip') >= 0:
                    val = self.opt_traj[-1, self.state_inds[param, attr]][0]
                    val = 0.1 if val <= 0. else -0.1
                    u[self.action_inds[param, attr]] = val 
        if np.any(np.isnan(u)):
            u[np.isnan(u)] = 0.
        return u


class EnvWrapper():
    def __init__(self, env, robot, mode='ee_pos', render=True):
        self.env = env
        self.physics = env.physics
        self.robot = robot
        self.geom = robot.geom
        self._type_cache = {}
        self.physics = env.physics
        self.model = self.physics.model
        self.mode = mode
        self.z_offsets = {}
        self.z_offsets = {'flat_block': 0.0}#1}
        self.upright_rot = Rotation.from_euler('xyz', [1.57, 1.57, 0.])
        self.upright_rot_inv = self.upright_rot.inv()
        self.flat_rot = Rotation.from_euler('xyz', [0., 0., 0.])
        self.flat_rot_inv = self.flat_rot.inv()
        self.use_render = render
        if render:
            try:
                self.im_font = ImageFont.truetype('E:/PythonPillow/Fonts/FreeMono.ttf', 10)
            except:
                self.im_font = None
        else:
            self.env._get_obs = lambda: {'image': np.zeros((4,4,3))}
        self.cur_obs = self.reset()

    def render(self, mode='rgb_array', resize=True, overlays=(), imsize=None, cam_ids=[-1]):
        params = {'distance': 1.8, 'azimuth': 90, 'elevation': -60,
                  'crop_box': (16.75, 25.0, 105.0, 88.75), 'size': 120}
        imsize = self.env.image_size if imsize is None else imsize
        wid = 88.25
        height = 63.75
        ratio = imsize / max(wid, height)
        if ratio > 1:
            params['size'] = int(ratio * 120)
            params['crop_box'] = (ratio*16.75, ratio*25., params['size']-ratio*15., params['size']-ratio*31.25)

        ims = []
        for cam_id in cam_ids:
            camera = mujoco.Camera(
                physics=self.physics, height=params['size'],
                width=params['size'], camera_id=cam_id)
            camera._render_camera.distance = params['distance']  # pylint: disable=protected-access
            camera._render_camera.azimuth = params['azimuth']  # pylint: disable=protected-access
            camera._render_camera.elevation = params['elevation']  # pylint: disable=protected-access
            camera._render_camera.lookat[:] = [0, 0.535, 1.1]  # pylint: disable=protected-access

            mjc_overlays = tuple(ovr for ovr in overlays if type(ovr) is not str)
            str_overlays = tuple(ovr for ovr in overlays if type(ovr) is str)
            image = camera.render(depth=False, segmentation=False, overlays=mjc_overlays)
            camera._scene.free()  # pylint: disable=protected-access

            if resize:
                image = Image.fromarray(image).crop(box=params['crop_box'])
                image = image.resize([imsize, imsize],
                                   resample=Image.ANTIALIAS)

            if len(str_overlays) and self.im_font is not None:
                border = 30
                image = ImageOps.expand(image, border=border, fill=(0, 0, 0))
                im_draw = ImageDraw.Draw(image)
                _, texth = self.im_font.getsize(str_overlays[0])
                pos = [(2,2), 
                        (2, imsize+2*border-texth-2), 
                        (2, 4+texth), 
                        (2, imsize+2*border-2*texth-4)]
                for ind, ovr in enumerate(str_overlays):
                    w, h = self.im_font.getsize(ovr)
                    x, y = pos[ind]
                    im_draw.rectangle((x, y, x+w, y+h), fill='black')
                    im_draw.text(pos[ind], ovr, fill=(255,255,255), font=self.im_font)

            image = np.asarray(image)
            ims.append(image)

        return np.concatenate(ims, axis=0)


    def get_attr(self, obj, attr, euler=False):
        if attr.find('ee_pos') >= 0:
            obj = 'end_effector'
            ind = self.env.physics.model.name2id(obj, 'site')
            return self.env.physics.data.site_xpos[ind]

        if attr.find('ee_rot') >= 0:
            obj = 'end_effector'
            ind = self.env.physics.model.name2id(obj, 'site')
            mat = self.env.physics.data.site_xmat[ind].reshape((3,3))
            rot = T.mat2quat(mat)
            if euler:
                rot = T.quaternion_to_euler(rot, 'xyzw')
            return rot

        use_vel = False
        if attr.find('_vel') >= 0:
            attr = attr[:attr.index('_vel')]
            use_vel = True

        if attr in self.geom.jnt_names:
            jnts = self.geom.jnt_names[attr]
            if use_vel:
                vals = self.get_joint_vels(jnts)
            else:
                vals = self.get_joints(jnts)
            return vals

        if attr.find('hinge') >= 0:
            if obj.find('drawer') >= 0:
                jnts = ['drawer_joint']
            elif obj.find('shelf') >= 0:
                jnts = ['slide_joint']
            return self.get_joints(jnts)

        if obj == 'panda':
            obj = 'panda0_link0'

        if attr == 'pose' or attr == 'pos':
            return self.get_item_pose(obj)[0]

        if attr.find('rot') >= 0 or attr.find('quat') >= 0:
            return self.get_item_pose(obj, euler=euler)[1]

        raise NotImplementedError('Could not retrieve', attr, 'of', obj)


    def set_attr(self, obj, attr, val, euler=False, forward=False):
        if attr in self.geom.jnt_names:
            jnts = self.geom.jnt_names[attr]
            return self.set_joints(jnts, val, forward=forward)

        if attr.find('hinge') >= 0:
            if obj.find('drawer') >= 0:
                jnts = ['drawer_joint']
            elif obj.find('shelf') >= 0:
                jnts = ['slide_joint']
            return self.set_joints(jnts, val, forward=forward)

        if attr.find('ee_pos') >= 0 or attr.find('ee_rot') >= 0:
            return

        if attr == 'pose' or attr == 'pos':
            return self.set_item_pose(obj, val, forward=forward)

        if attr.find('rot') >= 0 or attr.find('quat') >= 0:
            return self.set_item_pose(obj, quat=val, euler=euler, forward=forward)
   

    def get_item_pose(self, item_name, order='xyzw', euler=False):
        pos, quat = None, None
        if item_name.find('handle') >= 0:
            return self.get_handle_pose(item_name, order, euler)

        if item_name.find('drawer') >= 0: item_name = 'drawer'
        if item_name.find('shelf') >= 0: item_name = 'slide'

        pos = self.env.physics.named.data.xpos[item_name]
        quat = self.env.physics.named.data.xquat[item_name]
        #try:
        #    pose = self.env.physics.named.data.qpos[item_name]
        #    pos, quat = pose[:3], pose[3:7]
        #except Exception as e:
        #    pos = self.env.physics.named.data.xpos[item_name]
        #    quat = self.env.physics.named.data.xquat[item_name]

        if item_name.find('drawer') >= 0:
            pos = np.array([0., 0.85, 0.655])

        if item_name.find('shelf') >= 0 or item_name.find('slide') >= 0:
            pos = np.array([0., 0.85, 0.])

        if item_name.find('ball') >= 0:
            #quat = T.euler_to_quaternion([0., -0.8, 1.57], 'wxyz')
            if pos[2] > 0.65:
                quat = T.euler_to_quaternion([0., 1., -1.57], 'wxyz')
            else:
                quat = T.euler_to_quaternion([0., 0., 0.], 'wxyz')

        if item_name.find('button') >= 0:
            pos = pos.copy()
            pos[1] -= 0.035

        if order != 'xyzw':
            raise Exception()

        pos = [pos[0], pos[1], pos[2]+self.z_offsets.get(item_name, 0.0)]

        quat = [quat[1], quat[2], quat[3], quat[0]]
        if item_name.find('upright') >= 0:
            base_rot = Rotation.from_quat(quat)
            quat = (base_rot * self.upright_rot).as_quat()

        if item_name.find('flat') >= 0:
            base_rot = Rotation.from_quat(quat)
            quat = (base_rot * self.flat_rot).as_quat()

        rot = quat
        if euler:
            rot = T.quaternion_to_euler(quat, 'xyzw')
        return np.array(pos), np.array(rot)


    def set_item_pose(self, item_name, pos=None, quat=None, forward=False, order='xyzw', euler=False):
        if item_name == 'panda': return
        if item_name.find('desk') >= 0: return
        if item_name.find('shelf') >= 0: return
        if item_name.find('drawer') >= 0: return
        if item_name.find('handle') >= 0: return
        if item_name.find('button') >= 0: return

        if quat is not None and len(quat) == 3:
            quat = T.euler_to_quaternion(quat, order)

        if item_name.find('upright') >= 0 and quat is not  None:
            base_rot = Rotation.from_quat(quat)
            #quat = (self.upright_rot_inv * base_rot).as_quat()
            quat = (base_rot * self.upright_rot_inv).as_quat()

        if item_name.find('flat') >= 0 and quat is not None:
            base_rot = Rotation.from_quat(quat)
            quat = (self.flat_rot_inv * base_rot).as_quat()

        if quat is not None and order != 'wxyz':
            quat = [quat[3], quat[0], quat[1], quat[2]]

        if pos is not None:
            pos = [pos[0], pos[1], pos[2]-self.z_offsets.get(item_name, 0.0)]

        try:
            if pos is not None:
                self.env.physics.named.data.qpos[item_name][:3] = pos

            if quat is not None:
                self.env.physics.named.data.qpos[item_name][3:] = quat

        except KeyError as e:
            if pos is not None:
                self.env.physics.named.data.xpos[item_name][:] = pos

            if quat is not None:
                self.env.physics.named.data.xquat[item_name][:] = quat

        if forward:
            self.forward()

    def get_joints(self, jnt_names):
        vals = []
        for jnt in jnt_names:
            ind = self.env.physics.model.name2id(jnt, 'joint')
            adr = self.env.physics.model.jnt_qposadr[ind]
            vals.append(self.env.physics.data.qpos[adr])
        return np.array(vals)

    def get_joint_vels(self, jnt_names):
        vals = []
        for jnt in jnt_names:
            ind = self.env.physics.model.name2id(jnt, 'joint')
            adr = self.env.physics.model.jnt_qposadr[ind]
            vals.append(self.env.physics.data.qvel[adr])
        return np.array(vals)

    def set_joints(self, jnt_names, jnt_vals, forward=False):
        if len(jnt_vals) != len(jnt_names):
            print(jnt_names, jnt_vals, 'MAKE SURE JNTS MATCH')

        for jnt, val in zip(jnt_names, jnt_vals):
            ind = self.env.physics.model.name2id(jnt, 'joint')
            adr = self.env.physics.model.jnt_qposadr[ind]
            self.env.physics.data.qpos[adr] = val

        if forward:
            self.forward()

    def step(self, action):
        if self.mode.find('joint') >= 0:
            total_reward = 0.
            joint_position = action
            for t in range(self.env.action_repeat):
                for _ in range(10):
                    self.env.physics.data.ctrl[0:9] = joint_position[0:9]
                    # Ensure gravity compensation stays enabled.
                    self.env.physics.data.qfrc_applied[0:9] = self.physics.data.qfrc_bias[0:9]
                    ee_pos = self.env.physics.named.data.site_xpos['end_effector']
                    self.env.physics.named.data.xfrc_applied['ball'][2] = 0.
                    self.env.physics.named.data.xfrc_applied['flat_block'][2] = 0.
                    self.env.physics.named.data.xfrc_applied['upright_block'][2] = 0.
                    if self.env.physics.data.ctrl[-1] > 0.03:
                        if np.all(np.abs(ee_pos - self.env.physics.named.data.xpos['ball']) < 0.05):
                            self.env.physics.named.data.xfrc_applied['ball'][2] = -3.
                        if np.all(np.abs(ee_pos - self.env.physics.named.data.xpos['flat_block']) < 0.05):
                            self.env.physics.named.data.xfrc_applied['flat_block'][2] = -2.
                        #if np.all(np.abs(ee_pos - self.env.physics.named.data.xpos['upright_block']) < 0.05):
                        #    self.env.physics.named.data.xfrc_applied['upright_block'][2] = -2.
                    else:
                        #if np.all(np.abs(ee_pos - self.env.physics.named.data.xpos['ball']) < 0.05):
                        #    self.env.physics.named.data.xfrc_applied['ball'][2] = 3.
                        if np.all(np.abs(ee_pos - self.env.physics.named.data.xpos['flat_block']) < 0.04):
                            self.env.physics.named.data.xfrc_applied['flat_block'][2] = 5.
                        if np.all(np.abs(ee_pos - self.env.physics.named.data.xpos['upright_block']) < 0.04):
                            self.env.physics.named.data.xfrc_applied['upright_block'][2] = 3.

                    self.env.physics.step()
                    self.env.physics_copy.data.qpos[:] = self.physics.data.qpos[:]

                    if self.env.reward == 'dense':
                        total_reward += self.env._get_task_reward(self.env.task, 'dense_reward')
                    elif self.env.reward == 'sparse':
                        total_reward += float(self.env._get_task_reward(self.env.task, 'success'))
                    elif self.env.reward == 'success':
                        if self.env.success:
                            total_reward += 0  # Only give reward once in case episode continues.
                        else:
                            self.env.success = self.env._get_task_reward(self.env.task, 'success')
                            total_reward += float(self.env.success)
                    else:
                        raise ValueError(self.env.reward)

                self.env.num_steps += self.env.action_repeat
                if self.env.episode_length and self.env.num_steps >= self.env.episode_length:
                    done = True
                else:
                    done = False
            obs = self.env._get_obs()
            self.cur_obs = obs
            return obs, total_reward, done, {'discount': 1.0}

        obs, rew, done, info = self.env.step(action)
        self.cur_obs = obs
        return obs, rew, done, info

    def get_text_overlay(self, title='', body='', style='normal', position='top left'):
        return TextOverlay(title, body, style, position)

    def zero(self):
        self.env.physics.data.time = 0.0
        self.env.physics.data.qvel[:] = 0
        self.env.physics.data.qacc[:] = 0
        self.env.physics.data.qfrc_bias[:] = 0
        self.env.physics.data.qacc_warmstart[:] = 0
        self.env.physics.data.ctrl[:] = 0
        self.env.physics.data.qfrc_applied[:] = 0
        self.env.physics.data.xfrc_applied[:] = 0

    def forward(self):
        self.env.physics.forward()

    def reset(self):
        obs = self.env.reset()
        self.forward()
        self.cur_obs = obs
        self.init_obs = obs
        self.trans_obs = obs
        self.prev_obs = obs
        self._prev_task = None
        return obs


    def close(self):
        self.env.close()


class RobotAgent(TAMPAgent):
    def __init__(self, hyperparams):
        super(RobotAgent, self).__init__(hyperparams)

        self.optimal_pol_cls =  optimal_pol
        self.load_render = hyperparams['master_config'].get('load_render', False)
        self.ctrl_mode = 'joint' if ('panda', 'right') in self.action_inds else 'ee_pos'
        self.compound_goals = hyperparams['master_config'].get('compound_goals', False)
        self.max_goals = hyperparams['master_config'].get('max_goals', 3)
        self.max_goals = min(self.max_goals, len(self.prob.GOAL_OPTIONS))
        self.rlen = 6 if not self.compound_goals else 5 * (len(self.prob.INVARIANT_GOALS) + self.max_goals)
        self._load_goals()
        self.hor = 18


        config = {
            "obs_include": ["block{0}".format(i) for i in range(self.prob.N_OBJS)],
            "include_files": [PANDA_XML],
            "include_items": [],
            "items": [('robotview', '<camera mode="fixed" name="robotview" pos="2.0 0 2.4" quat="0.653 0.271 0.271 0.653"/>', {})],
            "view": False,
            "load_render": self.load_render,
            "sim_freq": 25,
            "timestep": 0.002,
            "image_dimensions": [],
            "step_mult": 5e0,
            "act_jnts": [
                "panda0_joint1",
                "panda0_joint2",
                "panda0_joint3",
                "panda0_joint4",
                "panda0_joint5",
                "panda0_joint6",
                "panda0_joint7",
                "panda0_finger_joint1",
                "panda0_finger_joint2"
            ],
        }

        for i in range(self.prob.N_OBJS):
            config["include_items"].append({
                    "name": "block{0}".format(i),
                    "type": "box",
                    "is_fixed": False,
                    "pos": plan.params['block{}'.format(i)].pose[:,0] - [0., 0.1, 0.61],
                    "dimensions": [0.03, 0.03, 0.03],
                    "rgba": (0.2, 0.2, 0.2, 1.0),
                })

        config["include_items"].append({
                "name": "table",
                "type": "box",
                "is_fixed": True,
                "pos": [0, 0, 0],
                "dimensions": [3., 3., 0.05],
                "rgba": (1.0, 1.0, 1.0, 1.0),
            })

        self.mjc_env = MJCEnv.load_config(config)
        prim_options = self.prob.get_prim_choices(self.task_list)
        self.obj_list = prim_options[OBJ_ENUM]
        self.panda = list(self.plans.values())[0].params['panda']
        self.check_col = hyperparams['master_config'].get('check_col', True)
        self.use_glew = False
        self._viewer = None
        self.cur_im = None
        self._matplot_im = None
        self.camera_id = 1
        self.main_camera_id = 0
        no = self._hyperparams['num_objs']
        self.targ_labels = {}
        self.cur_obs = self.mjc_env.reset()
        self.replace_cond(0)


    def _load_goals(self):
        self.goals = {}
        self.base_goals = {}
        for goal in self.prob.GOAL_OPTIONS:
            descr = goal.replace('(', '').replace(')', '')
            descr = descr.split(' ')
            self.goals[goal] = [descr[0], descr[1:]]
            self.base_goals[goal] = [descr[0], descr[1:]]

        for goal in self.prob.INVARIANT_GOALS:
            descr = goal.replace('(', '').replace(')', '')
            descr = descr.split(' ')
            self.goals[goal] = [descr[0], descr[1:]]


    def get_annotated_image(self, s, t, cam_id=None):
        #x = s.get_X(t=t)
        #self.reset_to_state(x, full=False)
        qpos = s.get(QPOS_ENUM, t=t)
        self.base_env.physics.data.qpos[:] = qpos
        self.base_env.physics.forward()
        targets = s.get(ONEHOT_GOAL_ENUM, t=t)
        task = s.get(FACTOREDTASK_ENUM, t=t).astype(int)
        taskname = self.task_list[task[0]]
        pos = s.get(END_POSE_ENUM, t=t).round(3)
        truepos = s.get(TRUE_POSE_ENUM, t=t).round(3)
        precost = round(self.precond_cost(s, tuple(task), t), 4)
        postcost = round(self.postcond_cost(s, tuple(task), t), 4)

        precost = str(precost)[1:]
        postcost = str(postcost)[1:]

        gripcmd = round(s.get_U(t=t)[self.action_inds['panda', 'right_gripper']][0], 2)

        #textover1 = self.mjc_env.get_text_overlay(body='Task: {0} Err: {1}'.format(task, truepos-pos))
        #textover2 = self.mjc_env.get_text_overlay(body='{0: <6} {1: <6}'.format(precost, postcost), position='bottom left')
        textover1 = 'TASK: {0} {1}'.format(task, taskname)
        textover2 = 'COST: {0: <5} {1: <5}'.format(precost, postcost)
        textover3 = '{0}'.format(str(truepos-pos)[1:-1])
        goalover = ''
        targets = s.targets

        for goal in self.prob.GOAL_OPTIONS + self.prob.INVARIANT_GOALS:
            if not self.compound_goals:
                if targets[self.target_inds[goal, 'value']] == 1.:
                    goalover += goal
            else:
                if targets[self.target_inds[goal, 'value']] == 1.:
                    goalover += "1 "
                else:
                    goalover += "0 "
        overlays = (textover1, textover2, textover3, goalover)
        #for ctxt in self.base_env.sim.render_contexts:
        #    ctxt._overlay[mj_const.GRID_TOPLEFT] = ['{}'.format(task), '']
        #    ctxt._overlay[mj_const.GRID_BOTTOMLEFT] = ['{0: <7} {1: <7} {2}'.format(precost, postcost, gripcmd), '']
        #return self.base_env.sim.render(height=self.image_height, width=self.image_width, camera_name="frontview")
        im = self.mjc_env.render(overlays=overlays, resize=True, imsize=112, cam_ids=self.cam_ids)
        #for ctxt in self.base_env.sim.render_contexts:
        #    for key in list(ctxt._overlay.keys()):
        #        del ctxt._overlay[key]
        return im


    def get_image(self, x, depth=False, cam_id=None):
        self.reset_to_state(x, full=False)
        im = self.base_env.render()
        return im


    def run_policy_step(self, u, x):
        self._col = []
        ctrl = {attr: u[inds] for (param_name, attr), inds in self.action_inds.items()}
        cur_grip = x[self.state_inds['panda', 'right_gripper']][0]

        panda = list(self.plans.values())[0].params['panda']
        true_lb, true_ub = panda.geom.get_joint_limits('right')
        factor = (np.array(true_ub) - np.array(true_lb)) / 5
        gripper = ctrl['right_gripper']
        #if 'right_ee_pos' in ctrl:
        #    targ_pos = self.mjc_env.get_attr('panda', 'right_ee_pos') + ctrl['right_ee_pos']
        #    rotoff = Rotation.from_rotvec(ctrl['right_ee_rot'])
        #    curquat = self.mjc_env.get_attr('panda', 'right_ee_rot', euler=False)
        #    targrot = (rotoff * Rotation.from_quat(curquat)).as_quat()
        #    for n in range(n_steps+1):
        #        curquat = self.mjc_env.get_attr('panda', 'right_ee_rot', euler=False)
        #        pos_ctrl = targ_pos - self.mjc_env.get_attr('panda', 'right_ee_pos')
        #        sign1 = np.sign(targrot[np.argmax(np.abs(targrot))])
        #        sign2 = np.sign(curquat[np.argmax(np.abs(curquat))])
        #        rot_ctrl = -sign1 * sign2 * robo_T.get_orientation_error(sign1*targrot, sign2*curquat)
        #        self.cur_obs, rew, done, _ = self.mjc_env.step(np.r_[pos_ctrl, rot_ctrl, [gripper]])

        if 'right' in ctrl:
            targ_pos = self.mjc_env.get_attr('panda', 'right') + ctrl['right']
            ctrl = np.r_[targ_pos, gripper]
            self.cur_obs, rew, done, _ = self.mjc_env.step(ctrl)

        rew = 0. # self.base_env.reward()
        self._rew = rew
        self._ret += rew
        col = 0 # 1 if len(self._col) > 0 else 0
        self.render_viewer(self.cur_obs['image'])
        return True, col


    def set_symbols(self, plan, task, anum=0, cond=0, targets=None, st=0):
        act_st, et = plan.actions[anum].active_timesteps
        st = max(act_st, st)
        if targets is None:
            targets = self.target_vecs[cond].copy()
        prim_choices = self.prob.get_prim_choices(self.task_list)
        act = plan.actions[anum]
        params = act.params
        #if params[2].is_symbol():
        #    params[2].value[:,0] = params[1].pose[:,st]
        #    params[2].rotation[:,0] = params[1].rotation[:,st]
        #params[3].value[:,0] = params[0].pose[:,st]
        #for arm in params[0].geom.arms:
        #    getattr(params[3], arm)[:,0] = getattr(params[0], arm)[:,st]
        #    gripper = params[0].geom.get_gripper(arm)
        #    getattr(params[3], gripper)[:,0] = getattr(params[0], gripper)[:,st]
        #    ee_attr = '{}_ee_pos'.format(arm)
        #    rot_ee_attr = '{}_ee_rot'.format(arm)
        #    if hasattr(params[0], ee_attr):
        #        getattr(params[3], ee_attr)[:,0] = getattr(params[0], ee_attr)[:,st]
        #    if hasattr(params[0], rot_ee_attr):
        #        getattr(params[3], rot_ee_attr)[:,0] = getattr(params[0], rot_ee_attr)[:,st]

        for tname, attr in self.target_inds:
            if tname in plan.params:
                getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]

        for pname in plan.params:
            if '{0}_init_target'.format(pname) in plan.params:
                plan.params['{0}_init_target'.format(pname)].value[:,0] = plan.params[pname].pose[:,st]
                if hasattr(plan.params[pname], 'rotation'):
                    plan.params['{0}_init_target'.format(pname)].rotation[:,0] = plan.params[pname].rotation[:,st]


    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        mp_state = mp_state.copy()
        if targets is None:
            targets = self.target_vecs[cond].copy()

        for (pname, aname), inds in self.state_inds.items():
            if aname == 'right_ee_pos':
                sample.set(RIGHT_EE_POS_ENUM, mp_state[inds], t)
                ee_pose = mp_state[inds]
                ee_rot = mp_state[self.state_inds[pname, 'right_ee_rot']]
                sample.set(RIGHT_EE_ROT_ENUM, ee_rot, t)
            elif aname.find('ee_pos') >= 0:
                sample.set(EE_ENUM, mp_state[inds], t)
                ee_pose = mp_state[inds]
                ee_rot = mp_state[self.state_inds[pname, 'ee_rot']]

        ee_spat = Rotation.from_euler('xyz', ee_rot)
        ee_quat = T.euler_to_quaternion(ee_rot, 'xyzw')
        ee_mat = T.quat2mat(ee_quat)
        sample.set(STATE_ENUM, mp_state, t)
        sample.set(DONE_ENUM, np.zeros(1), t)
        sample.set(TASK_DONE_ENUM, np.array([1, 0]), t)
        sample.set(QPOS_ENUM, self.base_env.physics.data.qpos.copy(), t)
        if self.hist_len > 0:
            sample.set(TRAJ_HIST_ENUM, self._prev_U.flatten(), t)
            x_delta = self._x_delta[1:] - self._x_delta[:1]
            sample.set(STATE_DELTA_ENUM, x_delta.flatten(), t)

        sample.set(STATE_HIST_ENUM, self._x_delta.flatten(), t)
        if self.task_hist_len > 0:
            sample.set(TASK_HIST_ENUM, self._prev_task.flatten(), t)

        robot = 'panda'
        if RIGHT_ENUM in self.sensor_dims:
            sample.set(RIGHT_ENUM, mp_state[self.state_inds[robot, 'right']], t)
        if RIGHT_VEL_ENUM in self.sensor_dims:
            sample.set(RIGHT_VEL_ENUM, self.mjc_env.get_attr('panda', 'right_vel'), t)
        if RIGHT_GRIPPER_ENUM in self.sensor_dims:
            sample.set(RIGHT_GRIPPER_ENUM, mp_state[self.state_inds[robot, 'right_gripper']], t)
            sample.set(GRIP_CMD_ENUM, self.base_env.physics.data.ctrl[-2:], t)

        prim_choices = self.prob.get_prim_choices(self.task_list)
        if task is not None:
            task = list(task)
            plan = list(self.plans.values())[0]
            task_ind = task[0]
            obj_ind = task[1]
            targ_ind = task[2]
            door_ind = task[3]

            task_name = prim_choices[TASK_ENUM][task_ind]
            obj_name = list(prim_choices[OBJ_ENUM])[task[1]]
            targ_name = list(prim_choices[TARG_ENUM])[task[2]]
            door_name = list(prim_choices[DOOR_ENUM])[task[3]]

            if task_name.find('slide') >= 0:
                obj_name = '{}_handle'.format(door_name)
                obj_ind = prim_choices[OBJ_ENUM].index(obj_name)
                task[1] = obj_ind

            task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
            task_vec[task[0]] = 1.
            sample.task_ind = task[0]
            sample.set(TASK_ENUM, task_vec, t)
            for ind, enum in enumerate(prim_choices):
                if ind >= len(task): break
                if hasattr(prim_choices[enum], '__len__'):
                    vec = np.zeros((len(prim_choices[enum])), dtype='float32')
                    vec[task[ind]] = 1.
                else:
                    vec = np.array(task[ind])
                sample.set(enum, vec, t)

            sample.set(FACTOREDTASK_ENUM, np.array(task), t)
            for (pname, aname), inds in self.state_inds.items():
                if aname.find('right_ee_pos') >= 0:
                    obj_pose = mp_state[self.state_inds[obj_name, 'pose']]
                    targ_pose = targets[self.target_inds[targ_name, 'value']]
                    obj_off_pose = mp_state[self.state_inds[obj_name, 'pose']] - mp_state[inds]
                    targ_off_pose = targets[self.target_inds[targ_name, 'value']] - mp_state[inds]
                    break

            obj_quat = T.euler_to_quaternion(mp_state[self.state_inds[obj_name, 'rotation']], 'xyzw')
            targ_quat = T.euler_to_quaternion(targets[self.target_inds[targ_name, 'rotation']], 'xyzw')
            sample.task = task
            sample.obj = task[1]
            sample.targ = task[2]
            sample.task_name = self.task_list[task[0]]

            grasp_pt = list(self.plans.values())[0].params[obj_name].geom.grasp_point
            task_name = self.task_list[task[0]].lower()

            if task_name.find('door') < 0:
                door_vec = np.zeros(len(prim_choices[DOOR_ENUM]))
                door_vec[:] = 1. / len(door_vec)
                sample.set(DOOR_ENUM, door_vec, t)
            else:
                door_name = prim_choices[DOOR_ENUM][task[2]]
                door_vec = np.zeros(len(prim_choices[DOOR_ENUM]))
                door_vec[task[2]] = 1.
                sample.set(DOOR_ENUM, door_vec, t)

            if task_name.lower() in ['move_to_grasp_right', 'lift_right', \
                                     'hold_right', 'hold_box_right', \
                                     'hold_can_right', 'hold_ball_right']:
                sample.set(END_POSE_ENUM, obj_pose, t)
                sample.set(END_ROT_ENUM, mp_state[self.state_inds[obj_name, 'rotation']], t)
                sample.set(ABS_POSE_ENUM, mp_state[self.state_inds[obj_name, 'pose']], t)
                targ_vec = np.zeros(len(prim_choices[TARG_ENUM]))
                targ_vec[:] = 1. / len(targ_vec)
                sample.set(TARG_ENUM, targ_vec, t)

            elif task_name.find('place_in_door') >= 0:
                door_geom = plan.params[door_name].geom
                door_pos = mp_state[self.state_inds[door_name, 'pose']] + door_geom.in_pos
                obj_pose = door_pos #- mp_state[self.state_inds[obj_name, 'pose']]
                sample.set(END_POSE_ENUM, obj_pose, t)
                sample.set(ABS_POSE_ENUM, door_pos, t)
                sample.set(END_ROT_ENUM, np.array(door_geom.in_orn), t)
                targ_vec = np.zeros(len(prim_choices[TARG_ENUM]))
                targ_vec[:] = 1. / len(targ_vec)
                sample.set(TARG_ENUM, targ_vec, t)

            elif task_name.find('slide') >= 0:
                door_geom = plan.params[door_name].geom
                cur_hinge = mp_state[self.state_inds[door_name, 'hinge']]
                targ_hinge = door_geom.open_val if task_name.find('open') >= 0 else door_geom.close_val
                obj_pose = mp_state[self.state_inds['{}_handle'.format(door_name), 'pose']] #(targ_hinge - cur_hinge) * np.abs(door_geom.open_dir) 
                sample.set(END_POSE_ENUM, obj_pose, t)
                sample.set(ABS_POSE_ENUM, mp_state[self.state_inds['{}_handle'.format(door_name), 'pose']], t)
                sample.set(END_ROT_ENUM, mp_state[self.state_inds[obj_name, 'rotation']], t)
                targ_vec = np.zeros(len(prim_choices[TARG_ENUM]))
                targ_vec[:] = 1. / len(targ_vec)
                sample.set(TARG_ENUM, targ_vec, t)

            elif self.task_list[task[0]].find('place') >= 0:
                sample.set(END_POSE_ENUM, targ_off_pose, t)
                sample.set(ABS_POSE_ENUM, targets[self.target_inds[targ_name, 'value']], t)
                sample.set(END_ROT_ENUM, targets[self.target_inds[targ_name, 'rotation']], t)

            elif self.task_list[task[0]].find('stack') >= 0:
                targ_off_pose = mp_state[self.state_inds['flat_block', 'pose']] # - obj_pose
                sample.set(END_POSE_ENUM, targ_off_pose, t)
                sample.set(ABS_POSE_ENUM, mp_state[self.state_inds['flat_block', 'pose']], t)
                sample.set(END_ROT_ENUM, mp_state[self.state_inds['flat_block', 'rotation']], t)
                targ_vec = np.zeros(len(prim_choices[TARG_ENUM]))
                targ_vec[:] = 1. / len(targ_vec)
                sample.set(TARG_ENUM, targ_vec, t)

            else:
                raise NotImplementedError()

            sample.set(TRUE_POSE_ENUM, sample.get(END_POSE_ENUM, t=t), t)
            sample.set(TRUE_ROT_ENUM, sample.get(END_ROT_ENUM, t=t), t)

        sample.condition = cond
        sample.set(TARGETS_ENUM, targets.copy(), t)
        sample.set(ONEHOT_GOAL_ENUM, targets, t)
        sample.targets = targets.copy()
        for i, obj in enumerate(prim_choices[OBJ_ENUM]):
            grasp_pt = list(self.plans.values())[0].params[obj].geom.grasp_point
            sample.set(OBJ_ENUMS[i], mp_state[self.state_inds[obj, 'pose']]+grasp_pt, t)
            sample.set(OBJ_DELTA_ENUMS[i], mp_state[self.state_inds[obj, 'pose']]-ee_pose+grasp_pt, t)
            obj_spat = Rotation.from_euler('xyz', mp_state[self.state_inds[obj, 'rotation']])
            sample.set(OBJ_ROTDELTA_ENUMS[i], (obj_spat.inv() * ee_spat).as_rotvec(), t)

        if fill_obs:
            if IM_ENUM in self._hyperparams['obs_include'] or \
               IM_ENUM in self._hyperparams['prim_obs_include']:
                im = self.mjc_env.cur_obs['image']
                im = (im - 128.) / 128.
                if self.incl_grip_obs:
                    grip_im = (self.mjc_env.cur_obs['hand_image'] - 128.) / 128.
                    im = np.c_[im, grip_im]

                if self.incl_init_obs:
                    init_im = (self.mjc_env.cur_obs['image']-self.mjc_env.init_obs['image'])  / 256.
                    im = np.c_[im, init_im]

                if self.incl_trans_obs:
                    trans_im = (self.mjc_env.cur_obs['image']-self.mjc_env.trans_obs['image'])  / 256.
                    im = np.c_[im, trans_im]

                sample.set(IM_ENUM, im.flatten(), t)


    def goal_f(self, condition, state, targets=None, cont=False, anywhere=False, tol=LOCAL_NEAR_TOL, verbose=False):
        if targets is None:
            targets = self.target_vecs[condition]

        cost = 0
        x = np.array(state)
        if len(x.shape) > 1: x = x[-1]
        for goal in self.goals:
            if targets[self.target_inds[goal, 'value']][0] == 1:
                key, params = self.goals[goal]
                suc = self.parse_goal(x, key, params)
                cost += 1 if not suc else 0

        return 1. if cost > 0 else 0.


    def parse_goal(self, x, key, params):
        key = key.lower()
        if key.find('lift') >= 0:
            suc = self._lifted(x, params[0])
        elif key.find('open') >= 0:
            suc = self._door_open(x, params[1])
        elif key.find('close') >= 0:
            suc = self._door_close(x, params[1])
        elif key.find('stack') >= 0:
            suc = self._stacked(x, params[0])
        elif key.find('near') >= 0 and params[1].find('off_desk') >= 0:
            suc = self._off_desk(x, params[0])
        elif key.find('near') >= 0 and params[1].find('bin') >= 0:
            suc = self._in_bin(x, params[0])
        elif key.find('inslide') >= 0 and params[1].find('shelf') >= 0:
            suc = self._in_shelf(x, params[0])
        elif key.find('inslide') >= 0 and params[1].find('drawer') >= 0:
            suc = self._in_drawer(x, params[0])
        elif key.find('grip') >= 0 and params[1].find('button') >= 0:
            suc = self._button(x, params[1])
        else:
            raise NotImplementedError('Cannot parse goal for {} {}'.format(key, params))

        return suc
    

    def reset_to_sample(self, sample):
        self.reset_to_state(sample.get_X(sample.T-1))


    def reset(self, m):
        self.reset_to_state(self.x0[m])


    def reset_to_state(self, x, full=True):
        mp_state = x[self._x_data_idx[STATE_ENUM]]
        self._done = 0.
        self._ret = 0.
        self._rew = 0.
        self._prev_U = np.zeros((self.hist_len, self.dU))
        self._x_delta = np.zeros((self.hist_len+1, self.dX))
        self.eta_scale = 1.
        self._noops = 0
        self._x_delta[:] = x.reshape((1,-1))
        self._prev_task = np.zeros((self.task_hist_len, self.dPrimOut))
        self.cur_state = x.copy()
        obs = self.mjc_env.reset()
        for (pname, aname), inds in self.state_inds.items():
            if aname.find('ee_pos') >= 0 or aname.find('ee_rot') >= 0: continue
            val = x[inds]
            self.mjc_env.set_attr(pname, aname, val, forward=False)
        self.base_env.physics.forward()


    def get_state(self, clip=False):
        x = np.zeros(self.dX)
        for (pname, aname), inds in self.state_inds.items():
            val = self.mjc_env.get_attr(pname, aname, euler=True)
            '''
            if clip:
                if aname in ['right']:
                    lb, ub = self.mjc_env.geom.get_joint_limits(aname)
                    val = np.maximum(np.minimum(val, ub), lb)
                elif aname.find('gripper') >= 0:
                    cv, ov = self.panda.geom.get_gripper_closed_val(), self.panda.geom.get_gripper_open_val()
                    val = ov if np.max(np.abs(val-cv)) > np.max(np.abs(val-ov)) else cv
            '''

            if len(inds) != len(val):
                raise Exception('Bad state retrieval for', pname, aname, 'expected', len(inds), 'but got', len(val))

            x[inds] = val

        if clip: x = self.clip_state(x)
        return x


    def reset_mjc_env(self, x, targets=None, draw_targets=True):
        pass


    def get_mjc_obs(self, x):
        return self.mjc_env.render()


    def sample_optimal_trajectory(self, state, task, condition, opt_traj=[], traj_mean=[], targets=[]):
        if not len(targets):
            old_targets = self.target_vecs[condition].copy()
        else:
            old_targets = self.target_vecs[condition].copy()
            for tname, attr in self.target_inds:
                if attr == 'value':
                    self.targets[condition][tname] = targets[self.target_inds[tname, attr]]
            self.target_vecs[condition] = targets

        exclude_targets = []
        plan = self.plans[task]
        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, opt_traj), condition, state, task, noisy=False, skip_opt=True, hor=len(opt_traj))
        sample.set_ref_X(sample.get_X())
        sample.set_ref_U(sample.get_U())

        self.target_vecs[condition] = old_targets
        for tname, attr in self.target_inds:
            if attr == 'value':
                self.targets[condition][tname] = old_targets[self.target_inds[tname, attr]]
        return sample


    def get_random_initial_state_vec(self, config, plans, dX, state_inds, ncond):
        self.cur_obs = self.mjc_env.reset()
        x = np.zeros(self.dX)
        for pname, aname in self.state_inds:
            inds = self.state_inds[pname, aname]
            val = self.mjc_env.get_attr(pname, aname, euler=True)
            if len(inds) == 1: val = np.mean(val)
            x[inds] = val

        targets = {goal: 0 for goal in self.goals}
        plan = list(self.plans.values())[0]
        for targ in ['off_desk_target', 'bin_target']:
            targets[targ] = plan.params[targ].value[:,0].copy()

        used_params = []
        goal_descrs = list(self.base_goals.keys())
        order = np.random.permutation(len(goal_descrs))

        i = 0
        for ind in order:
            goal = goal_descrs[ind]
            key, params = self.goals[goal]
            if params[0] in used_params: continue
            if key in self.prob.INVARIANT_GOALS: continue
            i += 1
            used_params.append(params[0])
            targets[goal] = 1
            if not self.compound_goals: break
            if i >= self.max_goals: break

        for goal in self.prob.INVARIANT_GOALS:
            key, params = self.goals[goal]
            targets[goal] = 1

        return [x], [targets] 
   

    def set_to_target(self, obj, targets=None):
        pass

    
    def replace_cond(self, cond, curric_step=-1):
        self.cur_obs = self.mjc_env.reset()
        x, targets = self.get_random_initial_state_vec(self.config, self.plans, self.dX, self.state_inds, 1)
        x, targets = x[0], targets[0]
        self.init_vecs[cond] = x
        self.x0[cond] = self.init_vecs[cond][:self.symbolic_bound]
        self.target_vecs[cond] = np.zeros((self.target_dim,))
        self.targets[cond] = targets

        prim_choices = self.prob.get_prim_choices(self.task_list)
        for target_name in self.targets[cond]:
            self.target_vecs[cond][self.target_inds[target_name, 'value']] = self.targets[cond][target_name]


    def goal(self, cond, targets=None):
        if targets is None:
            targets = self.target_vecs[cond]

        goal = ''
        for (descr, attr), inds in self.target_inds.items():
            if type(descr) is str and targets[inds][0] == 1:
                goal += descr

        return goal


    def onehot_encode_goal(self, targets, descr=None, debug=False):
        vec = np.zeros(len(self.goals))
        for ind, goal in enumerate(self.goals):
            if targets[self.target_inds[goal, 'value']][0] == 1:
                vec[ind] = 1.
        return vec


    def permute_hl_data(self, hl_mu, hl_obs, hl_wt, hl_prc, aux):
        return hl_mu, hl_obs, hl_wt, hl_prc


    def permute_tasks(self, tasks, targets, plan=None, x=None):
        raise NotImplementedError()


    def encode_plan(self, plan, permute=False):
        encoded = []
        prim_choices = self.prob.get_prim_choices(self.task_list)
        for a in plan.actions:
            encoded.append(self.encode_action(a))
        encoded = [tuple(l) for l in encoded]
        return encoded


    def encode_action(self, action):
        prim_choices = self.prob.get_prim_choices(self.task_list)
        astr = str(action).lower()
        l = [0]
        for i, task in enumerate(self.task_list):
            if action.name.lower() == task:
                l[0] = i
                break

        ind = len(action.params)
        if action.name.lower().find('move_to_grasp') >= 0:
            ind =  2
        elif action.name.lower().find('place_in_door') >= 0:
            ind = 3
        elif action.name.lower().find('lift') >= 0:
            ind =  2
        elif action.name.lower().find('hold') >= 0:
            ind =  2

        for enum in prim_choices:
            if enum is TASK_ENUM: continue
            if hasattr(prim_choices[enum], '__len__'):
                l.append(0)
                for i, opt in enumerate(prim_choices[enum]):
                    if opt in [p.name for p in action.params[:ind]]:
                        if action.name.lower().find('stack') >= 0:
                            if enum is OBJ_ENUM and opt.find('flat') >= 0:
                                continue
                        l[-1] = i
                        break
            #else:
            #    param = action.params[1]
            #    l[-1] = param.value[:,0] if param.is_symbol() else param.pose[:,action.active_timesteps[0]]
        return l # tuple(l)


    def retime_traj(self, traj, vel=0.01, inds=None, minpts=10):
        new_traj = []
        if len(np.shape(traj)) == 2:
            traj = [traj]
        for step in traj:
            xpts = []
            fpts = []
            grippts= []
            d = 0
            rotinds = self.state_inds['panda', 'right_ee_rot']
            rotpts =[]
            if inds is None:
                if ('panda', 'right_ee_pos') in self.action_inds:
                    inds = self.state_inds['panda', 'right_ee_pos']
                elif ('panda', 'right') in self.action_inds:
                    inds = self.state_inds['panda', 'right']

            for t in range(len(step)):
                xpts.append(d)
                fpts.append(step[t])
                grippts.append(step[t][self.state_inds['panda', 'right_gripper']])
                rotpts.append(step[t][self.state_inds['panda', 'right_ee_rot']])
                if t < len(step) - 1:
                    disp = np.linalg.norm(step[t+1][inds] - step[t][inds])
                    d += disp
            assert not np.any(np.isnan(xpts))
            assert not np.any(np.isnan(fpts))
            interp = scipy.interpolate.interp1d(xpts, fpts, axis=0, fill_value='extrapolate')
            grip_interp = scipy.interpolate.interp1d(np.array(xpts), grippts, kind='next', bounds_error=False, axis=0)
            rot_interp = scipy.interpolate.interp1d(np.array(xpts), rotpts, kind='next', bounds_error=False, axis=0)

            fix_pts = []
            if type(vel) is float:
                # x = np.arange(0, d+vel/2, vel)
                # npts = max(int(d/vel), minpts)
                # x = np.linspace(0, d, npts)

                x = []
                for i, d in enumerate(xpts):
                    if i == 0:
                        x.append(0)
                        fix_pts.append((len(x)-1, fpts[i]))
                    elif xpts[i] - xpts[i-1] <= 1e-6:
                        continue
                    elif xpts[i] - xpts[i-1] <= vel:
                        x.append(x[-1] + xpts[i] - xpts[i-1])
                        fix_pts.append((len(x)-1, fpts[i]))
                    else:
                        n = max(2, int((xpts[i]-xpts[i-1])//vel))
                        for _ in range(n):
                            x.append(x[-1] + (xpts[i]-xpts[i-1])/float(n))
                        x[-1] = d
                        fix_pts.append((len(x)-1, fpts[i]))
                # x = np.cumsum(x)
            elif type(vel) is list:
                x = np.r_[0, np.cumsum(vel)]
            else:
                raise NotImplementedError('Velocity undefined')
            out = interp(x)
            grip_out = grip_interp(x)
            rot_out = rot_interp(x)
            out[:, self.state_inds['panda', 'right_gripper']] = grip_out
            #out[:, self.state_inds['panda', 'right_ee_rot']] = rot_out
            for step in out:
                self.panda.openrave_body.set_pose(step[self.state_inds['panda', 'pose']])
                self.panda.openrave_body.set_dof({'right': step[self.state_inds['panda', 'right']]})
                info = self.panda.openrave_body.fwd_kinematics('right')
                out[:, self.state_inds['panda', 'right_ee_pos']] = info['pos']
                out[:, self.state_inds['panda', 'right_ee_rot']] = T.quaternion_to_euler(info['quat'], 'xyzw')
            for pt, val in fix_pts:
                out[pt] = val
            out[0] = step[0]
            out[-1] = step[-1]
            if len(new_traj):
                new_traj = np.r_[new_traj, out]
            else:
                new_traj = out
            if np.any(np.isnan(out)): print(('NAN in out', out, x))
        return new_traj


    def compare_tasks(self, t1, t2):
        return t1[0] == t2[0] and t1[1] == t2[1]

   
    def get_inv_cov(self):
        vec = np.ones(self.dU)
        robot = 'panda'
        if ('panda', 'right') in self.action_inds:
            return np.eye(self.dU)
        elif ('panda', 'right_ee_pos') in self.action_inds and ('panda', 'right_ee_rot') in self.action_inds:
            vecs = np.array([1e1, 1e1, 1e1, 1e-2, 1e-2, 1e-2, 1e0])
        return np.diag(vec)


    def clip_state(self, x):
        x = x.copy()
        lb, ub = self.panda.geom.get_joint_limits('right')
        lb = np.array(lb) + 2e-3
        ub = np.array(ub) - 2e-3
        jnt_vals = x[self.state_inds['panda', 'right']]
        x[self.state_inds['panda', 'right']] = np.clip(jnt_vals, lb, ub)
        cv, ov = self.panda.geom.get_gripper_closed_val(), self.panda.geom.get_gripper_open_val()
        grip_vals = x[self.state_inds['panda', 'right_gripper']]
        #grip_vals = ov if np.mean(np.abs(grip_vals-cv)) > np.mean(np.abs(grip_vals-ov)) else cv
        grip_vals = cv
        x[self.state_inds['panda', 'right_gripper']] = grip_vals
        return x


    def feasible_state(self, x, targets):
        return True

    
    def reward(self, x=None, targets=None, center=False):
        return 0. # self.base_env.reward()


    def add_viewer(self):
        if self._viewer is not None or self.cur_im is not None: return
        self.cur_im = np.zeros((2*self.image_height, self.image_width, 3))
        self._launch_viewer(self.image_width, 2*self.image_height)


    def _launch_viewer(self, width, height, title='Main'):
        self._matplot_view_thread = None
        if self.use_glew:
            from dm_control.viewer import viewer
            from dm_control.viewer import views
            from dm_control.viewer import gui
            from dm_control.viewer import renderer
            self._renderer = renderer.NullRenderer()
            self._render_surface = None
            self._viewport = renderer.Viewport(width, height)
            self._window = gui.RenderWindow(width, height, title)
            self._viewer = viewer.Viewer(
                self._viewport, self._window.mouse, self._window.keyboard)
            self._viewer_layout = views.ViewportLayout()
            self._viewer.render()
        else:
            self._viewer = None
            self._matplot_im = None
            self._run_matplot_view()


    def _reload_viewer(self):
        if self._viewer is None or not self.use_glew: return

        if self._render_surface:
          self._render_surface.free()

        if self._renderer:
          self._renderer.release()

        self._render_surface = render.Renderer(
            max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE)
        self._renderer = renderer.OffScreenRenderer(
            self.physics.model, self._render_surface)
        self._renderer.components += self._viewer_layout
        self._viewer.initialize(
            self.physics, self._renderer, touchpad=False)
        self._viewer.zoom_to_scene()


    def render_viewer(self, pixels):
        if self.use_glew and self.viewer is not None:
            with self._window._context.make_current() as ctx:
                ctx.call(
                    self._window._update_gui_on_render_thread, self._window._context.window, pixels)
            self._window._mouse.process_events()
            self._window._keyboard.process_events()
        elif not self.use_glew and self._matplot_im is not None:
            self._matplot_im.set_data(pixels)
            plt.draw()


    def _run_matplot_view(self):
        self._matplot_view_thread = Thread(target=self._launch_matplot_view)
        self._matplot_view_thread.daemon = True
        self._matplot_view_thread.start()


    def _launch_matplot_view(self):
        try:
            self._matplot_im = plt.imshow(self.cur_im)
            plt.show()
        except TclError:
            print('\nCould not find display to launch viewer (this does not affect the ability to render images)\n')


    def distance_to_goal(self, x=None, targets=None):
        return 0.


    def reward(self, x=None, targets=None, center=False):
        if x is None: x = self.get_state()
        if targets is None: targets = self.target_vecs[0]
        return 0.


    def _in_shelf(self, x, item_name):
        pos = x[self.state_inds[item_name, 'pose']]
        #return pos[0] > 0.2 and pos[1] > 0.9
        return pos[0] > 0. and pos[1] > SHELF_Y


    def _in_bin(self, x, item_name):
        pos = x[self.state_inds[item_name, 'pose']]
        return pos[0] > 0.25 and pos[0] < 0.55 and pos[1] > 0.4 and pos[1] < 0.65 and pos[2] < 0.6


    def _off_desk(self, x, item_name):
        pos = x[self.state_inds[item_name, 'pose']]
        return np.abs(pos[0]) > 0.6 or pos[1] < 0.65


    def _lifted(self, x, item_name):
        pos = x[self.state_inds[item_name, 'pose']]
        thresh = 0.82
        if item_name.find('upright') >= 0:
            thresh = 0.87
        elif item_name.find('flat') >= 0:
            thresh = 0.81 + self.mjc_env.z_offsets['flat_block']
        return pos[2] > thresh


    def _door_open(self, x, door_name):
        door = list(self.plans.values())[0].params[door_name]
        open_val = door.geom.open_thresh
        pos = x[self.state_inds[door_name, 'hinge']][0]
        sgn = -1. if door.geom.open_val < door.geom.close_val else 1.
        return sgn*pos > sgn*open_val


    def _door_close(self, x, door_name):
        door = list(self.plans.values())[0].params[door_name]
        close_val = door.geom.close_thresh
        pos = x[self.state_inds[door_name, 'hinge']][0]
        sgn = -1. if door.geom.open_val < door.geom.close_val else 1.
        return sgn*pos < sgn*close_val


    def _stacked(self, x, item_name, base_item='flat_block'):
        pos1 = x[self.state_inds[item_name, 'pose']]
        pos2 = x[self.state_inds[base_item, 'pose']]
        #return np.linalg.norm((pos1-pos2) - [0., 0., 0.037]) < 0.04
        return np.linalg.norm((pos1-pos2) - [0., 0., 0.038]) < STACK_OFFSET


    def _button(self, x, button_name):
        color = button_name.split('_')[0]
        val = self.base_env.physics.named.data.qpos[color + '_light'][0]
        pressed = val < -0.00453
        return pressed


    def _slide(self, x, door_name, door_open=True):
        door = list(self.plans.values())[0].params[door_name]
        hinge_val = x[self.state_inds[door_name, 'hinge']][0]
        close_val, open_val = door.geom.close_thresh, door.geom.open_thresh
        sgn = 1. if open_val < close_val else -1.
        if door_open: return sgn*hinge_val < sgn*open_val
        return sgn*hinge_val >= sgn*close_val


    def _near(self, x, targets, obj_name, targ_name, near_tol=0.1):
        obj_pos = x[self.state_inds[obj_name, 'pose']]
        targ_pos = targets[self.target_inds[targ_name, 'value']]
        return np.linalg.norm(obj_pos-targ_pos) < near_tol


    def _off_desk(self, x, obj_name):
        return x[self.state_inds[obj_name, 'pose']][2] < 0.6


    def _in_drawer(self, x, obj_name):
        pos = x[self.state_inds[obj_name, 'pose']]
        return pos[2] < 0.7 and pos[2] > 0.5 and pos[0] > -0.3 and pos[0] < 0.3 # and pos[1] > 0.35


    def set_task_info(self, sample, cur_state, t, cur_task, task_f, policies=None):
        task, policy = super(RobotAgent, self).set_task_info(sample, cur_state, t, cur_task, task_f, policies)
        self.mjc_env.update_task(self.task_list[task[0]])
        return task, policy

    def precond_cost(self, sample, task=None, t=0, tol=1e-3, x0=None, debug=False):
        if task is None: task = tuple(sample.get(FACTOREDTASK_ENUM, t=t))
        return self.cost_f(sample.get_X(t), task, sample.condition, active_ts=(0, 0), targets=sample.targets, tol=tol, x0=x0, debug=debug)


    def postcond_cost(self, sample, task=None, t=None, debug=False, tol=1e-3, x0=None):
        if t is None: t = sample.T-1
        if task is None: task = tuple(sample.get(FACTOREDTASK_ENUM, t=t))
        task_name = self.task_list[task[0]].lower()
        obj_name = self.prim_choices[OBJ_ENUM][task[1]].lower()
        targ_name = self.prim_choices[TARG_ENUM][task[2]].lower()
        door_name = self.prim_choices[DOOR_ENUM][task[3]].lower()
        x = sample.get_X(t=t)
        if task_name.find('place_in_door') >= 0:
            pos = x[self.state_inds[obj_name, 'pose']]
            if debug: print('POSTCOND INFO PLACE IN DOOR:', obj_name, door_name, pos)
            if door_name.find('shelf') >= 0:
                if (pos[0] < 0. or pos[1] < SHELF_Y): return 1.
            elif door_name.find('drawer') >= 0:
                if not (pos[0] > -0.3 and pos[0] < 0.3 and pos[2] < 0.73 and pos[2] > 0.5): return 1.
        elif task_name.find('place') >= 0:
            pos = x[self.state_inds[obj_name, 'pose']]
            if debug: print('POSTCOND INFO PLACE:', obj_name, targ_name, pos, sample.targets[self.target_inds[targ_name, 'value']])
            if targ_name.find('bin') >= 0:
                return 0. if (pos[2] < 0.6 and pos[0] > 0.25 and pos[0] < 0.55 and pos[1] > 0.35) else 1.
            elif targ_name.find('off') >= 0:
                return 0. if pos[2] < 0.6 else 1.
        elif task_name.find('stack') >= 0:
            pos = x[self.state_inds[obj_name, 'pose']]
            base_pos = x[self.state_inds['flat_block', 'pose']]
            targ_offset = [0., 0., 0.03778]
            offset = np.linalg.norm((pos-base_pos) - targ_offset)
            if debug: print('POSTCOND INFO STACK:', obj_name, pos, base_pos, offset)
            if offset > STACK_OFFSET: return offset
        elif task_name.find('hold') >= 0 and obj_name.find('green_button') >= 0:
            val = self.base_env.physics.named.data.qpos['green_light'][0]
            return 0. if val < -0.00453 else 1.

        return self.cost_f(sample.get_X(t), task, sample.condition, active_ts=(-1, -1), targets=sample.targets, debug=debug, tol=tol, x0=x0)


    def fill_cont(self, policy, sample, t):
        vals = policy.act(sample.get_X(t=t), sample.get_cont_obs(t=t), t)
        old_vals = {}
        for ind, enum in enumerate(self.continuous_opts):
            old_vals[enum] = sample.get(enum, t=t).copy()
            sample.set(enum, vals[ind], t=t)
        sample.set(TRUE_POSE_ENUM, sample.get(END_POSE_ENUM, t=t), t=t)
        sample.set(TRUE_ROT_ENUM, sample.get(END_ROT_ENUM, t=t), t=t)
        return old_vals


    def get_hist_info(self):
        info = {'cur_obs': self.mjc_env.cur_obs,
                'init_obs': self.mjc_env.init_obs,
                'trans_obs': self.mjc_env.trans_obs,
                'qpos': self.base_env.physics.data.qpos.copy(),
                'qvel': self.base_env.physics.data.qvel.copy(),
                }
        return info


    def store_hist_info(self, info):
        #self.mjc_env.cur_obs = info['cur_obs']
        self.mjc_env.init_obs = info['init_obs']
        self.mjc_env.trans_obs = info['trans_obs']
        #self.base_env.physics.data.qpos[:] = info['qpos']
        #self.base_env.physics.data.qvel[:] = info['qvel']
        #self.base_env.physics.forward()

    
    def update_hist_info(self, info):
        info['trans_obs'] = self.mjc_env.cur_obs
        info['cur_obs'] = self.mjc_env.cur_obs
        info['qpos'] = self.base_env.physics.data.qpos.copy()
        info['qvel'] = self.base_env.physics.data.qvel.copy()

