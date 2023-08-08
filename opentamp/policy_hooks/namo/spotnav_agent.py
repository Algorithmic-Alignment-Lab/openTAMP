import copy
import sys
import time
import traceback

import pickle as pickle

import ctypes

import numpy as np
import scipy.interpolate

import xml.etree.ElementTree as xml

from sco_py.expr import *

import pybullet as P

import opentamp
import opentamp.core.util_classes.common_constants as const
from opentamp.envs.mjc_env import MJCEnv
import opentamp.core.util_classes.items as items
from opentamp.core.util_classes.namo_grip_predicates import dsafe, NEAR_TOL, dmove, HLGraspFailed, HLTransferFailed, HLPlaceFailed, GRIP_VAL
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
import opentamp.pma.backtrack_ll_solver_OSQP as bt_ll_OSQP
from opentamp.policy_hooks.sample import Sample
from opentamp.policy_hooks.utils.policy_solver_utils import *
import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.policy_hooks.utils.tamp_eval_funcs import *
from opentamp.policy_hooks.namo.namo_agent import NAMOSortingAgent


bt_ll_OSQP.INIT_TRAJ_COEFF = 1e-2

HUMAN_TARGS = [
                (9.0, 0.),
                (9.0, 2.0),
                (9.0, 1.0),
                (9.0, -1.0),
                (9.0, -2.0),
                (9.0, -3.0),
                (9.0, -4.0),
                (-9.0, 2.),
                (-9.0, 1.),
                (-9.0, 0.),
                (-9.0, -1.0),
                (-9.0, -2.0),
                (-9.0, -3.0),
                (-9.0, -4.0),
                ]

MAX_SAMPLELISTS = 1000
MAX_TASK_PATHS = 100
GRIP_TOL = 0.
MIN_STEP = 1e-2
LIDAR_DIST = 2.
# LIDAR_DIST = 1.5
DSAFE = 5e-1
MAX_STEP = max(1.5*dmove, 1)
LOCAL_FRAME = True
NAMO_XML = opentamp.__path__[0] + '/robot_info/lidar_spot_imit.xml'


class optimal_pol:
    def __init__(self, dU, action_inds, state_inds, opt_traj):
        self.dU = dU
        self.action_inds = action_inds
        self.state_inds = state_inds
        self.opt_traj = opt_traj

    def act(self, X, O, t, noise=None):
        u = np.zeros(self.dU)
        if t < len(self.opt_traj) - 1:
            for param, attr in self.action_inds:
                # if attr == 'gripper':
                #     u[self.action_inds[param, attr]] = self.opt_traj[t, self.state_inds[param, attr]]
                if attr == 'pose':
                    x, y = self.opt_traj[t+1, self.state_inds['pr2', 'pose']]
                    curx, cury = X[self.state_inds['pr2', 'pose']]
                    theta = -self.opt_traj[t, self.state_inds['pr2', 'theta']][0]
                    if LOCAL_FRAME:
                        relpos = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).dot([x-curx, y-cury])
                    else:
                        relpos = [x-curx, y-cury]
                    u[self.action_inds['pr2', 'pose']] = relpos
                elif attr == 'vel':
                    vel = self.opt_traj[t+1, self.state_inds['pr2', 'vel']] # np.linalg.norm(self.opt_traj[t+1, inds]-X[inds])
                    vel = np.linalg.norm(self.opt_traj[t+1, self.state_inds['pr2', 'pose']] - X[self.state_inds['pr2', 'pose']])
                    if self.opt_traj[t+1, self.state_inds['pr2', 'vel']] < 0:
                        vel *= -1
                    u[self.action_inds[param, attr]] = vel
                elif attr == 'theta':
                    u[self.action_inds[param, attr]] = self.opt_traj[t+1, self.state_inds[param, attr]] - X[self.state_inds[param, attr]]
                else:
                    u[self.action_inds[param, attr]] = self.opt_traj[t+1, self.state_inds[param, attr]] - X[self.state_inds[param, attr]]
        # else:
        #     u[self.action_inds['pr2', 'gripper']] = self.opt_traj[-1, self.state_inds['pr2', 'gripper']]
        if np.any(np.isnan(u)):
            u[np.isnan(u)] = 0.
        return u


class SpotNavAgent(NAMOSortingAgent):
    def __init__(self, hyperparams):
        super(NAMOSortingAgent, self).__init__(hyperparams)

        self.optimal_pol_cls = optimal_pol
        self._feasible = True
        for plan in list(self.plans.values()):
            for t in range(plan.horizon):
                plan.params['obs0'].pose[:,t] = plan.params['obs0'].pose[:,0]

        self.check_col = hyperparams['master_config'].get('check_col', True)
        self.robot_height = 1
        self.use_mjc = hyperparams.get('use_mjc', False)
        self.vel_rat = 0.05
        n_obj = hyperparams['master_config']['num_objs']
        self.rlen = 30
        self.hor = 15 # 4 * n_obj + 4
        #self.vel_rat = 0.05
        #self.rlen = self.num_objs * len(self.task_list)
        #if self.retime: self.rlen *= 2
        #self.hor = 15
        wall_dims = OpenRAVEBody.get_wall_dims('closet')
        config = {
            'obs_include': ['can{0}'.format(i) for i in range(hyperparams['num_objs'])],
            'include_files': [NAMO_XML],
            'include_items': [],
            'view': False,
            'sim_freq': 50,
            'timestep': 0.002,
            'image_dimensions': (hyperparams['image_width'], hyperparams['image_height']),
            'step_mult': 5e0,
            'act_jnts': ['robot_x', 'robot_y', 'robot_theta']
        }

        self.main_camera_id = 0
        # colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0.5, 0.75, 0.25, 1], [0.75, 0.5, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0.7, 0.7, 0.1, 1], [1., 0.1, 0.8, 1], [0.5, 0.95, 0.5, 1], [0.75, 0.4, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]

        items = config['include_items']
        prim_options = self.prob.get_prim_choices(self.task_list)
        for name in prim_options[OBJ_ENUM]:
            if name =='pr2': continue
            cur_color = colors.pop(0)
            # items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.3, 0.4), 'rgba': tuple(cur_color), 'mass': 10.})
            #items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.3, 0.2), 'rgba': tuple(cur_color), 'mass': 10.})
            items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.3, 0.2), 'rgba': tuple(cur_color), 'mass': 40.})
            targ_color = cur_color[:3] + [1.] # [0.75] # [0.25]
            #items.append({'name': '{0}_end_target'.format(name), 'type': 'box', 'is_fixed': True, 'pos': (0, 0, 1.5), 'dimensions': (0.45, 0.45, 0.045), 'rgba': tuple(targ_color), 'mass': 1.})
            items.append({'name': '{0}_end_target'.format(name), 'type': 'box', 'is_fixed': True, 'pos': (0, 0, 1.5), 'dimensions': (0.35, 0.35, 0.045), 'rgba': tuple(targ_color), 'mass': 1.})

        for i in range(len(wall_dims)):
            dim, next_trans = wall_dims[i]
            next_trans[0,3] -= 3.5
            next_dim = dim # [dim[1], dim[0], dim[2]]
            pos = next_trans[:3,3] # [next_trans[1,3], next_trans[0,3], next_trans[2,3]]
            items.append({'name': 'wall{0}'.format(i), 'type': 'box', 'is_fixed': True, 'pos': pos, 'dimensions': next_dim, 'rgba': (0.2, 0.2, 0.2, 1)})

        self.humans = {}
        self.human_trajs = {}
        for human_id in range(self.prob.N_HUMAN):
            self.humans['human{}'.format(human_id)] = HUMAN_TARGS[np.random.randint(len(HUMAN_TARGS))]
            self.human_trajs['human{}'.format(human_id)] = np.zeros(2) 
            items.append({'name': 'human{}'.format(human_id),
                          'type': 'cylinder',
                          'is_fixed': False,
                          'pos': [0., 0., 0.],
                          'dimensions': [0.3, 0.2],
                          'mass': 10,
                          'rgba': (1., 1., 1., 1.)})

        no = self._hyperparams['num_objs']
        nt = self._hyperparams['num_targs']
        config['load_render'] = hyperparams['master_config'].get('load_render', False)
        config['xmlid'] = '{0}_{1}_{2}_{3}'.format(self.process_id, self.rank, no, nt)
        self.mjc_env = MJCEnv.load_config(config)
        self.targ_labels = {i: np.array(self.prob.END_TARGETS[i]) for i in range(len(self.prob.END_TARGETS))}
        if hasattr(self.prob, 'ALT_END_TARGETS'):
            self.alt_targ_labels = {i: np.array(self.prob.ALT_END_TARGETS[i]) for i in range(len(self.prob.ALT_END_TARGETS))}
        else:
            self.alt_targ_labels = self.targ_labels
        self.targ_labels.update({i: self.targets[0]['aux_target_{0}'.format(i-no)] for i in range(no, no+self.prob.n_aux)})


    def _sample_task(self, policy, condition, state, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True, task_f=None, hor=None, policies=None):
        x0 = state[self._x_data_idx[STATE_ENUM]].copy()
        task = tuple(task)
        onehot_task = tuple([val for val in task if np.isscalar(val)])
        plan = self.plans[onehot_task] if onehot_task in self.plans else self.plans[task[0]]

        if hor is None:
            hor = plan.horizon if task_f is None else max([p.horizon for p in list(self.plans.values())])

        self.T = hor
        sample = Sample(self)
        sample.init_t = 0
        col_ts = np.zeros(self.T)
        prim_choices = self.prob.get_prim_choices(self.task_list)
        sample.targets = self.target_vecs[condition].copy()
        n_steps = 0
        end_state = None
        cur_state = self.get_state() # x0
        sample.task = task

        self.fill_sample(condition, sample, cur_state.copy(), 0, task, fill_obs=True)
        for t in range(0, self.T):
            noise_full = np.zeros((self.dU,))
            self.fill_sample(condition, sample, cur_state.copy(), t, task, fill_obs=True)
            if task_f is not None:
                prev_task = task
                task = task_f(sample, t, task)
                onehot_task = tuple([val for val in task if np.isscalar(val)])
                self.fill_sample(condition, sample, cur_state, t, task, fill_obs=False)
                taskname = self.task_list[task[0]]
                if policies is not None: policy = policies[taskname]
                self.fill_sample(condition, sample, cur_state.copy(), t, task, fill_obs=False)

            prev_vals = {}
            if policies is not None and 'cont' in policies and \
               len(self.continuous_opts):
                prev_vals = self.fill_cont(policies['cont'], sample, t)

            if self.prob.N_HUMAN > 0:
                self.solve_humans(policy, task)

            sample.set(NOISE_ENUM, noise_full, t)

            U_full = policy.act(cur_state.copy(), sample.get_obs(t=t).copy(), t, noise_full)
            sample.set(ACTION_ENUM, U_full.copy(), t)

            U_nogrip = U_full.copy()
            for (pname, aname), inds in self.action_inds.items():
                if aname.find('grip') >= 0: U_nogrip[inds] = 0.

            if np.all(np.abs(U_nogrip)) < 1e-3:
                self._noops += 1
                self.eta_scale = 1. / np.log(self._noops+2)
            else:
                self._noops = 0
                self.eta_scale = 1.

            for enum, val in prev_vals.items():
                sample.set(enum, val, t=t)
            if len(self._prev_U): self._prev_U = np.r_[self._prev_U[1:], [U_full]]

            suc, col = self.run_policy_step(U_full, cur_state)
            col_ts[t] = col
            new_state = self.get_state()

            if len(self._x_delta)-1:
                self._x_delta = np.r_[self._x_delta[1:], [new_state]]

            if len(self._prev_task)-1:
                self._prev_task = np.r_[self._prev_task[1:], [sample.get_prim_out(t=t)]]

            if np.all(np.abs(cur_state - new_state) < 1e-3):
                sample.use_ts[t] = 0

            cur_state = new_state

        sample.end_state = self.get_state()
        sample.task_cost = self.goal_f(condition, sample.end_state)
        sample.prim_use_ts[:] = sample.use_ts[:]
        sample.col_ts = col_ts

        if len(self.continuous_opts):
            self.add_cont_sample(sample)

        return sample


    def run_policy_step(self, u, x):
        if not self._feasible:
            return False, 0

        for human in self.humans:
            if not self._eval_mode:
                self.mjc_env.physics.named.data.qvel[human][:] = 0.
            else:
                self.mjc_env.physics.named.data.qvel[human][:2] = self.human_trajs[human]
                self.mjc_env.physics.named.data.qvel[human][2:] = 0. 
        self.mjc_env.physics.forward()

        self._col = []
        poses = {}
        for pname, aname in self.state_inds:
            if aname != 'pose': continue
            if pname.find('box') < 0 or pname.find('can') < 0: continue
            poses[pname] = self.mjc_env.get_item_pos(pname)

        cmd_theta = u[self.action_inds['pr2', 'theta']][0]
        if ('pr2', 'pose') not in self.action_inds:
            cmd_vel = u[self.action_inds['pr2', 'vel']]
            self.mjc_env.set_user_data('vel', cmd_vel)
            cur_theta = x[self.state_inds['pr2', 'theta']][0]
            cmd_x, cmd_y = -cmd_vel*np.sin(cur_theta), cmd_vel*np.cos(cur_theta)
        else:
            cur_theta = x[self.state_inds['pr2', 'theta']][0]
            rel_x, rel_y = u[self.action_inds['pr2', 'pose']]
            cmd_x, cmd_y = rel_x, rel_y
            if LOCAL_FRAME:
                cmd_x, cmd_y = np.array([[np.cos(cur_theta), -np.sin(cur_theta)], \
                                         [np.sin(cur_theta), np.cos(cur_theta)]]).dot([rel_x, rel_y])

        if np.isnan(cmd_x): cmd_x = 0#np.random.normal(0.05)
        if np.isnan(cmd_y): cmd_y = 0#np.random.normal(0.05)
        if np.isnan(cmd_theta): cmd_theta = 0#np.random.normal(0.05)
        nsteps = int(max(abs(cmd_x), abs(cmd_y)) / self.vel_rat) + 1
        # nsteps = min(nsteps, 10)
        # gripper = u[self.action_inds['pr2', 'gripper']][0]
        # gripper = -0.1 if gripper < 0 else 0.1
        cur_x, cur_y, _ = self.mjc_env.get_item_pos('pr2')
        ctrl_vec = np.array([cur_x+cmd_x, cur_y+cmd_y, cur_theta+cmd_theta])
        for n in range(nsteps):
            x = cur_x + float(n)/nsteps * cmd_x
            y = cur_y + float(n)/nsteps * cmd_y
            theta = cur_theta + float(n)/nsteps * cmd_theta
            ctrl_vec = np.array([x, y, theta])
            self.mjc_env.step(ctrl_vec, mode='velocity', gen_obs=False)
        ctrl_vec = np.array([cur_x+cmd_x, cur_y+cmd_y, cur_theta+cmd_theta])
        #self.mjc_env.step(ctrl_vec, mode='velocity')
        #self.mjc_env.step(ctrl_vec, mode='velocity')
        self.mjc_env.step(ctrl_vec, mode='velocity')
        self.mjc_env.step(ctrl_vec, mode='velocity')

        new_poses = {}
        #for pname, aname in self.state_inds:
        #    if aname != 'pose': continue
        #    if pname.find('box') < 0 or pname.find('can') < 0: continue
        #    new_poses[pname] = self.mjc_env.get_item_pos(pname)

        #for pname in poses:
        #    if np.any(np.abs(poses[pname]-new_poses[pname])) > 5e-2:
        #        self._col.append(pname)
        #        if pname.find('box') > 0: self._feasible = False

        for human in self.humans:
            if np.linalg.norm(self.mjc_env.get_item_pos(human)[:2] - self.mjc_env.get_item_pos('pr2')[:2]) < 0.65:
                self._feasible = False

        col = 1 if len(self._col) > 0 else 0
        self._rew = self.reward()
        self._ret += self._rew
        return True, col


    def get_state(self):
        x = np.zeros(self.dX)
        for pname, attr in self.state_inds:
            if attr == 'pose':
                val = self.mjc_env.get_item_pos(pname)
                x[self.state_inds[pname, attr]] = val[:2]
            elif attr == 'rotation':
                val = self.mjc_env.get_item_rot(pname)
                x[self.state_inds[pname, attr]] = val
            # elif attr == 'gripper':
            #     vals = self.mjc_env.get_joints(['left_finger_joint','right_finger_joint'])
            #     val1 = vals['left_finger_joint']
            #     val2 = vals['right_finger_joint']
            #     val = (val1 + val2) / 2.
            #     x[self.state_inds[pname, attr]] = 0.1 if val > 0 else -0.1
            elif attr == 'theta':
                val = self.mjc_env.get_joints(['robot_theta'])
                val = val['robot_theta']
                x[self.state_inds[pname, 'theta']] = val
            elif attr == 'vel':
                val = self.mjc_env.get_user_data('vel', 0.)
                x[self.state_inds[pname, 'vel']] = val

        assert not np.any(np.isnan(x))
        return x.round(5)


    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        if task is None:
            task = list(self.plans.keys())[0]
        mp_state = mp_state.copy()
        onehot_task = tuple([val for val in task if np.isscalar(val)])
        plan = self.plans[onehot_task]
        ee_pose = mp_state[self.state_inds['pr2', 'pose']]
        if targets is None:
            targets = self.target_vecs[cond].copy()

        theta = mp_state[self.state_inds['pr2', 'theta']][0]
        while theta < -np.pi:
            theta += 2*np.pi
        while theta > np.pi:
            theta -= 2*np.pi

        if LOCAL_FRAME:
            rot = np.array([[np.cos(-theta), -np.sin(-theta)],
                            [np.sin(-theta), np.cos(-theta)]])
        else:
            rot = np.eye(2)

        sample.set(EE_ENUM, ee_pose, t)
        sample.set(THETA_ENUM, np.array([theta]), t)
        dirvec = np.array([-np.sin(theta), np.cos(theta)])
        sample.set(THETA_VEC_ENUM, dirvec, t)
        velx = self.mjc_env.physics.named.data.qvel['robot_x'][0]
        vely = self.mjc_env.physics.named.data.qvel['robot_y'][0]
        sample.set(VEL_ENUM, np.array([velx, vely]), t)
        sample.set(STATE_ENUM, mp_state, t)
        # sample.set(GRIPPER_ENUM, mp_state[self.state_inds['pr2', 'gripper']], t)
        if self.hist_len > 0:
            sample.set(TRAJ_HIST_ENUM, self._prev_U.flatten(), t)
            for human in self.humans:
                self._x_delta[:,self.state_inds[human, 'pose']] = 0.
            x_delta = self._x_delta[1:] - self._x_delta[:1]
            sample.set(STATE_DELTA_ENUM, x_delta.flatten(), t)
            sample.set(STATE_HIST_ENUM, self._x_delta.flatten(), t)
        if self.task_hist_len > 0:
            sample.set(TASK_HIST_ENUM, self._prev_task.flatten(), t)
        
        prim_choices = self.prob.get_prim_choices(self.task_list)

        task_ind = task[0]
        task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
        task_vec[task[0]] = 1.
        sample.task_ind = task[0]
        sample.set(TASK_ENUM, task_vec, t)

        #post_cost = self.cost_f(sample.get_X(t=t), task, cond, active_ts=(sample.T-1, sample.T-1), targets=targets)
        #done = np.ones(1) if post_cost == 0 else np.zeros(1)
        #sample.set(DONE_ENUM, done, t)
        sample.set(DONE_ENUM, np.zeros(1), t)
        sample.set(TASK_DONE_ENUM, np.array([1, 0]), t)
        grasp = np.array([0, -0.601])
        onehottask = tuple([val for val in task if np.isscalar(val)])
        sample.set(FACTOREDTASK_ENUM, np.array(onehottask), t)

        if OBJ_ENUM in prim_choices:
            obj_vec = np.zeros((len(prim_choices[OBJ_ENUM])), dtype='float32')
            obj_vec[task[1]] = 1.
            if self.task_list[task[0]].find('place') >= 0:
                obj_vec[:] = 1. / len(obj_vec)

            sample.obj_ind = task[1]
            obj_ind = task[1]
            obj_name = list(prim_choices[OBJ_ENUM])[obj_ind]
            sample.set(OBJ_ENUM, obj_vec, t)
            obj_pose = mp_state[self.state_inds[obj_name, 'pose']] - mp_state[self.state_inds['pr2', 'pose']]
            base_pos = obj_pose
            obj_pose = rot.dot(obj_pose)
            sample.set(OBJ_POSE_ENUM, obj_pose.copy(), t)
            sample.obj = task[1]
            if self.task_list[task[0]].find('move') >= 0:
                sample.set(END_POSE_ENUM, obj_pose, t)
                sample.set(REL_POSE_ENUM, base_pos, t)
                sample.set(ABS_POSE_ENUM, mp_state[self.state_inds[obj_name, 'pose']].copy(), t)

        if TARG_ENUM in prim_choices:
            targ_vec = np.zeros((len(prim_choices[TARG_ENUM])), dtype='float32')
            targ_vec[task[2]] = 1.
            if self.task_list[task[0]].find('move') >= 0:
                targ_vec[:] = 1. / len(targ_vec)
            sample.targ_ind = task[2]
            targ_ind = task[2]
            targ_name = list(prim_choices[TARG_ENUM])[targ_ind]
            sample.set(TARG_ENUM, targ_vec, t)
            targ_pose = targets[self.target_inds[targ_name, 'value']] - mp_state[self.state_inds['pr2', 'pose']]
            targ_off_pose = targets[self.target_inds[targ_name, 'value']] - mp_state[self.state_inds[obj_name, 'pose']]
            base_pos = targ_pose
            targ_pose = rot.dot(targ_pose)
            targ_off_pose = rot.dot(targ_off_pose)
            sample.set(TARG_POSE_ENUM, targ_pose.copy(), t)
            sample.targ = task[2]
            if self.task_list[task[0]].find('place') >= 0 or self.task_list[task[0]].find('transfer') >= 0:
                sample.set(END_POSE_ENUM, targ_pose, t)
                sample.set(REL_POSE_ENUM, base_pos, t)
                sample.set(ABS_POSE_ENUM, targets[self.target_inds[targ_name, 'value']].copy(), t)

        sample.set(TRUE_POSE_ENUM, sample.get(ABS_POSE_ENUM, t=t), t)
        if ABS_POSE_ENUM in prim_choices:
            ind = list(prim_choices.keys()).index(ABS_POSE_ENUM)
            if ind < len(task) and not np.isscalar(task[ind]):
                sample.set(ABS_POSE_ENUM, task[ind], t)
                sample.set(END_POSE_ENUM, rot.dot(task[ind] - mp_state[self.state_inds['pr2', 'pose']]), t)

        if REL_POSE_ENUM in prim_choices:
            ind = list(prim_choices.keys()).index(REL_POSE_ENUM)
            if ind < len(task) and not np.isscalar(task[ind]):
                sample.set(REL_POSE_ENUM, task[ind], t)
                sample.set(END_POSE_ENUM, rot.dot(task[ind]), t)

        if END_POSE_ENUM in prim_choices:
            ind = list(prim_choices.keys()).index(END_POSE_ENUM)
            if ind < len(task) and type(task[ind]) is not int:
                sample.set(END_POSE_ENUM, task[ind], t)

        sample.task = task
        sample.condition = cond
        sample.task_name = self.task_list[task[0]]
        sample.set(TARGETS_ENUM, targets.copy(), t)
        sample.set(GOAL_ENUM, np.concatenate([targets[self.target_inds['{0}_end_target'.format(o), 'value']] for o in prim_choices[OBJ_ENUM]]), t)
        if ONEHOT_GOAL_ENUM in self._hyperparams['sensor_dims']:
            sample.set(ONEHOT_GOAL_ENUM, self.onehot_encode_goal(sample.get(GOAL_ENUM, t)), t)
        sample.targets = targets.copy()

        for i, obj in enumerate(prim_choices[OBJ_ENUM]):
            sample.set(OBJ_ENUMS[i], mp_state[self.state_inds[obj, 'pose']], t)
            targ = targets[self.target_inds['{0}_end_target'.format(obj), 'value']]
            #sample.set(OBJ_DELTA_ENUMS[i], rot.dot(mp_state[self.state_inds[obj, 'pose']]-ee_pose), t)
            sample.set(OBJ_DELTA_ENUMS[i], mp_state[self.state_inds[obj, 'pose']]-ee_pose, t)
            sample.set(TARG_ENUMS[i], targ, t)
            #sample.set(TARG_DELTA_ENUMS[i], rot.dot(targ-mp_state[self.state_inds[obj, 'pose']]), t)
            sample.set(TARG_DELTA_ENUMS[i], targ-mp_state[self.state_inds[obj, 'pose']], t)

        if INGRASP_ENUM in self._hyperparams['sensor_dims']:
            vec = np.zeros(len(prim_choices[OBJ_ENUM]))
            for i, o in enumerate(prim_choices[OBJ_ENUM]):
                if np.all(np.abs(mp_state[self.state_inds[o, 'pose']] - mp_state[self.state_inds['pr2', 'pose']] - grasp) < NEAR_TOL):
                    vec[i] = 1.
            sample.set(INGRASP_ENUM, vec, t=t)

        if ATGOAL_ENUM in self._hyperparams['sensor_dims']:
            vec = np.zeros(len(prim_choices[OBJ_ENUM]))
            for i, o in enumerate(prim_choices[OBJ_ENUM]):
                if np.all(np.abs(mp_state[self.state_inds[o, 'pose']] - targets[self.target_inds['{0}_end_target'.format(o), 'value']]) < NEAR_TOL):
                    vec[i] = 1.
            sample.set(ATGOAL_ENUM, vec, t=t)

        if fill_obs:
            if LIDAR_ENUM in self._hyperparams['obs_include']:
                plan = list(self.plans.values())[0]
                set_params_attrs(plan.params, plan.state_inds, mp_state, 0)
                lidar = self.dist_obs(plan, 1)
                sample.set(LIDAR_ENUM, lidar.flatten(), t)

            if MJC_SENSOR_ENUM in self._hyperparams['obs_include']:
                plan = list(self.plans.values())[0]
                sample.set(MJC_SENSOR_ENUM, self.mjc_env.get_sensors(), t)

            if IM_ENUM in self._hyperparams['obs_include'] or \
               IM_ENUM in self._hyperparams['prim_obs_include'] or \
               IM_ENUM in self._hyperparams['cont_obs_include']:
                im = self.mjc_env.render(height=self.image_height, width=self.image_width, view=self.view)
                im = (im - 128.) / 128.
                sample.set(IM_ENUM, im.flatten().astype(np.float32), t)

    
    def reset_mjc_env(x, targets, draw_targets=True):
        # this is elsewhere
        pass


    def reset_to_sample(self, sample):
        self.reset_to_state(sample.get_X(sample.T-1))


    def reset(self, m):
        self.reset_to_state(self.x0[m])


    def reset_to_state(self, x):
        mp_state = x[self._x_data_idx[STATE_ENUM]]
        self._done = 0.
        self._ret = 0.
        self._rew = 0.
        self._feasible = True
        self._prev_U = np.zeros((self.hist_len, self.dU))
        self._x_delta = np.zeros((self.hist_len+1, self.dX))
        self._x_delta[:] = x.reshape((1,-1))
        self.eta_scale = 1.
        self._noops = 0
        self.mjc_env.reset()
        xval, yval = mp_state[self.state_inds['pr2', 'pose']]
        # grip = x[self.state_inds['pr2', 'gripper']][0]
        theta = x[self.state_inds['pr2', 'theta']][0]
        self.mjc_env.set_user_data('vel', 0.)
        self.mjc_env.set_joints({'robot_x': xval, 'robot_y': yval, 'robot_theta': theta}, forward=False)
        for param_name, attr in self.state_inds:
            if param_name == 'pr2': continue
            if attr == 'pose':
                pos = mp_state[self.state_inds[param_name, 'pose']].copy()
                self.mjc_env.set_item_pos(param_name, np.r_[pos, 0.5], forward=False)
                if param_name.find('can') >= 0:
                    targ = self.target_vecs[0][self.target_inds['{0}_end_target'.format(param_name), 'value']]
                    self.mjc_env.set_item_pos('{0}_end_target'.format(param_name), np.r_[targ, -0.15], forward=False)
        self.mjc_env.physics.data.qvel[:] = 0.
        self.mjc_env.physics.forward()


    def set_to_targets(self, condition=0):
        prim_choices = self.prob.get_prim_choices(self.task_list)
        objs = prim_choices[OBJ_ENUM]
        for obj_name in objs:
            self.mjc_env.set_item_pos(obj_name, np.r_[self.targets[condition]['{0}_end_target'.format(obj_name)], 0], forward=False)
        self.mjc_env.physics.forward()


    def get_mjc_obs(self, x):
        self.reset_to_state(x)
        # return self.mjc_env.get_obs(view=False)
        return self.mjc_env.render()


    def sample_optimal_trajectory(self, state, task, condition, opt_traj=[], traj_mean=[], targets=[], run_traj=True):
        if not len(opt_traj):
            return self.solve_sample_opt_traj(state, task, condition, traj_mean, targets=targets)
        if not len(targets):
            old_targets = self.target_vecs[condition]
        else:
            old_targets = self.target_vecs[condition]
            for tname, attr in self.target_inds:
                self.targets[condition][tname] = targets[self.target_inds[tname, attr]]
            self.target_vecs[condition] = targets

        exclude_targets = []
        onehot_task = tuple([val for val in task if np.isscalar(val)])
        plan = self.plans[onehot_task]
        if run_traj:
            sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, opt_traj), condition, state, task, noisy=False, skip_opt=True, hor=len(opt_traj))
        else:
            self.T = plan.horizon
            sample = Sample(self)
            for t in range(len(opt_traj)-1):
                pos = opt_traj[t][self.state_inds['pr2', 'pose']]
                pos_2 = opt_traj[t+1][self.state_inds['pr2', 'pose']]
                theta = opt_traj[t][self.state_inds['pr2', 'theta']]
                theta_2 = opt_traj[t+1][self.state_inds['pr2', 'theta']]
                vel = opt_traj[t+1][self.state_inds['pr2', 'vel']]
                # grip = opt_traj[t][self.state_inds['pr2', 'gripper']]
                U = np.zeros(self.dU)
                # U[self.action_inds['pr2', 'pose']] = pos_2 - pos
                U[self.action_inds['pr2', 'vel']] = vel
                U[self.action_inds['pr2', 'theta']] = theta_2 - theta
                # U[self.action_inds['pr2', 'gripper']] = grip
                sample.set(ACTION_ENUM, U, t=t)
                self.reset_to_state(opt_traj[t])
                self.fill_sample(condition, sample, opt_traj[t], t, task, fill_obs=True, targets=targets)
            if len(opt_traj)-1 < sample.T:
                for j in range(len(opt_traj)-1, sample.T):
                    sample.set(ACTION_ENUM, np.zeros_like(U), t=j)
                    self.reset_to_state(opt_traj[-1])
                    self.fill_sample(condition, sample, opt_traj[-1], j, task, fill_obs=True, targets=targets)
            sample.use_ts[-1] = 0.
            sample.prim_use_ts[-1] = 0.
            sample.prim_use_ts[len(opt_traj)-1:] = 0.
            sample.use_ts[len(opt_traj)-1:] = 0.
            sample.end_state = opt_traj[-1].copy()
            sample.set(NOISE_ENUM, np.zeros((sample.T, self.dU)))
            sample.task_cost = self.goal_f(condition, sample.end_state)
            sample.col_ts = np.zeros(sample.T)
        sample.set_ref_X(sample.get_X())
        sample.set_ref_U(sample.get_U())

        # for t in range(sample.T):
        #     if np.all(np.abs(sample.get(ACTION_ENUM, t=t))) < 1e-3:
        #         sample.use_ts[t] = 0.

        self.target_vecs[condition] = old_targets
        for tname, attr in self.target_inds:
            self.targets[condition][tname] = old_targets[self.target_inds[tname, attr]]
        # self.optimal_samples[self.task_list[task[0]]].append(sample)
        return sample


    # def solve_sample_opt_traj(self, state, task, condition, traj_mean=[], inf_f=None, mp_var=0, targets=[], x_only=False, t_limit=60, n_resamples=5, out_coeff=None, smoothing=False, attr_dict=None):
    #     success = False
    #     old_targets = self.target_vecs[condition]
    #     if not len(targets):
    #         targets = self.target_vecs[condition]
    #     else:
    #         self.target_vecs[condition] = targets.copy()
    #         for tname, attr in self.target_inds:
    #             self.targets[condition][tname] = targets[self.target_inds[tname, attr]]

    #     x0 = state[self._x_data_idx[STATE_ENUM]]

    #     failed_preds = []
    #     iteration = 0
    #     iteration += 1
    #     onehot_task = tuple([val for val in task if np.isscalar(val)])
    #     plan = self.plans[onehot_task]
    #     prim_choices = self.prob.get_prim_choices(self.task_list)
    #     # obj_name = prim_choices[OBJ_ENUM][task[1]]
    #     # targ_name = prim_choices[TARG_ENUM][task[2]]
    #     set_params_attrs(plan.params, plan.state_inds, x0, 0)

    #     for param_name in plan.params:
    #         param = plan.params[param_name]
    #         if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
    #             param.pose[:, 0] = x0[self.state_inds[param_name, 'pose']]
    #             plan.params['{0}_init_target'.format(param_name)].value[:,0] = param.pose[:,0]

    #     for tname, attr in self.target_inds:
    #         getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]

    #     grasp = np.array([0, -0.601])
    #     if GRASP_ENUM in prim_choices:
    #         grasp = self.set_grasp(grasp, task[3])

    #     plan.params['pr2'].pose[:, 0] = x0[self.state_inds['pr2', 'pose']]
    #     plan.params['pr2'].gripper[:, 0] = x0[self.state_inds['pr2', 'gripper']]
    #     plan.params['obs0'].pose[:] = plan.params['obs0'].pose[:,:1]

    #     run_solve = True

    #     plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
    #     for param in list(plan.params.values()):
    #         for attr in param._free_attrs:
    #             if np.any(np.isnan(getattr(param, attr)[:,0])):
    #                 getattr(param, attr)[:,0] = 0

    #     old_out_coeff = self.solver.strong_transfer_coeff
    #     if out_coeff is not None:
    #         self.solver.strong_transfer_coeff = out_coeff
    #     try:
    #         if smoothing:
    #             success = self.solver.quick_solve(plan, n_resamples=n_resamples, traj_mean=traj_mean, attr_dict=attr_dict)
    #         elif run_solve:
    #             success = self.solver._backtrack_solve(plan, n_resamples=n_resamples, traj_mean=traj_mean, inf_f=inf_f, task=task, time_limit=t_limit)
    #         else:
    #             success = False
    #     except Exception as e:
    #         print(e)
    #         # traceback.print_exception(*sys.exc_info())
    #         success = False

    #     self.solver.strong_transfer_coeff = old_out_coeff

    #     try:
    #         if not len(failed_preds):
    #             for action in plan.actions:
    #                 failed_preds += [(pred, t) for negated, pred, t in plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)]
    #     except:
    #         failed_preds += ['Nan in pred check for {0}'.format(action)]

    #     traj = np.zeros((plan.horizon, self.symbolic_bound))
    #     for pname, aname in self.state_inds:
    #         if plan.params[pname].is_symbol(): continue
    #         inds = self.state_inds[pname, aname]
    #         for t in range(plan.horizon):
    #             traj[t][inds] = getattr(plan.params[pname], aname)[:,t]

    #     class _optimal_pol:
    #         def act(self, X, O, t, noise):
    #             U = np.zeros((plan.dU), dtype=np.float32)
    #             if t < len(traj)-1:
    #                 for param, attr in plan.action_inds:
    #                     if attr == 'pose':
    #                         U[plan.action_inds[param, attr]] = traj[t+1][plan.state_inds[param, attr]] - X[plan.state_inds[param, attr]]
    #                     elif attr == 'gripper':
    #                         U[plan.action_inds[param, attr]] = traj[t][plan.state_inds[param, attr]]
    #                     elif attr == 'theta':
    #                         U[plan.action_inds[param, attr]] = traj[t+1][plan.state_inds[param, attr]] - traj[t][plan.state_inds[param, attr]]
    #                     elif attr == 'vel':
    #                         U[plan.action_inds[param, attr]] = traj[t+1][plan.state_inds[param, attr]]
    #                     else:
    #                         raise NotImplementedError
    #             if np.any(np.isnan(U)):
    #                 if success: print(('NAN in {0} plan act'.format(success)))
    #                 U[:] = 0.
    #             return U
    #     sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, traj), condition, state, task, noisy=False, skip_opt=True)
    #     # sample = self.sample_task(optimal_pol(), condition, state, task, noisy=False, skip_opt=True)

    #     # for t in range(sample.T):
    #     #     if np.all(np.abs(sample.get(ACTION_ENUM, t=t))) < 1e-3: sample.use_ts[t] = 0.

    #     traj = sample.get(STATE_ENUM)
    #     for param_name, attr in self.state_inds:
    #         param = plan.params[param_name]
    #         if param.is_symbol(): continue
    #         diff = traj[:, self.state_inds[param_name, attr]].T - getattr(param, attr)
    #         # if np.any(np.abs(diff) > 1e-3): print(diff, param_name, attr, 'ERROR IN OPT ROLLOUT')

    #     # self.optimal_samples[self.task_list[task[0]]].append(sample)
    #     # print(sample.get_X())
    #     if not smoothing and self.debug:
    #         if not success:
    #             sample.use_ts[:] = 0.
    #             print(('Failed to plan for: {0} {1} smoothing? {2} {3}'.format(task, failed_preds, smoothing, state)))
    #             print('FAILED PLAN')
    #         else:
    #             print(('SUCCESSFUL PLAN for {0}'.format(task)))
    #     # else:
    #     #     print('Plan success for {0} {1}'.format(task, state))
    #     return sample, failed_preds, success


    def retime_traj(self, traj, vel=0.3, inds=None, minpts=10):
        new_traj = []
        if len(np.shape(traj)) == 2:
            traj = [traj]
        for step in traj:
            xpts = []
            fpts = []
            # grippts= []
            d = 0
            if inds is None:
                #inds = np.r_[self.state_inds['pr2', 'vel'], \
                #             self.state_inds['pr2', 'pose']]
                inds = self.state_inds['pr2', 'pose']
            for t in range(len(step)):
                xpts.append(d)
                fpts.append(step[t])
                # grippts.append(step[t][self.state_inds['pr2', 'gripper']])
                if t < len(step) - 1:
                    disp = np.linalg.norm(step[t+1][inds] - step[t][inds])
                    d += disp
            assert not np.any(np.isnan(xpts))
            assert not np.any(np.isnan(fpts))
            interp = scipy.interpolate.interp1d(xpts, fpts, axis=0, fill_value='extrapolate')
            # grip_interp = scipy.interpolate.interp1d(np.array(xpts), grippts, kind='previous', bounds_error=False, axis=0)

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
                    # elif xpts[i] - xpts[i-1] <= 1e-6:
                    #     continue
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
            # grip_out = grip_interp(x)
            # out[:, self.state_inds['pr2', 'gripper']] = grip_out
            out[0] = step[0]
            for pt, val in fix_pts:
                out[pt] = val
            out[-1] = step[-1]
            #out = np.r_[out, [out[-1]]]
            if len(new_traj):
                new_traj = np.r_[new_traj, out]
            else:
                new_traj = out
            if np.any(np.isnan(out)): print(('NAN in out', out, x))
        return new_traj


    def set_symbols(self, plan, task, anum=0, cond=0, targets=None, st=0):
        act_st, et = plan.actions[anum].active_timesteps
        st = max(act_st, st)
        if targets is None:
            targets = self.target_vecs[cond].copy()
        prim_choices = self.prob.get_prim_choices(self.task_list)
        act = plan.actions[anum]
        params = act.params
        if self.task_list[task[0]] == 'moveto':
            params[3].value[:,0] = params[0].pose[:,st]
            #params[2].value[:,0] = params[1].pose[:,st]
        elif self.task_list[task[0]] == 'transfer':
            params[1].value[:,0] = params[0].pose[:,st]
            #params[6].value[:,0] = params[3].pose[:,st]
        elif self.task_list[task[0]] == 'place':
            params[1].value[:,0] = params[0].pose[:,st]
            #params[6].value[:,0] = params[3].pose[:,st]

        for tname, attr in self.target_inds:
            getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]

        for pname in plan.params:
            if '{0}_init_target'.format(pname) in plan.params:
                plan.params['{0}_init_target'.format(pname)].value[:,0] = plan.params[pname].pose[:,st]


    def goal(self, cond, targets=None):
        # if self.goal_type == 'move':
        #     assert ('can1', 'pose') not in self.state_inds
        #     return '(NearGraspAngle  pr2 can0) '
        # if targets is None:
        #     targets = self.target_vecs[cond]
        # prim_choices = self.prob.get_prim_choices(self.task_list)
        # goal = ''
        # for i, obj in enumerate(prim_choices[OBJ_ENUM]):
        #     targ = targets[self.target_inds['{0}_end_target'.format(obj), 'value']]
        #     targ_labels = self.targ_labels if not self._eval_mode else self.alt_targ_labels
        #     for ind in targ_labels:
        #         if np.all(np.abs(targ - targ_labels[ind]) < NEAR_TOL):
        #             goal += '(Near {0} end_target_{1}) '.format(obj, ind)
        #             break
        return '(RobotNear pr2 can0) '
    
    # get distance of robot to Can
    def goal_f(self, condition, state, targets=None, cont=False, anywhere=False, tol=0.4):
        if targets is None:
            targets = self.target_vecs[condition]
        cost = self.prob.NUM_OBJS
        alldisp = 0
        plan = list(self.plans.values())[0]
        no = self._hyperparams['num_objs']
        if len(np.shape(state)) < 2:
            state = [state]
        for param in list(plan.params.values()):
            if param._type == 'Can':
                # if anywhere:
                #     vals = [targets[self.target_inds[key, 'value']] for key, _ in self.target_inds if key.find('end_target') >= 0]
                # else:
                #     vals = [targets[self.target_inds['{0}_end_target'.format(param.name), 'value']]]
                # dist = np.inf 
                # disp = None
                # for x in state:
                #     if self.goal_type == 'moveto':
                #         vals = [x[self.state_inds['pr2', 'pose']]]
                #     for val in vals:
                #         curdisp = x[self.state_inds[param.name, 'pose']] - val
                #         curdist = np.linalg.norm(curdisp)
                #         if curdist < dist:
                #             disp = curdisp
                #             dist = curdist
                # # np.sum((state[self.state_inds[param.name, 'pose']] - self.targets[condition]['{0}_end_target'.format(param.name)])**2)
                # # cost -= 1 if dist < 0.3 else 0
                # alldisp += curdist # np.linalg.norm(disp)
                for x in state:
                    disp = x[self.state_inds['pr2', 'pose']] - x[self.state_inds['can0', 'pose']]
                cost -= 1 if np.all(np.abs(disp) < tol) else 0

        if cont: return alldisp / float(no)
        # return cost / float(self.prob.NUM_OBJS)
        return 1. if cost > 0 else 0.



    def get_annotated_image(self, s, t, cam_id=None):
        if cam_id is None: cam_id = self.camera_id
        x = s.get_X(t=t)
        task = s.get(FACTOREDTASK_ENUM, t=t).astype(int)
        pos = s.get(TRUE_POSE_ENUM, t=t)
        #pos = str(pos.round(2))[1:-1]
        predpos = s.get(ABS_POSE_ENUM, t=t)
        #predpos = str(predpos.round(2))[1:-1]
        precost = round(self.precond_cost(s, tuple(task), t), 5)
        postcost = round(self.postcond_cost(s, tuple(task), t, x0=s.base_x), 5)
        offset = str((pos - predpos).round(2))[1:-1]
        act = str(s.get(ACTION_ENUM, t=t).round(3))[1:-1]
        textover1 = self.mjc_env.get_text_overlay(body='Task: {0} {1}'.format(task, act))
        textover2 = self.mjc_env.get_text_overlay(body='{0: <6} {1: <6} Error: {2}'.format(precost, postcost, offset), position='bottom left')
        self.reset_to_state(x)
        im = self.mjc_env.render(camera_id=cam_id, height=self.image_height, width=self.image_width, view=False, overlays=(textover1, textover2))
        return im


    def center_cont(self, abs_val, x):
        theta = x[:, self.state_inds['pr2', 'theta']]
        ee_pos = x[:, self.state_inds['pr2', 'pose']]
        new_val = []
        for t in range(len(abs_val)):
            if LOCAL_FRAME:
                rot = theta[t,0]
                rot = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                new_val.append(rot.dot(abs_val[t] - ee_pos[t]))
            else:
                new_val.append(abs_val[t] - ee_pos[t])
        return np.array(new_val)


    # useless here, no gripper
    def clip_state(self, x):
        x = x.copy()
        # inds = self.state_inds['pr2', 'gripper']
        # val = x[inds][0]
        # x[inds] = GRIP_VAL if x[inds][0] >= 0 else -GRIP_VAL
        return x

    
    def fill_cont(self, policy, sample, t):
        vals = policy.act(sample.get_X(t=t), sample.get_cont_obs(t=t), t)
        old_vals = {}
        for ind, enum in enumerate(self.continuous_opts):
            old_vals[enum] = sample.get(enum, t=t).copy()
            sample.set(enum, vals[ind], t=t)
            if enum is utils.ABS_POSE_ENUM:
                mp_state = sample.get_X(t=t)
                theta = mp_state[self.state_inds['pr2', 'theta']][0]
                if LOCAL_FRAME:
                    rot = np.array([[np.cos(-theta), -np.sin(-theta)],
                                    [np.sin(-theta), np.cos(-theta)]])
                else:
                    rot = np.eye(2)
                old_vals[END_POSE_ENUM] = sample.get(END_POSE_ENUM, t=t).copy()
                sample.set(utils.END_POSE_ENUM, rot.dot(vals[ind] - mp_state[self.state_inds['pr2', 'pose']]), t)
                sample.set(utils.TRUE_POSE_ENUM, vals[ind], t)
        return old_vals


    def feasible_state(self, x, targets):
        return self._feasible


    def human_cost(self, x, goal_wt=1e1, col_wt=2e0, rcol_wt=5e0):
        cost = 0
        for human in self.humans:
            hpos = x[self.state_inds[human, 'pose']]
            cost += goal_wt * np.linalg.norm(hpos-self.humans[human])

            for (pname, aname), inds in self.state_inds.items():
                if pname == human: continue
                if aname != 'pose': continue
                if pname.find('pr2') >= 0 and np.linalg.norm(x[inds]-hpos) < 0.8:
                    cost += rcol_wt
                elif pname.find('human') >= 0 and np.linalg.norm(x[inds]-hpos) < 0.8:
                    cost += rcol_wt
                elif pname.find('can') >= 0:
                    cost -= col_wt * np.linalg.norm(x[inds]-hpos)
                #elif pname.find('pr2') >= 0:
                #    cost -= col_wt * np.linalg.norm(x[inds]-hpos)
        return cost


    def solve_humans(self, policy, task, hor=2, N=30):
        if not self._eval_mode or not self.prob.N_HUMAN:
            for n in range(self.prob.N_HUMAN):
                self.human_trajs['human{}'.format(n)] = np.zeros(2)
            return

        for human_id in range(self.prob.N_HUMAN):
            if np.random.uniform() < 0.05:
                self.humans['human{}'.format(human_id)] = HUMAN_TARGS[np.random.randint(len(HUMAN_TARGS))]

        old_feas = self._feasible
        self._feasible = True
        init_t = time.time()
        qpos = self.mjc_env.physics.data.qpos.copy()
        qvel = self.mjc_env.physics.data.qvel.copy()
        init_state = self.get_state()
        trajs = []
        sample = Sample(self)
        for _ in range(N):
            self.mjc_env.physics.data.qpos[:] = qpos.copy()
            self.mjc_env.physics.data.qvel[:] = qvel.copy()
            self.mjc_env.physics.forward()
            #traj = np.random.uniform(-2, 2, (self.prob.N_HUMAN, hor, 2))
            traj = np.random.uniform(-1, 1, (self.prob.N_HUMAN, hor, 2))
            traj[:,:,1] *= 2
            cost = 0
            for t in range(hor):
                x = self.get_state()
                self.fill_sample(0, sample, x, t, task, fill_obs=True)
                act = policy.act(sample.get_X(t=t), sample.get_obs(t=t), t)
                for n in range(self.prob.N_HUMAN):
                    self.human_trajs['human{}'.format(n)] = traj[n, t]
                self.run_policy_step(act, x)
                self._feasible = True
                goal_wt = 0 if t < hor-1 else hor * 1e0
                cost += self.human_cost(x, goal_wt=goal_wt)
            trajs.append((cost, traj[:,0]))

        self.mjc_env.physics.data.qpos[:] = qpos
        self.mjc_env.physics.data.qvel[:] = qvel
        self.mjc_env.physics.forward()
        cur_cost, cur_traj = trajs[0]
        for cost, traj in trajs:
            if cost < cur_cost:
                cur_cost = cost
                cur_traj = traj

        for n in range(self.prob.N_HUMAN):
            self.human_trajs['human{}'.format(n)] = traj[n]

        #print('TIME TO GET HUMAN ACTS FOR {} N {} HOR: {}'.format(N, hor, time.time() - init_t))
        self._feasible = old_feas
        return traj


    def reward(self, x=None, targets=None, center=False, gamma=0.9):
        if x is None: x = self.get_state()
        if targets is None: targets = self.target_vecs[0]
        l2_coeff = 2e-2 # 1e-2
        log_coeff = 1.
        obj_coeff = 1.
        targ_coeff = 1.

        opts = self.prob.get_prim_choices(self.task_list)
        rew = 0
        eeinds = self.state_inds['pr2', 'pose']
        ee_pos = x[eeinds]
        ee_theta = x[self.state_inds['pr2', 'theta']][0]
        dist = 0.61
        tol_coeff = 0.8
        # grip_pos = ee_pos + [-dist*np.sin(ee_theta), dist*np.cos(ee_theta)]
        max_per_obj = 3.2
        info_per_obj = []
        min_dist = np.inf
        for opt in opts[OBJ_ENUM]:
            xinds = self.state_inds[opt, 'pose']
            targinds = self.target_inds['{}_end_target'.format(opt), 'value']
            dist_to_targ = np.linalg.norm(x[xinds]-targets[targinds])
            dist_to_self= np.linalg.norm(ee_pos - x[xinds])

            if dist_to_targ < tol_coeff*NEAR_TOL:
                rew += 2 * (obj_coeff + targ_coeff) * max_per_obj / (1-gamma)
                info_per_obj.append((np.inf,0))
            else:
                # TODO: quick janky fix, just make ee_pos 
                self_l2_term = -l2_coeff * dist_to_self**2
                self_log_term = -np.log(log_coeff * dist_to_self + 1e-6)
                targ_l2_term = -l2_coeff * dist_to_targ**2
                targ_log_term = -log_coeff * np.log(dist_to_targ + 1e-6)
                self_obj_rew = obj_coeff * np.min([self_l2_term + self_log_term, max_per_obj])
                targ_obj_rew = targ_coeff * np.min([targ_l2_term + targ_log_term, max_per_obj])
                rew += targ_obj_rew # Always penalize obj to target distance
                min_dist = np.min([min_dist, dist_to_self])
                info_per_obj.append((dist_to_self, self_obj_rew)) # Only penalize closest object to self 

        for dist, obj_rew in info_per_obj:
            if dist <= min_dist:
                rew += obj_rew
                break

        return rew / 1e1

    def check_target(self, targ):
        targ_labels = self.targ_labels if not self._eval_mode else self.alt_targ_labels
        vec = np.zeros(len(list(targ_labels.keys())))
        for ind in targ_labels:
            if np.all(np.abs(targ - targ_labels[ind]) < NEAR_TOL):
                vec[ind] = 1.
                break
        return vec
