from abc import ABCMeta, abstractmethod
import copy
import ctypes
import itertools
import pickle as pickle
import random
import sys
import time
import traceback

import numpy as np
import scipy.interpolate

import xml.etree.ElementTree as xml

from sco_py.expr import *

import opentamp.core.util_classes.common_constants as common_const
import opentamp.pma.backtrack_ll_solver_OSQP as bt_ll
from opentamp.pma.pr_graph import *
import pybullet as p

from opentamp.core.util_classes.openrave_body import OpenRAVEBody
from opentamp.policy_hooks.agent import Agent
from opentamp.policy_hooks.sample import Sample
from opentamp.policy_hooks.sample_list import SampleList
from opentamp.policy_hooks.save_video import save_video
from opentamp.policy_hooks.utils.policy_solver_utils import *
import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.policy_hooks.utils.tamp_eval_funcs import *
from opentamp.policy_hooks.utils.load_task_definitions import *


MAX_SAMPLELISTS = 1000
MAX_TASK_PATHS = 1000
ROLL_TOL = 1e-3

ACTION_SCALE = 1


class OptimalPolicy:
    def __init__(self, dU, action_inds, state_inds, opt_traj):
        self.dU = dU
        self.action_inds = action_inds
        self.state_inds = state_inds
        self.opt_traj = opt_traj


    ## optimal policy simply emulates the actions taken along the optimal trajectory, as given here
    def act(self, X, O, t, noise=None):
        u = np.zeros(self.dU)
        for param, attr in self.action_inds:
            if attr == 'pose':
                if t < len(self.opt_traj) - 1:
                    u[self.action_inds[param, attr]] = (self.opt_traj[t + 1, self.action_inds[param, attr]] - self.opt_traj[t, self.action_inds[param, attr]]) * ACTION_SCALE
                else:
                    u[self.action_inds[param, attr]] = (self.opt_traj[-1, self.action_inds[param, attr]] - self.opt_traj[-2, self.action_inds[param, attr]]) * ACTION_SCALE
            else:
                if t < len(self.opt_traj) - 1:
                    u[self.action_inds[param, attr]] = (self.opt_traj[t + 1, self.action_inds[param, attr]]) * ACTION_SCALE
                else:
                    u[self.action_inds[param, attr]] = (self.opt_traj[-1, self.action_inds[param, attr]]) * ACTION_SCALE
        return u


    def is_initialized(self):
        return True

class OptimalAbsolutePolicy:
    def __init__(self, dU, action_inds, state_inds, opt_traj):
        self.dU = dU
        self.action_inds = action_inds
        self.state_inds = state_inds
        self.opt_traj = opt_traj


    ## optimal policy simply emulates the actions taken along the optimal trajectory, as given here
    def act(self, X, O, t, noise=None):
        u = np.zeros(self.dU)
        for param, attr in self.action_inds:
            if t < len(self.opt_traj) - 1:
                u[self.action_inds[param, attr]] = (self.opt_traj[t + 1, self.action_inds[param, attr]]) * ACTION_SCALE
            else:
                u[self.action_inds[param, attr]] = (self.opt_traj[-1, self.action_inds[param, attr]]) * ACTION_SCALE
        return u


    def is_initialized(self):
        return True


class OptimalTrigPolicy:
    def __init__(self, dU, action_inds, state_inds, opt_traj):
        self.dU = dU
        self.action_inds = action_inds
        self.state_inds = state_inds
        self.opt_traj = opt_traj

    def compute_angle(self, arr):
        if np.abs(arr[0]) < 0.01:
            return np.pi/2 if arr[0]*arr[1] > 0 else -np.pi/2
        elif arr[0] > 0:
            return np.arctan(arr[1]/arr[0])
        else:
            if arr[1] > 0:
                return np.arctan(arr[1]/arr[0]) + np.pi
            else:
                return np.arctan(arr[1]/arr[0]) - np.pi


    ## optimal policy simply emulates the actions taken along the optimal trajectory, as given here
    def act(self, X, O, t, noise=None):
        u = np.zeros(self.dU)
        for param, attr in self.action_inds:
            if attr == 'pose':
                if t < len(self.opt_traj) - 1:
                    relative_vec = (self.opt_traj[t + 1, self.action_inds[param, attr]] - self.opt_traj[t, self.action_inds[param, attr]]) * ACTION_SCALE
                else:
                    relative_vec = (self.opt_traj[-1, self.action_inds[param, attr]] - self.opt_traj[-2, self.action_inds[param, attr]]) * ACTION_SCALE
                u[self.action_inds[param, attr]] = np.array([np.linalg.norm(relative_vec, ord=2), self.compute_angle(relative_vec)])
            else:
                if t < len(self.opt_traj) - 1:
                    u[self.action_inds[param, attr]] = (self.opt_traj[t + 1, self.action_inds[param, attr]]) * ACTION_SCALE
                else:
                    u[self.action_inds[param, attr]] = (self.opt_traj[-1, self.action_inds[param, attr]]) * ACTION_SCALE
        return u


    def is_initialized(self):
        return True


class TAMPAgent(Agent, metaclass=ABCMeta):
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        self.config = hyperparams

        self.prob = hyperparams['prob']
        self.master_config = hyperparams['master_config']
        self.rank = hyperparams['master_config'].get('rank', 0)
        self.process_id = self.master_config['id']

        self._setup_exp_info()
        self.policies = {task: None for task in self.task_list}
        self.eta_scale = 1.
        self._noops = 0.
        self.optimal_samples = {task: [] for task in self.task_list}
        self.cont_samples = []
        self.task_paths = []

        self._setup_traj_opt()
        self._setup_tracking_info()


    def _setup_exp_info(self):
        self.sensor_dims = self.config['sensor_dims']
        self.task_list = self.config['task_list']
        self.task_encoding = self.config['task_encoding']
        self._samples = {task: [] for task in self.task_list}
        self.state_inds = self.config['state_inds']
        self.action_inds = self.config['action_inds']
        self.image_width = self.config.get('image_width', utils.IM_W)
        self.image_height = self.config.get('image_height', utils.IM_H)
        self.image_channels = self.config.get('image_channels', utils.IM_C)
        self.symbolic_bound = self.config['symbolic_bound']
        self.rollout_seed = self.config['rollout_seed']
        self.num_objs = self.config.get('num_objs', 1)
        self.rlen = self.config.get('rlen', 4 + 2 * self.num_objs * len(self.task_list))
        self.hor = self.config.get('horizon', 40)
        # self.hor = 20 ## add 5 more steps for rollout
        self._eval_mode = False
        self.retime = self.config['master_config'].get('retime', False)
        if self.retime: self.rlen *= 2
        self.init_vecs = self.config['x0']
        self.x0 = [x[:self.symbolic_bound] for x in self.init_vecs]
        self.targets = self.config['targets']
        self.target_dim = self.config['target_dim']
        self.target_inds = self.config['target_inds']
        self.target_vecs = []
        self.n_dirs = self.config['n_dirs']
        self.seed = 1234
        self.prim_dims = self.config['prim_dims']
        self.prim_dims_keys = list(self.prim_dims.keys())
        self.permute_hl = self.master_config.get('permute_hl', False)
        self.opt_wt = self.master_config['opt_wt']
        self.incl_init_obs = self.master_config.get('incl_init_obs', False)
        self.incl_trans_obs = self.master_config.get('incl_trans_obs', False)
        self.incl_grip_obs = self.master_config.get('incl_grip_obs', False)
        self.view = self.config['master_config'].get('view', False)
        self.camera_id = 0
        self.goal_type = self.master_config.get('goal_type', 'default')
        for condition in range(len(self.x0)):
            target_vec = np.zeros((self.target_dim,))
            # TODO -- FIX TARGET LOGIC
            for target_name in self.targets[condition]:
                if (target_name, 'value') in self.target_inds:
                    target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
            self.target_vecs.append(target_vec)
        self.cur_state = self.x0[0]
        self.discrete_prim = self.config.get('discrete_prim', True)
        self.swap = self.config['master_config'].get('swap', False)
        if self.config['master_config'].get('absolute_policy', False):
            self.optimal_pol_cls = OptimalAbsolutePolicy
        elif self.config['master_config'].get('trig_policy', False):
            self.optimal_pol_cls = OptimalTrigPolicy
        else:
            self.optimal_pol_cls = OptimalPolicy
        self.hist_len = self.config['hist_len']
        self.task_hist_len = self.config['master_config'].get('task_hist_len', 0)
        self._prev_U = np.zeros((self.hist_len, self.dU))
        self._x_delta = np.zeros((self.hist_len+1, self.dX))
        self._prev_task = np.zeros((self.task_hist_len, self.dPrimOut))


    def _setup_traj_opt(self):
        plans, _, env = self.prob.get_plans()
        self.plans = plans
        self.openrave_bodies = self._hyperparams['openrave_bodies']
        self.env = env
        self.plans_list = list(self.plans.values())

        bt_ll.COL_COEFF = self.master_config['col_coeff']
        self.solver = self._hyperparams['mp_solver_type'](self._hyperparams)
        if 'll_solver_type' in self._hyperparams['master_config']:
            self.ll_solver = self._hyperparams['master_config']['ll_solver_type'](self._hyperparams)
        else:
            self.ll_solver = self._hyperparams['mp_solver_type'](self._hyperparams)

        self.traj_smooth = self.master_config['traj_smooth']
        self.hl_solver = get_hl_solver(self.master_config['meta_file'], self.master_config['acts_file'])
        self.hl_pol = False
        self.prim_choices = self._hyperparams['prob'].get_prim_choices(self.task_list)
        opts = self._hyperparams['prob'].get_prim_choices(self.task_list)
        self.discrete_opts = [enum for enum, enum_opts in opts.items() if hasattr(enum_opts, '__len__')]
        self.continuous_opts = [enum for enum, enum_opts in opts.items() if not hasattr(enum_opts, '__len__')]
        #self.label_options = list(itertools.product(*[list(range(len(opts[e]))) for e in opts])) # range(self.num_tasks), *[range(n) for n in self.num_prims]))
        self.traj_hist = None
        self.reset_hist()


    def _setup_tracking_info(self):
        self._done = 0.
        self._task_done = 0.
        self._ret = 0.
        self._rew = 0.
        self._n_steps = 0.
        self.current_cond = 0
        self.cur_vid_id = 0

        self.debug = True
        self.n_opt = {task: 0 for task in self.plans}
        self.n_fail_opt = {task: 0 for task in self.plans}
        self.n_hl_plan = 0
        self.n_hl_fail = 0
        self.n_plans_run = 0
        self.n_plans_suc_run = 0
        self.n_policy_calls = {}


    def add_viewer(self):
        if hasattr(self, 'mjc_env'):
            self.mjc_env.add_viewer()
        else:
            self.viewer = OpenRAVEViewer(self.env)


    def get_opt_samples(self, task=None, clear=False):
        data = []
        if task is None:
            tasks = list(self.optimal_samples.keys())
        else:
            tasks = [task]
        for task in tasks:
            data.extend(self.optimal_samples[task])
            if clear:
                self.optimal_samples[task] = []
        return data


    def add_cont_sample(self, sample, max_buf=1e5):
        max_buf = int(max_buf)
        self.cont_samples.append(sample)
        self.cont_samples = self.cont_samples[-max_buf:]


    def get_cont_samples(self, clear=True):
        data = self.cont_samples
        if clear: self.cont_samples = []
        return data


    def get_samples(self, task):
        samples = []
        for batch in self._samples[task]:
            samples.append(batch)

        return samples


    def get_hist(self):
        return {'x': self._x_delta.copy(), 'task': self._prev_task.copy()}


    def store_hist(self, hist):
        self._x_delta[:] = hist['x'][:]
        self._prev_task[:] = hist['task'][:]


    def clear_samples(self, keep_prob=0., keep_opt_prob=1.):
        for task in self.task_list:
            n_keep = int(keep_prob * len(self._samples[task]))
            self._samples[task] = random.sample(self._samples[task], n_keep)

            n_opt_keep = int(keep_opt_prob * len(self.optimal_samples[task]))
            self.optimal_samples[task] = random.sample(self.optimal_samples[task], n_opt_keep)


    def store_x_hist(self, x):
        self._x_delta = x.reshape((self.hist_len+1, self.dX))


    def store_act_hist(self, u):
        self._prev_U = u.reshape((self.hist_len, self.dU))


    def store_task_hist(self, task):
        self._prev_task = task.reshape((self.task_hist_len, self.dPrimOut))


    def reset_sample_refs(self):
        for task in self.task_list:
            for batch in self._samples[task]:
                for sample in batch:
                    sample.set_ref_X(np.zeros((sample.T, self.symbolic_bound)))
                    sample.set_ref_U(np.zeros((sample.T, self.dU)))


    def reset_hist(self):
        self._prev_U = np.zeros((self.hist_len, self.dU))


    def add_task_paths(self, paths):
        self.task_paths.extend(paths)
        while len(self.task_paths) > MAX_TASK_PATHS:
            del self.task_paths[0]


    def get_task_paths(self):
        return copy.copy(self.task_paths)


    def clear_task_paths(self, keep_prob=0.):
        n_keep = int(keep_prob * len(self.task_paths))
        self.task_paths = random.sample(self.task_paths, n_keep)


    def animate_sample(self, sample):
        breakpoint()
        if hasattr(self, 'mjc_env'):
            for t in range(sample.T):
                mp_state = sample.get(STATE_ENUM, t)
                for param_name, attr in self.state_inds:
                    if attr == 'pose':
                        self.mjc_env.set_item_pos(param_name, mp_state[self.state_inds[param_name, attr]], mujoco_frame=False)
                    elif attr == 'rotation':
                        self.mjc_env.set_item_rot(param_name, mp_state[self.state_inds[param_name, attr]], mujoco_frame=False, use_euler=True)
                self.mjc_env.physics.forward()
                self.mjc_env.render(view=True)
                time.sleep(0.2)

        if self.viewer is None: return
        plan = self.plans_list[0]
        for p in list(plan.params.values()):
            if p.is_symbol(): continue
            p.openrave_body.set_pose(p.pose[:,0])

        state = sample.get(STATE_ENUM)
        for t in range(sample.T):
            for p_name, a_name in self.state_inds:
                if a_name == 'rotation' or plan.params[p_name].is_symbol(): continue
                if a_name == 'pose':
                    if (p_name, 'rotation') in self.state_inds:
                        pos = state[t, self.state_inds[p_name, a_name]]
                        rot = state[t, self.state_inds[p_name, 'rotation']]
                        plan.params[p_name].openrave_body.set_pose(pos, rot)
                    else:
                        pos = state[t, self.state_inds[p_name, a_name]]
                        plan.params[p_name].openrave_body.set_pose(pos)
                else:
                    attr_val = state[t, self.state_inds[p_name, a_name]]
                    plan.params[p_name].openrave_body.set_dof({a_name: attr_val})
            time.sleep(0.2)


    def draw_sample_ts(self, sample, t):
        # breakpoint()
        if self.viewer is None: return
        plan = self.plans_list[0]
        for p in list(plan.params.values()):
            if p.is_symbol(): continue
            p.openrave_body.set_pose(p.pose[:,0])
        state = sample.get(STATE_ENUM)
        for p_name, a_name in self.state_inds:
            if a_name == 'rotation' or plan.params[p_name].is_symbol(): continue
            if a_name == 'pose':
                if (p_name, 'rotation') in self.state_inds:
                    pos = state[t, self.state_inds[p_name, a_name]]
                    rot = state[t, self.state_inds[p_name, 'rotation']]
                    plan.params[p_name].openrave_body.set_pose(pos, rot)
                else:
                    pos = state[t, self.state_inds[p_name, a_name]]
                    plan.params[p_name].openrave_body.set_pose(pos)
            else:
                attr_val = state[t, self.state_inds[p_name, a_name]]
                plan.params[p_name].openrave_body.set_dof({a_name: attr_val})


    def sample(self, policy, condition, save_global=False, verbose=False, noisy=False):
        raise NotImplementedError


    def policies_initialized(self):
        for policy_name, policy in self.policies.items():
            if not len(self.continuous_opts) and policy_name == 'cont': continue
            if not self.policy_initialized(policy): return False
        return True


    def policy_initialized(self, policy):
        if hasattr(policy, 'is_initialized'):
            return policy.is_initialized()

        return hasattr(policy, 'scale') and policy.scale is not None


    def set_task_info(self, sample, cur_state, t, cur_task, task_f, policies=None):
        if task_f is None: return cur_task, None
        task = task_f(sample, t, cur_task)
        onehot_task = tuple([val for val in task if np.isscalar(val)])
        self.fill_sample(sample.condition, sample, cur_state, t, task, fill_obs=False)
        taskname = self.task_list[task[0]]
        policy = None
        if policies is not None: policy = policies[taskname]
        self.fill_sample(sample.condition, sample, cur_state.copy(), t, task, fill_obs=False)

        return task, policy


    def _sample_task(self, policy, condition, state, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True, task_f=None, hor=None, policies=None):
        # x0 = state[self._x_data_idx[STATE_ENUM]].copy()
        # task = tuple(task)
        # onehot_task = tuple([val for val in task if np.isscalar(val)])
        # plan = self.plans[onehot_task] if onehot_task in self.plans else self.plans[task[0]]

        # if hor is None:
        #     hor = plan.horizon if task_f is None else max([p.horizon for p in list(self.plans.values())])

        # breakpoint()

        self.T = self.hor
        sample = Sample(self)
        sample.init_t = 0
        col_ts = np.zeros(self.T)
        prim_choices = self.prob.get_prim_choices(self.task_list)
        sample.targets = self.target_vecs[condition].copy()
        n_steps = 0
        end_state = None
        cur_state = self.get_state() # x0
        sample.task = task
        sample.set(TASK_ENUM, np.tile(np.array([1. if i == task[0] else 0. for i in range(len(self.task_list))]), (self.T, 1)))

        self.fill_sample(condition, sample, cur_state.copy(), 0, task, fill_obs=True)
        for t in range(0, self.T):
            noise_full = np.zeros((self.dU,))

            self.fill_sample(condition, sample, cur_state.copy(), t, task, fill_obs=True)
            task, next_policy = self.set_task_info(sample, cur_state, t, task, task_f, policies)
            if next_policy is not None: policy = next_policy

            prev_vals = {}
            if policies is not None and 'cont' in policies and \
               len(self.continuous_opts):
                prev_vals = self.fill_cont(policies['cont'], sample, t)

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

            suc, col = self.run_policy_step(U_full / ACTION_SCALE, cur_state)
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


    def sample_task(self, 
                    policy, 
                    condition, 
                    state, 
                    task, 
                    use_prim_obs=False, 
                    save_global=False, 
                    verbose=False, 
                    use_base_t=True, 
                    noisy=True, 
                    fixed_obj=True, 
                    task_f=None, 
                    skip_opt=False, 
                    hor=None, 
                    policies=None):

        s = self._sample_task(policy, 
                              condition, 
                              state, 
                              task, 
                              save_global=save_global, 
                              noisy=noisy, 
                              task_f=task_f, 
                              hor=hor, 
                              policies=policies)
        s.opt_strength = 0.
        s.opt_suc = False
        return s


    @abstractmethod
    def goal(self, cond, targets=None):
        raise NotImplementedError


    @abstractmethod
    def goal_f(self, condition, state, targets=None, cont=False):
        raise NotImplementedError


    @abstractmethod
    def set_symbols(self, plan, task, anum=0, cond=0):
        raise NotImplementedError


    @abstractmethod
    def reset(self, m):
        raise NotImplementedError


    def _reset_to_state(self, x):
        self._done = 0.
        self._prev_U = np.zeros((self.hist_len, self.dU))
        self._x_delta = np.zeros((self.hist_len+1, self.dX))
        self._x_delta[:] = x.reshape((1,-1))
        if hasattr(self, 'mjc_env'):
            self.mjc_env.reset()


    @abstractmethod
    def reset_to_state(self, x):
        raise NotImplementedError


    @abstractmethod
    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        raise NotImplementedError


    def sample_optimal_trajectory(self, state, task, condition, opt_traj, traj_mean=[], targets=[], run_traj=True):
        if not len(targets):
            old_targets = self.target_vecs[condition]
        else:
            old_targets = self.target_vecs[condition]
            for tname, attr in self.target_inds:
                self.targets[condition][tname] = targets[self.target_inds[tname, attr]]
            self.target_vecs[condition] = targets

        # breakpoint()

        sample = self.sample_task(self.optimal_pol_cls(self.dU, 
                                                       self.action_inds, 
                                                       self.state_inds, 
                                                       opt_traj), 
                                  condition, 
                                  state, 
                                  task, 
                                  noisy=False, 
                                  skip_opt=True, 
                                  hor=len(opt_traj))
        sample.set_ref_X(sample.get_X())
        sample.set_ref_U(sample.get_U())

        for t in range(sample.T):
            pass

        self.target_vecs[condition] = old_targets
        for tname, attr in self.target_inds:
            self.targets[condition][tname] = old_targets[self.target_inds[tname, attr]]
            
        return sample


    def _sample_opt_traj(self, plan, state, task, condition):
        raise NotImplementedError('This should be defined in child')


    def replace_cond(self, cond=0, curric_step=-1):
        self.init_vecs[cond], self.targets[cond] = self.prob.get_random_initial_state_vec(self.config, False, self.dX, self.state_inds, 1)
        self.init_vecs[cond], self.targets[cond] = self.init_vecs[cond][0], self.targets[cond][0]
        self.x0[cond] = self.init_vecs[cond][:self.symbolic_bound]
        self.target_vecs[cond] = np.zeros((self.target_dim,))
        prim_choices = self.prob.get_prim_choices(self.task_list)

        if OBJ_ENUM in prim_choices and curric_step > 0:
            i = 0
            inds = np.random.permutation(list(range(len(prim_choices[OBJ_ENUM]))))
            for j in inds:
                obj = prim_choices[OBJ_ENUM][j]
                if '{0}_end_target'.format(obj) not in self.targets[cond]: continue
                if i >= len(prim_choices[OBJ_ENUM]) - curric_step: break
                self.x0[cond][self.state_inds[obj, 'pose']] = self.targets[cond]['{0}_end_target'.format(obj)]
                i += 1

        for target_name in self.targets[cond]:
            self.target_vecs[cond][self.target_inds[target_name, 'value']] = self.targets[cond][target_name]



    def first_postcond(self, sample, tol=1e-3, x0=None, task=None):
        for t in range(sample.T):
            cost = self.postcond_cost(sample, t=t, tol=tol, x0=x0, task=task)
            if cost == 0.:
                return t
        return -1


    def postcond_cost(self, sample, task=None, t=None, debug=False, tol=1e-3, x0=None):
        if t is None: t = sample.T-1
        if task is None: task = tuple(sample.get(FACTOREDTASK_ENUM, t=t))
        # active_ts = (-1, -1) for postcond verification
        return self.cost_f(sample.get_X(t), task, sample.condition, active_ts=(-1, -1), targets=sample.targets, debug=debug, tol=tol, x0=x0)


    def precond_cost(self, sample, task=None, t=0, tol=1e-3, x0=None, debug=False):
        if task is None: task = tuple(sample.get(FACTOREDTASK_ENUM, t=t))
        # active_ts = (0,0) for precond verification
        return self.cost_f(sample.get_X(t), task, sample.condition, active_ts=(0, 0), targets=sample.targets, tol=tol, x0=x0, debug=debug)


    def get_hl_info(self, state=None, targets=None, cond=0, plan=None, act=0):
    #     if targets is None: targets = self.target_vecs[cond].copy()

    #     initial = []
    #     plans = [plan]
    #     preds = []
    #     if plan is None:
    #         plans, reps = [], []
    #         for plan in self.plans.values():
    #             if str(plan.actions[0]) in reps: continue
    #             reps.append(str(plan.actions[0]))
    #             plans.append(plan)

    #     st = plans[0].actions[act].active_timesteps[0]

    #     for plan in plans:
    #         for pname, aname in self.state_inds:
    #             if plan.params[pname].is_symbol():
    #                 continue

    #             if state is not None:
    #                 val = state[self.state_inds[pname, aname]]
    #                 getattr(plan.params[pname], aname)[:,st] = val

    #             init_t = '{0}_init_target'.format(pname)
    #             if init_t in plan.params:
    #                 if st == 0:
    #                     pose = plan.params[pname].pose[:,st]
    #                     plan.params[init_t].value[:,0] = pose

    #                 at_pred = '(At {0} {1}) '.format(pname, init_t)
    #                 near_pred = '(Near {0} {1}) '.format(pname, init_t)
    #                 if at_pred not in initial:
    #                     initial.append(at_pred)

    #                 if near_pred not in initial:
    #                     initial.append(near_pred)

    #         for pname, aname in self.target_inds:
    #             if pname in plan.params:
    #                 val = targets[self.target_inds[pname, aname]]
    #                 getattr(plan.params[pname], aname)[:,0] = val

    #         init_preds = parse_state(plan, [], st, preds)
    #         initial.extend([p.get_rep() for p in init_preds])

        goal = self.goal(cond, targets)
        return [], goal
        # return list(set(initial)), goal


    def encode_action(self, action):
        prim_choices = self.prob.get_prim_choices(self.task_list)
        astr = str(action).lower()
        l = [0]
        for i, task in enumerate(self.task_list):
            if action.name.lower().find(task) >= 0:
                l[0] = i
                break

        # for enum in prim_choices:
        #     if enum is TASK_ENUM or not hasattr(prim_choices[enum], '__len__'): continue
        #     l.append(-1)
        #     act = action
        #     # for act in [action, next_act]:
        #     for i, opt in enumerate(prim_choices[enum]):
        #         if opt in [p.name for p in act.params]:
        #             l[-1] = i
        #             break

        #         if l[-1] >= 0:
        #             break

        #     if l[-1] < 0.:
        #         l[-1] = 0.
        

        return l # tuple(l)


    def encode_plan(self, plan, permute=False):
        encoded = []
        for a in plan.actions:
            encoded.append(self.encode_action(a))
        encoded = [tuple(l) for l in encoded]
        return encoded


    def get_encoded_tasks(self):
        if hasattr(self, '_cached_encoded_tasks'):
            return self._cached_encoded_tasks

        opts = self.prob.get_prim_choices(self.task_list)
        nacts = np.prod([len(opts[e]) for e in opts])
        dact = np.sum([len(opts[e]) for e in opts])
        out = np.zeros((len(self.label_options), dact))
        for i, l in enumerate(self.label_options):
            cur_vec = np.zeros(0)
            for j, e in enumerate(opts.keys()):
                v = np.zeros(len(opts[e]))
                v[l[j]] = 1.
                cur_vec = np.concatenate([cur_vec, v])

            out[i, :] = cur_vec

        self._cached_encoded_tasks = out
        return out


    def sample_rollout(self, x0, act_st, act_et, task, label='dagger'):
        policy = self.policies[self.task_list[task[0]]]
        if not self.policy_initialized(policy): return [], 1.
        hor = 2 * (act_et - act_st)
        if self.retime: hor *= 2

        policies = {}
        if 'cont' in self.policies and self.policies['cont'].is_initialized():
            policies['cont'] = self.policies['cont']


        sample = self.sample_task(policy, 
                                  0, 
                                  x0.copy(), 
                                  task, 
                                  hor=hor, 
                                  skip_opt=True, 
                                  policies=policies,)

        sample.source_label = label
        last_t = self.first_postcond(sample, 
                                     x0=x0, 
                                     task=task, 
                                     tol=ROLL_TOL)
        if last_t < 0:
            last_t = None
        else:
            sample.use_ts[last_t:] = 0.
            sample.prim_use_ts[last_t:] = 0.

        cost = 0
        if last_t is None:
            cost = self.postcond_cost(sample, 
                                      task, 
                                      sample.T-1, 
                                      x0=x0, 
                                      tol=ROLL_TOL)

        ref_traj, _, _, _ = self.reverse_retime([sample], (act_st, act_et), label=True, T=last_t)
        return ref_traj, cost


    def _set_plan_inds(self, plan):
        plan.state_inds = self.state_inds
        plan.action_inds = self.action_inds
        plan.dX = self.symbolic_bound
        plan.dU = self.dU



    def _run_solver(self, 
                    plan, 
                    task, 
                    a, 
                    ref_x0, 
                    act_st, 
                    ref_traj,
                    targets,
                    rollout,
                    n_resamples):

        set_params_attrs(plan.params, 
                         self.state_inds, 
                         ref_x0, 
                         act_st, 
                         plan=plan)

        self.set_symbols(plan, 
                         task, 
                         anum=a, 
                         st=act_st, 
                         targets=targets)

        old_free = plan.get_free_attrs()
        if not rollout: ref_traj = []

        try:
            success = self.ll_solver._backtrack_solve(plan, 
                                                      anum=a, 
                                                      amax=a,
                                                      n_resamples=n_resamples, 
                                                      init_traj=ref_traj, 
                                                      st=act_st)

        except AttributeError as e:
            print(('Non-fatal Opt Exception in full solve for', ref_x0, task, plan.actions[a]), e, act_st, plan.actions)
            success = False

        except ValueError as e:
            traceback.print_exception(*sys.exc_info())
            print(('Non-fatal EValue xception in full solve for', ref_x0, task, plan.actions[a]), e, act_st)
            success = False

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print(('Non-fatal Exception in full solve for', ref_x0, task, plan.actions[a]), e, act_st)
            success = False

        plan.store_free_attrs(old_free)
        self.n_opt[task] = self.n_opt.get(task, 0) + 1
        return success


    def _backtrack_solve(self, plan, anum=0, n_resamples=5, st=0):
        return self.backtrack_solve(plan, anum=anum, n_resamples=n_resamples, st=st)


    def backtrack_solve(self, 
                        plan, 
                        anum=0, 
                        n_resamples=5, 
                        x0=None, 
                        targets=None, 
                        rollout=False, 
                        traj=[], 
                        st=0, 
                        backup=False, 
                        label=None, 
                        permute=False, 
                        verbose=False, 
                        hist_info=None,
                        opt_backtrack=False,):

        # Handle to make PR Graph integration easier
        # Handle integration with Agent class
        a = anum
        bad_rollout = []
        info = {'to_render': []}
        init_t = time.time()
        path = []
        ref_traj = traj
        rollout_success = False
        success = False
        tasks = self.encode_plan(plan)
        used_rollout = False
        x_hist = {}
        
        if targets is None: targets = self.target_vecs[0]
        self.target_vecs[0] = targets
        # create perm_tasks vs tasks
        perm_tasks, perm_targets, perm = tasks, targets, {}

        # sets the state, target inds, dimensions
        self._set_plan_inds(plan)
        # if x0 is None, fill it in with the current state ind from plan
        if x0 is None:
            x0 = np.zeros_like(self.x0[0])
            fill_vector(plan.params, self.state_inds, x0, st)
        
        # populate plan params with ref_x0 state, store initial state for each action
        init_x0 = {}
        init_x0[anum] = x0
        ref_x0 = self.clip_state(x0)
        set_params_attrs(plan.params, self.state_inds, ref_x0, st, plan=plan)

        # initialize a dummy state, for use later
        dummy_sample = Sample(self)

        self.reset_to_state(x0)
        if hist_info is not None:
            self.store_hist_info(hist_info)
        else:
            hist_info = self.get_hist_info()

        # iterate through high-level plan
        while a < len(plan.actions):
            success = False
            act_ts = plan.actions[a].active_timesteps
            act_st, act_et = act_ts
            act_st = max(st, act_st)

            if permute:
                perm_tasks, perm_targets, perm = self.permute_tasks(tasks, targets, plan)

            task = tuple(tasks[a])
            perm_task = tuple(perm_tasks[a])

            # update state history
            if x_hist.get(a, None) is None:
                cur_x_hist = self._x_delta.copy()
                x_hist[a] = cur_x_hist
            else:
                cur_x_hist = x_hist[a]

            # get the initial state stored, and optionally clip
            if a in init_x0:
                x0 = init_x0[a]
                ref_x0 = self.clip_state(x0.copy())
            else:
                x0 = self.get_state()
                ref_x0 = self.clip_state(x0.copy())
                init_x0[a] = x0
            
            # initialize the dummy sample with the current state + targets
            dummy_sample.set(STATE_ENUM, ref_x0, t=0)
            dummy_sample.targets = targets

            # reset to state x0, and store the historical info
            self.update_hist_info(hist_info)
            self.reset_to_state(x0)
            self.store_hist_info(hist_info)

            # get precondition cost when executing in the simulator
            pre_cost = self.precond_cost(dummy_sample, task, 0)
            if pre_cost > 1e-4:
                print('FAILED EXECUTION OF PRECONDITIONS', 
                      plan.actions[a], 
                      plan.actions, 
                      task)
                self.precond_cost(dummy_sample, task, 0, debug=True)
                return False, False, path, info

            ref_traj, cost = [], 1.
            if (backup or rollout) and a not in bad_rollout:
                ref_traj, cost = self.sample_rollout(x0, act_st, act_et, task)

            if rollout and cost == 0:
                success = True
                used_rollout = opt_backtrack
                if last_t is None: last_t = sample.T-1
                next_x0 = sample.get_X(t=last_t)
                ref_x0 = next_x0.copy()
                ref_x0 = self.clip_state(ref_x0)
                path.append(sample)
                sample._postsuc = True
                sample.success = 1. - self.goal_f(0, next_x0, targets)
                #self.update_hist_info(hist_info)
                self.reset_to_state(next_x0)
                self.update_hist_info(hist_info)
                self.store_hist_info(hist_info)
                set_params_attrs(plan.params, self.state_inds, ref_x0, act_et, plan=plan)
                self._x_delta[:] = sample.get(STATE_HIST_ENUM, t=last_t).reshape(self._x_delta.shape)

            else:
                success = self._run_solver(plan, task, a, ref_x0, act_st, ref_traj, targets, rollout, n_resamples)

                if not success:
                    self.n_fail_opt[task] = self.n_fail_opt.get(task, 0) + 1
                    try:
                        print('FAILED TO SOLVE:', a, plan.actions[a], plan.get_failed_preds((act_st, act_et)), used_rollout)
                    except Exception as e:
                        print('FAILED, Error IN FAIL CHECK', e)

                    if not used_rollout:
                        # Scenario: 
                        # -> the optimizer failed
                        # -> and the previous sample tracked an optimal traj
                        # then the optimizer failed and we can return
                        return False, False, path, info

                    else:
                        # Scenario: 
                        # -> the optimizer failed
                        # -> and the previous sample didn't track an optimal traj
                        # then we can note we should not have rolled out at that time
                        # and backtrack to resolve it with the optimizer
                        used_rollout = False
                        bad_rollout.append(a-1)
                        x_hist[a] = None
                        path = path[:-1]
                        a -= 1
                        continue

                next_path, next_x0 = self.run_action(plan, 
                                                     a, 
                                                     x0, 
                                                     perm_targets, 
                                                     perm_task, 
                                                     act_st, 
                                                     reset=True, 
                                                     save=True, 
                                                     record=True, 
                                                     perm=perm, 
                                                     prev_hist=cur_x_hist, 
                                                     hist_info=hist_info,)
                for sample in next_path:
                    sample.opt_strength = 1.
                    sample.source_label = label

                path.extend(next_path)

                if not next_path[-1]._postsuc and not used_rollout:
                    # Scenario: 
                    # -> the action didn't reach post conditions
                    # -> and the previous sample tracked an optimal traj
                    # then the optimizer failed and we can return
                    self.n_plans_run += 1
                    return False, True, path, info

                elif not next_path[-1]._postsuc and used_rollout:
                    # Scenario: 
                    # -> the action didn't reach post conditions
                    # -> and the previous sample didn't track an optimal traj
                    # then we can note we should not have rolled out at that time
                    # and backtrack to resolve it with the optimizer
                    bad_rollout.append(a-1)
                    used_rollout = False
                    x_hist[a] = None 
                    path = path[:-1]
                    a -=1
                    continue

                used_rollout = False

            a += 1
           
        rollout_success = len(path) and self.goal_f(0, path[-1].end_state, targets) < 1e-3
        if success:
            self.n_plans_run += 1

        if not len(path):
            print('NO PATH SAMPLED FOR', plan.actions)

        if rollout_success:
            self.n_plans_suc_run += 1
            self.add_task_paths([path])

        print('Plans run vs. success:', 
              self.n_plans_run, 
              self.n_plans_suc_run, 
              self.process_id, 
              time.time()-init_t, 
              success, 
              rollout_success,)

        return success, True, path, info


    def run_action(self, 
                   plan, 
                   anum, 
                   x0, 
                   targets, 
                   task, 
                   start_ts=0, 
                   end_ts=None, 
                   reset=True, 
                   save=True, 
                   record=True, 
                   perm={}, 
                   base_x0=None, 
                   add_noop=True, 
                   prev_hist=None, 
                   hist_info=None,
                   aux_info=None):        
        # create static copy of x0
        x0 = x0.copy()
        static_x0 = x0.copy()
        start_ts = int(start_ts)
        # initialize misc variables, start + end times, active timesteps
        nzero = self.master_config.get('add_noop', 0)
        path = []
        st, et = plan.actions[anum].active_timesteps
        true_st, true_et = st, et
        if end_ts is not None:
            et = min(et, end_ts)

        if base_x0 is None:
            base_x0 = x0
        static_base = base_x0.copy()

        # reverse the permanent tasks key
        st = max(st, start_ts)
        rev_perm = {}
        for key, val in perm.items():
            rev_perm[val] = key

        
        opt_traj = np.zeros((et-st+1, self.symbolic_bound))
        if prev_hist is not None: self._x_delta[:] = prev_hist

        old_targets = self.target_vecs[0].copy()
        self.target_vecs[0] = targets
        cur_hist = prev_hist.copy() if prev_hist is not None else self._x_delta.copy()
        static_hist = cur_hist.copy()
        
        for pname, attr in self.action_inds:
            # populate opt-trajectory, x0, base_x0, and cur_history with static versions of these variables
            if plan.params[pname].is_symbol(): continue
            opt_traj[:,self.action_inds[perm.get(pname, pname), attr]] = getattr(plan.params[pname], attr)[:,st:et+1].T
        
        for pname, attr in self.state_inds:
            if plan.params[pname].is_symbol(): continue
            x0[self.state_inds[perm.get(pname, pname), attr]] = static_x0[self.state_inds[pname, attr]]
            base_x0[self.state_inds[perm.get(pname, pname), attr]] = static_base[self.state_inds[pname, attr]]
            cur_hist[:, self.state_inds[perm.get(pname, pname), attr]] = static_hist[:, self.state_inds[pname, attr]]

        # fill in _x_delta, historical info, and optioanally reset the state
        if reset: self.reset_to_state(x0)
        self._x_delta[:] = cur_hist
        if hist_info is not None:
            self.store_hist_info(hist_info)
        if aux_info is not None:
            self.store_aux_info(aux_info)

        # if self.retime:
        #     # get velocity
        #     vel = self.master_config.get('velocity', 0.3)
        #     # new_traj = self.retime_traj(opt_traj, vel=vel)
        #     new_traj = opt_traj # fuck retime
        #     if np.any(np.isnan(opt_traj)): print('NAN in opt traj')
        #     if np.any(np.isnan(new_traj)): print('NAN in retime')
        #     T = et-st+1
        #     t = 0
        #     ind = 0
        #     # get a sample of the optimal trajectory 
        #     while t < len(new_traj)-1:
        #         traj = new_traj[t:t+T]
        #         sample = self.sample_optimal_trajectory(x0, task, 0, traj, targets=targets)
        #         path.append(sample)
        #         sample.discount = 1.
        #         sample.opt_strength = 1.
        #         sample.opt_suc = True
        #         sample.step = ind
        #         sample.task_start = ind == 0
        #         sample.task_end = False
        #         sample.task = task
        #         ind += 1
        #         t += T - 1
        #         x0 = sample.end_state # sample.get_X(t=sample.T-1)
        #         sample.success = 1. - self.goal_f(0, x0, sample.targets)
        # else:

        # get a sample of the optimal trajectory, populate statistics
        sample = self.sample_optimal_trajectory(x0, task, 0, opt_traj, targets=targets)
        path.append(sample)
        sample.discount = 1.
        sample.opt_strength = 1.
        sample.opt_suc = True
        sample.task_start = True
        sample.draw = True
        sample.task = task
        x0 = sample.end_state # sample.get_X(sample.T-1)
        sample.success = 1. - self.goal_f(0, x0, targets)

        path[-1].task_end = True
        path[-1].set(TASK_DONE_ENUM, np.array([0, 1]), t=path[-1].T-1)
        #path[-1].prim_use_ts[-1] = 0.
        # if nzero > 0 and add_noop:
        #     # get a sample of the zero trajectory 
        #     zero_traj = np.tile(opt_traj[-1], [nzero, 1])
        #     zero_sample = self.sample_optimal_trajectory(path[-1].end_state, task, 0, zero_traj, targets=targets)
        #     x0 = zero_sample.end_state # sample.get_X(sample.T-1)
        #     zero_sample.use_ts[:] = 0.
        #     zero_sample.use_ts[:nzero] = 1.
        #     zero_sample.prim_use_ts[:] = np.zeros(len(zero_sample.prim_use_ts))
        #     zero_sample.step = path[-1].step + 1
        #     zero_sample.draw = False
        #     # zero_sample.success = path[-1].success
        #     zero_sample.success = 1. - self.goal_f(0, x0, targets)
        #     zero_sample.set(TASK_DONE_ENUM, np.tile([0,1], (zero_sample.T, 1)))
        #     zero_sample.task = task
        #     path.append(zero_sample)
        end_s = path[-1]
        end_s.task_end = True

        ## FOR NOW, UNCONDITIONALLY SAVE TRAJECTORIES (ASSUME ALL SUCCEEDED)
        for ind, s in enumerate(path):
            s.opt_strength = 1.
            s._postsuc = True
            if save: self.optimal_samples[self.task_list[task[0]]].append(s)
            # else:
            #     x1 = path[0].get_X(t=0)
            #     x2 = path[-1].end_state
            #     s._postsuc = False
            #     cost = self.postcond_cost(end_s, task, end_s.T-1, debug=(ind==0), x0=base_x0, tol=1e-3)
            #     if ind == 0 and save: print('Ran opt path w/postcond failure?', task, plan.actions[anum], self.process_id)

        # update x0 to the updated simulator state
        static_x0 = self.get_state().copy()
        static_hist = self._x_delta.copy()
        for pname, attr in self.state_inds:
            if plan.params[pname].is_symbol(): continue
            x0[self.state_inds[rev_perm.get(pname, pname), attr]] = static_x0[self.state_inds[pname, attr]]
            self._x_delta[:, self.state_inds[rev_perm.get(pname, pname), attr]] = static_hist[:, self.state_inds[pname, attr]]

        # permutation magic? not sure why permutation here... (for better HL policy learning?)
        # if len(perm.keys()):
        #     cur_hist = self._x_delta.copy()
        #     self.reset_to_state(x0)
        #     self._x_delta[:] = cur_hist

        self.target_vecs[0] = old_targets
        return path, x0


    def permute_ts(self, sample, ts):
        targets = sample.targets
        tasks = [tuple(sample.get(FACTOREDTASK_ENUM, t=ts).astype(np.int))]
        plan = self.plans[tasks[0]]
        perm_tasks, perm_targets, perm = self.permute_tasks(tasks, targets, plan)
        prev_hist = self._x_delta[:].copy()
        cur_x = sample.get_X(t=ts)
        perm_x = cur_x.copy()
        rev_perm = {}
        for key, val in perm.items():
            rev_perm[val] = key

        hist = None
        if STATE_HIST_ENUM in self.sensor_dims:
            hist = sample.get(STATE_HIST_ENUM, t=ts).reshape((-1, self.dX))

        for pname, attr in self.state_inds:
            if plan.params[pname].is_symbol(): continue
            perm_x[self.state_inds[perm.get(pname, pname), attr]] = cur_x[self.state_inds[perm.get(pname, pname), attr]] 
            if hist is not None:
                hist_val = hist[:, self.state_inds[perm.get(pname, pname), attr]]
                self._x_delta[:, self.state_inds[perm.get(pname, pname), attr]] = hist_val

        self.fill_sample(0, sample, perm_x, ts, perm_tasks[0], targets=perm_targets, fill_obs=True)
        self._x_delta[:] = prev_hist


    def retime_traj(self, traj, vel=0.3, inds=None):
        xpts = []
        fpts = []
        d = 0
        for t in range(len(traj)-1):
            xpts.append(d)
            fpts.append(traj[t])
            if t < len(traj):
                if inds is None:
                    d += np.linalg.norm(traj[t+1] - traj[t])
                else:
                    d += np.linalg.norm(traj[t+1][inds] - traj[t][inds])
        interp = scipy.interpolate.interp1d(xpts, fpts, axis=0)

        x = np.linspace(0, d, int(d / vel))
        # breakpoint()
        out = interp(x)
        return out


    def project_onto_constr(self, x0, plan, targets, ts=(0,0)):
        x0 = x0.copy()
        for pname, aname in self.state_inds:
            plan.params['robot_init_pose'].value[:,0] = x0[self.state_inds['pr2', 'pose']]
            if plan.params[pname].is_symbol(): continue
            getattr(plan.params[pname], aname)[:,0] = x0[self.state_inds[pname, aname]]
            plan.params[pname]._free_attrs[aname][:,0] = 1.
            if '{0}_init_target'.format(pname) in plan.params:
                plan.params['{0}_init_target'.format(pname)].value[:,0] = x0[self.state_inds[pname, aname]]
        suc = self.solver.solve(plan, traj_mean=np.array(x0).reshape((1,-1)), active_ts=(0,0))
        for pname, aname in self.state_inds:
            if plan.params[pname].is_symbol(): continue
            x0[self.state_inds[pname, aname]] = getattr(plan.params[pname], aname)[:,0]
        return x0

    # get failed predicates for a given simulator state, with a given task
    def _failed_preds(self, Xs, task, condition, active_ts=None, debug=False, targets=[], tol=1e-3, x0=None):
        # reshape simulator state
        Xs = Xs.reshape(1, Xs.shape[0])
        Xs = Xs[:, self._x_data_idx[STATE_ENUM]].copy()
        for n in range(len(Xs)):
            Xs[n] = self.clip_state(Xs[n])
        
        # reshape task
        true_task = task
        task = [val for val in true_task if np.isscalar(val)]
        task = tuple(task)

        # retrieve plan mapping desired val, get active_ts (repated from cost_f?)
        onehot_task = tuple([val for val in task if np.isscalar(val)])
        plan = self.plans[onehot_task] if onehot_task in self.plans else self.plans[task[0]]
        if active_ts[1] == -1:
            active_ts = (plan.horizon-1, plan.horizon-1)

        # use condition to index into targets if no target provided
        if targets is None or not len(targets):
            targets = self.target_vecs[condition]

        # set plan target attributes to those specified in-sim
        for tname, attr in self.target_inds:
            if tname in plan.params:
                getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]
        
        # for all active timesteps, set plan parameters to state_indices
        for t in range(active_ts[0], active_ts[1]+1):
            set_params_attrs(plan.params, self.state_inds, Xs[t-active_ts[0]], min(plan.horizon-1, t), plan=plan)

        # set an initial state for the plan, if given
        if x0 is not None:
            set_params_attrs(plan.params, self.state_inds, x0, 0, plan=plan)

        # set symbols for the plan, if more are needed than in the general case here
        self.set_symbols(plan, task)

        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, -1)

        # active_timesteps
        if active_ts == None:
            active_ts = (1, plan.horizon-1)
        elif active_ts[0] == -1:
            active_ts = (plan.horizon-1, plan.horizon-1)

        # get the failed predicates from the plan object, having constructed the appropriate number
        failed_preds = plan.get_failed_preds(active_ts=active_ts, priority=3, tol=tol)
        failed_preds = [p for p in failed_preds if ((p[1]._rollout or not type(p[1].expr) is EqExpr) and not p[1]._nonrollout)]
        if debug:
            print('FAILED:', failed_preds, plan.actions, task, self.process_id)
        return failed_preds


    # get pre- and post-condition success or failure at the sim-level for the 
    def cost_f(self, Xs, task, condition, active_ts=None, debug=False, targets=[], tol=1e-3, x0=None):
        # condition optionally indexes into the target vector
        
        true_task = task
        task = [val for val in true_task if np.isscalar(val)]

        onehot_task = tuple([val for val in task if np.isscalar(val)])
        # get stored plan corresponding to the currently executed task
        plan = self.plans[onehot_task] if onehot_task in self.plans else self.plans[task[0]]
        # get active_ts of current plan
        if active_ts == None:
            active_ts = (1, plan.horizon-1)
        elif active_ts[0] == -1:
            active_ts = (plan.horizon-1, plan.horizon-1)
        # retrieve the failed predicates (if any) at the current state, on the current task
        failed_preds = self._failed_preds(Xs, task, condition, active_ts=active_ts, debug=debug, targets=targets, tol=tol, x0=x0)
        cost = 0
        # iterate through failed_preds and check pred filation
        for failed in failed_preds:
            for t in range(active_ts[0], active_ts[1]+1):
                if t + failed[1].active_range[1] > active_ts[1]:
                    break

                viol = None
                try:
                    # get predicate violations
                    viol = failed[1].check_pred_violation(t, negated=failed[0], tol=1e-3)
                    if viol is not None:
                        cost += np.max(np.abs(viol))
                    if np.any(np.isnan(viol)):
                        print('Nan in failed pred check', failed, 'ts:', t, 'task:', task, 'viol:', viol)
                        print(failed[1].get_param_vector(t), failed)

                except Exception as e:
                    print('Exception in cost check for', failed, 'with viol', viol)
                    print(e)
                    cost += 1e1

        # always have a faiure if ANY fail at all
        if len(failed_preds) and cost < 1e-3:
            cost = 1
       
        if debug:
            print(active_ts, cost, failed_preds, task)
        return cost


    def plan_to_policy(self, plan=None, opt_traj=None):
        if opt_traj is None:
            opt_traj = plan_to_traj(plan, self.state_inds, self.dX)

        pol = self.optimal_pol_cls(self.dU, self.action_inds, self.state_inds, opt_traj)
        return pol


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
        im = self.mjc_env.render(camera_id=cam_id, height=self.image_height, width=self.image_width, view=False)
        return im


    def get_image(self, x, depth=False, cam_id=None):
        self.reset_to_state(x)
        if cam_id is None: cam_id = self.camera_id
        im = self.mjc_env.render(camera_id=cam_id, height=self.image_height, width=self.image_width, view=False, depth=depth)
        return im

    
    def compare_tasks(self, t1, t2):
        return np.all(np.array(t1) == np.array(t2))

    
    def reverse_retime(self, samples, ts, label=False, start_t=0, T=None):
        if T is None:
            T = sum([s.T-1 for s in samples]) + 1

        ts_range = np.linspace(start_t, T, ts[1]-ts[0]+1)
        cur_s, cur_t = 0, start_t
        cur_offset = 0
        traj = []
        env_state = []
        labels = []
        rev_labels = []
        prev_t = cur_t
        for ind, t in enumerate(ts_range):
            ts = int(t)
            cur_t = ts - cur_offset
            if cur_s >= len(samples) - 1 and cur_t >= samples[cur_s].T - 1:
                cur_s = len(samples)-1
                cur_t = samples[-1].T-1
            elif cur_t >= samples[cur_s].T - 1:
                cur_t = 0
                cur_s += 1
                cur_offset += samples[cur_s].T - 1

            sample = samples[cur_s]
            traj.append(sample.get(STATE_ENUM, t=cur_t))
            env_state.append(sample.env_state.get(cur_t, None))
            labels.append(cur_s)
            for _ in range(ts-prev_t):
                rev_labels.append(ts)

            prev_t = ts

        if label: return np.array(traj), labels, rev_labels, env_state
        return np.array(traj)


    def center_cont(self, abs_val, x):
        return abs_val


    def get_inv_cov(self):
        return np.eye(self.dU)


    def get_random_initial_state_vec(self, config, plans, dX, state_inds, n=1):
        xs, targets = self.prob.get_random_initial_state_vec(config, False, dX, state_inds, n)
        return xs, targets

    
    def clip_state(self, x):
        return x.copy()

    
    def fill_cont(self, policy, sample, t):
        vals = policy.act(sample.get_X(t=t), sample.get_cont_obs(t=t), t)
        old_vals = {}
        for ind, enum in enumerate(self.continuous_opts):
            # old_vals[enum] = sample.get(enum, t=t).copy()
            sample.set(enum, vals[ind], t=t)
            # sample.set(enum, 0.75, t=t)
        return old_vals


    def permute_hl_data(self, hl_mu, hl_obs, hl_wt, hl_prc, aux, x):
        return hl_mu, hl_obs, hl_wt, hl_prc


    def permute_cont_data(self, hl_mu, hl_obs, hl_wt, hl_prc, aux, x):
        return hl_mu, hl_obs, hl_wt, hl_prc


    def feasible_state(self, x, targets):
        return True


    def distance_to_goal(self, x=None, targets=None):
        if x is None: x = self.get_state()
        if targets is None: targets = self.target_vecs[0]
        opts = self.prob.get_prim_choices(self.task_list)
        rew = 0
        for opt in opts[OBJ_ENUM]:
            xinds = self.state_inds[opt, 'pose']
            targinds = self.target_inds['{}_end_target'.format(opt), 'value']
            dist = np.linalg.norm(x[xinds]-targets[targinds])
            rew -= dist

        rew /= len(opts[OBJ_ENUM])
        return -rew


    def reward(self, x=None, targets=None, center=False):
        if x is None: x = self.get_state()
        if targets is None: targets = self.target_vecs[0]
        opts = self.prob.get_prim_choices(self.task_list)
        rew = 0
        for opt in opts[OBJ_ENUM]:
            xinds = self.state_inds[opt, 'pose']
            targinds = self.target_inds['{}_end_target'.format(opt), 'value']
            dist = np.linalg.norm(x[xinds]-targets[targinds])
            rew -= dist

        rew /= (self.hor * self.rlen * len(opts[OBJ_ENUM]))
        #rew = np.exp(rew)
        return rew


    def save_video(self, rollout, savedir, success=None, ts=None, lab='', annotate=True, st=0):
        init_t = time.time()
        old_h = self.image_height
        old_w = self.image_width
        self.image_height = 256
        self.image_width = 256
        suc_flag = ''
        cam_ids = self.config['master_config'].get('visual_cameras', [self.camera_id])
        if success is not None:
            suc_flag = 'success' if success else 'fail'
        fname = savedir + '/{0}_{1}_{2}_{3}_{4}.npy'.format(self.process_id, self.cur_vid_id, suc_flag, lab, str(cam_ids)[1:-1].replace(' ', ''))
        self.cur_vid_id += 1
        buf = []
        for step in rollout:
            old_vec = self.target_vecs[0]
            if hasattr(step, '__len__'):
                ts_range = range(0, len(step))
                xs = step
            else:
                if not step.draw: continue
                self.target_vecs[0] = step.targets
                if ts is None: 
                    ts_range = range(st, step.T)
                else:
                    ts_range = range(ts[0], ts[1])
                xs = step.get_X()
            st = 0

            for t in ts_range:
                ims = []
                for ind, cam_id in enumerate(cam_ids):
                    ims.append(self.get_image(xs[t], cam_id=cam_id))
                im = np.concatenate(ims, axis=1)
                buf.append(im)
            self.target_vecs[0] = old_vec
        save_video(fname, dname=self.config['master_config']['descr'], arr=np.array(buf), savepath=savedir)
        self.image_height = old_h
        self.image_width = old_w


    def get_mask(self, sample, enum):
        mask = np.ones((sample.T, 1))
        return mask


    def get_hist_info(self):
        return {}
        

    def store_hist_info(self, info):
        return
    

    def store_aux_info(self, info):
        return


    def update_hist_info(self, info):
        return

