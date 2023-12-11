import pickle as pickle
from datetime import datetime
import numpy as np
import os
import pprint
import queue
import random
import sys
import time

from PIL import Image

from sco_py.expr import *

from opentamp.software_constants import *
from opentamp.core.internal_repr.plan import Plan
from opentamp.policy_hooks.sample import Sample
from opentamp.policy_hooks.sample_list import SampleList
from opentamp.policy_hooks.utils.policy_solver_utils import *
from opentamp.policy_hooks.rollout_supervisor import *
from opentamp.policy_hooks.server import *
from opentamp.policy_hooks.search_node import *


# imports for SPOT
import argparse
import os
import sys
import time

from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.robot_state import RobotStateClient
import bosdyn.client.util


ROLL_PRIORITY = 5

class RolloutServer(Server):
    def __init__(self, hyperparams):
        super(RolloutServer, self).__init__(hyperparams)
        self.hyperparams = hyperparams
        self.in_queue = self.rollout_queue
        self.out_queue = self.motion_queue
        self.check_precond = hyperparams['check_precond']
        self.check_postcond = hyperparams['check_postcond']
        self.neg_precond = hyperparams['neg_precond'] and self.use_neg
        self.neg_postcond = hyperparams['neg_postcond'] and self.use_neg
        self.check_midcond = hyperparams['check_midcond']
        self.check_random_switch = hyperparams['check_random_switch']
        self.fail_plan = hyperparams['train_on_fail']
        self.fail_mode = hyperparams['fail_mode']
        self.current_id = 0
        self.cur_step = 0
        self.ff_iters = self._hyperparams['warmup_iters']
        self.label_type = 'rollout'
        self.adj_eta = False
        self.run_hl_test = hyperparams.get('run_hl_test', False)
        self.prim_decay = hyperparams.get('prim_decay', 1.)
        self.prim_first_wt = hyperparams.get('prim_first_wt', 1.)
        self.check_prim_t = hyperparams.get('check_prim_t', 1)
        self.explore_eta = hyperparams['explore_eta']
        self.explore_wt = hyperparams['explore_wt']
        self.explore_n = hyperparams['explore_n']
        self.explore_nmax = hyperparams['explore_nmax']
        self.explore_success = hyperparams['explore_success']
        self.soft = hyperparams['soft_eval']
        self.eta = hyperparams['eta']
        self.init_supervisor()

        self.ll_rollout_opt = hyperparams['ll_rollout_opt']
        self.hl_rollout_opt = hyperparams['hl_rollout_opt']
        self.hl_test_log = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_'+'hl_test_{0}{1}log.npy'
        self.fail_log = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_'+'failure_{0}_log.txt'.format(self.id)
        self.fail_data_file = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_'+'failure_{0}_data.txt'.format(self.id)
        self.expert_data_file = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_exp_data.npy'
        self.hl_data = []
        self.fail_data = []
        self.postcond_info = []
        self.postcond_costs = {task: [] for task in self.task_list}
        self.fail_types = {}
        self.n_fails = 0
        self.dagger_log = LOG_DIR + hyperparams['weight_dir'] + '/RolloutInfo_{0}_log.txt'.format(self.id)
        self.last_hl_test = time.time()
        self.task_calls = {task: [] for task in self.task_list}
        self.task_successes = {task: [] for task in self.task_list}
        self.suc_trajs = [] # Each entry should be (traj, list of ts, list of prob under policy)
        self.failed_trajs = [] # Each entry should be (traj, list of ts, list of prob under policy)
        self.prev_suc = [] # Each entry is a state and a list of time, success pairs
        self.prev_fail = [] # Each entry is a state and a list of time, success pairs
        self.suc_per_goal = {}


    def init_supervisor(self):
        self.rollout_supervisor = RolloutSupervisor(self.agent, 
                                                    self.policy_opt,
                                                    self.hyperparams,
                                                    self.check_precond,
                                                    self.check_midcond,
                                                    self.check_postcond,
                                                    self.check_random_switch,
                                                    self.neg_precond,
                                                    self.neg_postcond,
                                                    self.soft,
                                                    self.eta)


    def hl_log_prob(self, path):
        log_l = 0
        for sample in path:
            for t in range(sample.T):
                hl_act = sample.get_prim_out(t=t)
                distrs = self.policy_opt.nets['primitive'].task_distr(sample.get_prim_obs(t=t), eta=1.)
                ind = 0
                p = 1.
                for d in distrs:
                    u = hl_act[ind:ind+len(d)]
                    p *= d[np.argmax(u)]
                    ind += len(d)
                log_l += np.log(p)
        return log_l


    def get_task(self, state, targets, prev_task, soft=False, eta=None):
        if eta is None:
            eta = self.eta
        sample = Sample(self.agent)
        sample.set_X(state.copy(), t=0)
        self.agent.fill_sample(0, sample, state.copy(), 0, prev_task, fill_obs=True, targets=targets)
        distrs = self.primitive_call(sample.get_prim_obs(t=0), soft, eta=eta, t=0, task=prev_task)
        #labels = list(self.agent.plans.keys())
        for d in distrs:
            for i in range(len(d)):
                d[i] = round(d[i], 5)
        #distr = [np.prod([distrs[i][l[i]] for i in range(len(l))]) for l in labels]
        #distr = np.array(distr)
        ind = []
        opts = self.agent.prob.get_prim_choices(self.task_list)
        enums = list(opts.keys())
        for i, d in enumerate(distrs):
            enum = enums[i]
            if not np.isscalar(opts[enum]):
                val = np.max(d)
                inds = [i for i in range(len(d)) if d[i] >= val]
                if not len(inds):
                    raise Exception('Bad network output in get_task: {} {}'.format(i, d))
                ind.append(np.random.choice(inds))
            else:
                ind.append(d)
        next_label = tuple(ind)
        return next_label

    def compare_tasks(self, task1, task2):
        for i in range(len(task1)):
            if type(task1[i]) is int and task1[i] != task2[i]:
                return False
            elif np.sum(np.abs(task1[i]-task2[i])) > 1e-2:
                return False

        return True
       

    def check_hl_statistics(self, xvar=None, thresh=0):
        from tabulate import tabulate
        ## NOTE -- turned off several of these, will reactivate gradually
        inds = {
                    'success_probability': 0,
                    'path length': 1,
                    # 'optimal rollout success': 9,
                    # 'time': 3,
                    # 'n data': 6,
                    # 'n plans': 10,
                    # 'subgoals anywhere': 11,
                    # 'subgoals closest dist': 12,
                    # 'collision': 8,
                    # 'any target': 13,
                    # 'smallest tol': 14,
                    # 'success anywhere': 7,
                 }
        data = np.array(self.hl_data)
        breakpoint()
        mean_data = np.mean(data, axis=0)[0]
        info = []
        headers = ['Statistic', 'Value']
        for key in inds:
            info.append((key, mean_data[inds[key]]))
        print(self._hyperparams['weight_dir'])
        print(tabulate(info))


    def test_hl(self, rlen=None, save=True, ckpt_ind=None,
                restore=False, debug=False, save_fail=False,
                save_video=False, eta=None):
        if ckpt_ind is not None:
            print(('Rolling out for index', ckpt_ind))

        self.set_policies()
        self.agent._eval_mode = True
        if restore:
            did_load = self.policy_opt.restore_ckpts(ckpt_ind)
            print('Successfully loaded ckpts: ', did_load)
        elif self.policy_opt.share_buffers:
            self.policy_opt.read_shared_weights()

        x0 = self.agent.x0[0]
        
        vals = []

        print('Initialized policies: ', self.agent.policies_initialized())

        for _ in range(20):
            self.agent.gym_env.resample_belief_true()  ## resample at the start of each rollout
            val, path = self.test_run(x0, [], max_t=20, hl=True, soft=self.config['soft_eval'], eta=eta, lab=-5, hor=25)
            vals.append(val)

        avg_val = np.mean(np.array(vals))
        if save:
            self.hl_data.append([(avg_val, len(path))])  ## update the HL statistics
            np.save(self.hl_test_log.format('', 'rerun_' if ckpt_ind is not None else ''), np.array(self.hl_data))
            
        # if self.agent.policies_initialized():
        #     init_t = time.time()
        #     self.agent.debug = False
        #     prim_opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        #     n_targs = list(range(len(prim_opts[OBJ_ENUM])))
        #     res = []
        #     ns = [self.config['num_targs']]
        #     if self.config['curric_thresh'] > 0:
        #         ns = list(range(1, self.config['num_targs']+1))
        #     n = np.random.choice(ns)
        #     s = []
        #     
        #     targets = self.agent.target_vecs[0].copy()
        #     for t in range(n, n_targs[-1]):
        #         breakpoint()
        #         obj_name = prim_opts[OBJ_ENUM][t]
        #         targ_name = '{0}_end_target'.format(obj_name)
        #         if (targ_name, 'value') in self.agent.target_inds:
        #             targets[self.agent.target_inds[targ_name, 'value']] = x0[self.agent.state_inds[obj_name, 'pose']]

        #     if rlen is None: rlen = self.agent.rlen
        #     hor = self.agent.hor
        #     nt = 500 # rlen * hor

        #     goal = self.agent.goal(0, targets)
        #     val, path = self.test_run(x0, targets, max_t=20, hl=True, soft=self.config['soft_eval'], eta=eta, lab=-5, hor=25)
        #     if goal not in self.suc_per_goal:
        #         self.suc_per_goal[goal] = []
        #     self.suc_per_goal[goal].append(val)

        #     adj_val = val
        #     #if not self.adj_eta:
        #     #    self.adj_eta = True
        #     #    adj_val, adj_path = self.test_run(x0, targets, rlen, hl=True, soft=True, eta=eta, lab=-10)
        #     #    self.adj_eta = False
        #     end_state = path[-1].end_state
        #     true_disp = self.agent.distance_to_goal(end_state, targets) #np.min(np.min([[self.agent.goal_f(0, step.get(STATE_ENUM, t), targets, cont=True) for t in range(step.T)] for step in path]))
        #     true_val = np.max(np.max([[1-self.agent.goal_f(0, step.get(STATE_ENUM, t), targets) for t in range(step.T)] for step in path]))
        #     subgoal_suc = 1-self.agent.goal_f(0, np.concatenate([s.get(STATE_ENUM) for s in path]), targets)
        #     anygoal_suc = 1-self.agent.goal_f(0, np.concatenate([s.get(STATE_ENUM) for s in path]), targets, anywhere=True)
        #     subgoal_dist = self.agent.goal_f(0, np.concatenate([s.get(STATE_ENUM) for s in path]), targets, cont=True)
        #     ncols = 1. if len(path) >1 and any([len(np.where(sample.col_ts > 0.99)[0]) > 3 for sample in path[:-1]]) else 0. # np.max([np.max(sample.col_ts) for sample in path])
        #     plan_suc_rate = np.nan if self.agent.n_plans_run == 0 else float(self.agent.n_plans_suc_run) / float(self.agent.n_plans_run)
        #     n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_plans'].value
        #     rew = self.agent.reward()
        #     ret = (self.agent._ret + rew * (nt - np.sum([s.T for s in path]))) #/ nt

        #     s.append((val,
        #               len(path), \
        #               true_disp, \
        #               time.time()-self.start_t, \
        #               self.config['num_objs'], \
        #               true_disp, \
        #               self.policy_opt.buf_sizes['n_data'].value, \
        #               true_val, \
        #               ncols, \
        #               plan_suc_rate, \
        #               n_plans,
        #               subgoal_suc,
        #               subgoal_dist,
        #               anygoal_suc,
        #               time.time()-init_t,
        #               n_plans/(time.time()-self.start_t)))
        #     if len(self.postcond_info):
        #         s[0] = s[0] + (np.mean(self.postcond_info[-5:]),)
        #     else:
        #         s[0] = s[0] + (0,)
        #     s[0] = s[0] + (adj_val,)
        #     s[0] = s[0] + (ret,)
        #     s[0] = s[0] + (rew,)
        #     #s[0] = s[0] + (-np.log(rew+1e-8),)
        #     if ckpt_ind is not None:
        #         s[0] = s[0] + (ckpt_ind,)
        #     res.append(s[0])
        #     if save:
        #         if all([s.opt_strength == 0 for s in path]): self.hl_data.append(res)
        #         if val > 1-1e-2:
        #             print('-----> SUCCESS! Rollout succeeded in test!', goal, self.id)
        #         # if self.use_qfunc: self.log_td_error(path)
        #         np.save(self.hl_test_log.format('', 'rerun_' if ckpt_ind is not None else ''), np.array(self.hl_data))

        #     if val < 1:
        #         fail_pt = {'time': time.time() - self.start_t,
        #                     'no': self.config['num_objs'],
        #                     'nt': self.config['num_targs'],
        #                     'N': self.policy_opt.N,
        #                     'x0': list(x0),
        #                     'targets': list(targets),
        #                     'goal': list(path[0].get(ONEHOT_GOAL_ENUM, t=0))}
        #         with open(self.fail_data_file, 'a+') as f:
        #             f.write(str(fail_pt))
        #             f.write('\n')

        #     opt_path = None
        #     if self.render and save_video:
        #         print('Saving video...', val)
        #         self.save_video(path, val > 0, lab='_{0}'.format(n_plans))
        #         if opt_path is not None: self.save_video(opt_path, val > 0, lab='_mp')
        #         print('Saved video. Rollout success was: ', val > 0)
        #     self.last_hl_test = time.time()
        # self.agent._eval_mode = False
        # self.agent.debug = True
        return avg_val, path


    def deploy(self, rlen=None, save=True, ckpt_ind=None,
                restore=False, debug=False, save_fail=False,
                save_video=False, eta=None):
        if ckpt_ind is not None:
            print(('Rolling out for index', ckpt_ind))

        self.set_policies()
        self.agent._eval_mode = True
        if restore:
            self.policy_opt.restore_ckpts(ckpt_ind)
        elif self.policy_opt.share_buffers:
            self.policy_opt.read_shared_weights()

        if self.agent.policies_initialized():
            print('SUCCESSFULLY INITIALIZED POLICIES')
            init_t = time.time()
            self.agent.debug = False
            prim_opts = self.agent.prob.get_prim_choices(self.agent.task_list)
            n_targs = list(range(len(prim_opts[OBJ_ENUM])))
            res = []
            ns = [self.config['num_targs']]
            if self.config['curric_thresh'] > 0:
                ns = list(range(1, self.config['num_targs']+1))
            n = np.random.choice(ns)
            s = []
            x0 = self.agent.x0[0]
            targets = self.agent.target_vecs[0].copy()
            for t in range(n, n_targs[-1]):
                obj_name = prim_opts[OBJ_ENUM][t]
                targ_name = '{0}_end_target'.format(obj_name)
                if (targ_name, 'value') in self.agent.target_inds:
                    targets[self.agent.target_inds[targ_name, 'value']] = x0[self.agent.state_inds[obj_name, 'pose']]

            if rlen is None: rlen = self.agent.rlen
            hor = self.agent.hor
            nt = 500 # rlen * hor

            goal = self.agent.goal(0, targets)
            val, path = self.deploy_run(x0, targets, 20, hl=True, soft=self.config['soft_eval'], eta=eta, lab=-5, hor=25)
        # print(path[-1].get_obs().shape)
        print('REACHED END')

    def run(self):
        step = 0
        self.agent.hl_pol = False
        while not self.stopped:
            step += 1
            cont_samples = self.agent.get_cont_samples()
            if self._n_plans <= self.ff_iters:
                n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_plans']
                self._n_plans = n_plans.value

            # make it so not running constantly!
            if self.run_hl_test and time.time() - self.last_hl_test > 120:
                self.agent._eval_mode = True
                self.agent.replace_cond(0)
                self.agent.reset(0)
                n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_plans'].value
                save_video = self.id.find('test') >= 0
                val, path = self.test_hl(save_video=save_video)

            if self.run_hl_test: 
                if self.debug or self.plan_only:
                    break # stop iteration after one loop

                continue
            if self._n_plans < self.ff_iters: 
                if self.debug or self.plan_only:
                    break # stop iteration after one loop

                continue

            self.set_policies()
            node = self.pop_queue(self.rollout_queue)
            if node is None:
                node = self.spawn_problem()

            self.send_rollout(node)


            for task in self.agent.task_list:
                data = self.agent.get_opt_samples(task, clear=True)
                if len(data) and self.ll_rollout_opt:
                    inv_cov = self.agent.get_inv_cov()
                    self.update_policy(data, label='rollout', task=task, inv_cov=inv_cov)

            if self.hl_rollout_opt:
                self.run_hl_update(label='rollout')

            self.agent.clear_task_paths()
            if len(cont_samples):
                self.update_cont_network(cont_samples)

            self.write_log()

            if self.debug or self.plan_only:
                break # stop iteration after one loop


    def send_rollout(self, node):
        x0 = node.x0
        targets = node.targets
        val, path = self.rollout_supervisor.rollout(x0, targets, node)

        if self.id.find('r0') >= 0:
            self.save_video(path, val > 0, lab='_rollout')

        self.log_path(path, -20)
        for llnode in self.rollout_supervisor.ll_nodes:
            self.push_queue(llnode, self.motion_queue)

        for hlnode in self.rollout_supervisor.hl_nodes:
            self.push_queue(hlnode, self.task_queue)

        if len(self.rollout_supervisor.neg_samples) and self.use_neg:
            self.update_negative_primitive(self.rollout_supervisor.neg_samples)

        for key in self.postcond_costs:
            if key in self.rollout_supervisor.postcond_costs:
                self.postcond_costs[key] += self.rollout_supervisor.postcond_costs[key]

        for key in self.fail_types:
            if key in self.rollout_supervisor.fail_types:
                self.fail_types[key] += self.rollout_supervisor.fail_types[key]

        for key in self.task_successes:
            if key in self.rollout_supervisor.task_successes:
                self.task_successes[key] += self.rollout_supervisor.task_successes[key]

        self.postcond_info.extend(self.rollout_supervisor.postcond_info)
        self.rollout_supervisor.reset()
        self.init_supervisor()
    
    def robot_deploy_setup(self):
        sdk = create_standard_sdk('spot')
        robot = sdk.create_robot('192.168.80.3')

        bosdyn.client.util.authenticate(robot)
        robot.start_time_sync(1.0)
        robot.time_sync.wait_for_sync()

        # Lease logic
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()

        # ESTOP logic
        estop_client = robot.ensure_client(EstopClient.default_service_name)
        estop_endpoint = EstopEndpoint(estop_client, 'GNClient', 9.0)

        # robot clients
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        power_client = robot.ensure_client(PowerClient.default_service_name)

        return robot, robot_command_client, estop_endpoint, robot_state_client, power_client


    def deploy_run(self, state, targets, max_t=1, hl=False, soft=False, check_cost=True, eta=None, lab=0, hor=30):
        def task_f(sample, t, curtask):
            return self.get_task(sample.get_X(t=t), sample.targets, curtask, soft)
        self.agent.reset_to_state(state)
        # deploy_setup = self.robot_deploy_setup()
        path = []
        val = 0
        nout = len(self.agent._hyperparams['prim_out_include'])
        l = None
        t = 0
        if eta is not None: self.eta = eta 
        old_eta = self.eta
        debug = np.random.uniform() < 0.1
        max_t=1
        while t < max_t and val < 1-1e-2 and self.agent.feasible_state(state, targets):
            # l = self.get_task(state, targets, l, soft)
            # TODO SOURCE THE TASK SELECTION ENTIRELY FROM THE STATE

            # modify where the task comes from

            # if l is None: break
            # task_name = self.task_list[l[0]]
            # print(task_name)
            pol = self.agent.policies['move']  # hardcode for now

            theta = 0  # maintain rotation through the sim
            # sensor = np.ones(shape=(69,)) * 2.5  # beyond detection threshold
            task = np.array([1, 0, 0])  # just motion for now...
            theta_vec = np.array([-np.sin(theta), np.cos(theta)])
            end_pose = np.array([0., 10.])  # goal hardcoded for now

            # add execution loop here
            for _ in range(0, 250):
                state = np.concatenate((task, end_pose, theta_vec))

                u = pol.act(None, state, None, [0, 0, 0, 0])
                # print(u[0], u[1], u[3])  # x, y, theta from gripper policy'
                rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                pos_offset = rot_matrix.dot(np.array([u[0], u[1]])) # simplified, do them in sequence
                end_pose -= pos_offset.reshape((2,))
                theta += u[3]
                theta_vec = np.array([-np.sin(theta), np.cos(theta)])
                print(end_pose)


                # vel = 1
                # cmd = RobotCommandBuilder.synchro_velocity_command(v_x=vel*u[1], v_y=vel*u[0], v_rot=vel*u[3])
                # deploy_setup[1].robot_command(command=cmd, end_time_secs=time.time() + 1.0)
                # time.sleep(5)


            # s = self.agent.sample_task(pol, 0, state, l, noisy=False, task_f=task_f, skip_opt=True, hor=hor, policies=self.agent.policies)
            # val = 1 - self.agent.goal_f(0, s.get_X(s.T-1), targets)
            t += 1
            # state = s.end_state # s.get_X(s.T-1)
            # path.append(s)
        self.eta = old_eta
        self.log_path(path, lab)
        # print('Full Path From Deploy-Run')
        # print([samp.get_obs() for samp in path])
        return val, path


    def test_run(self, state, targets, max_t=20, hl=False, soft=False, check_cost=True, eta=None, lab=0, hor=30):
        def task_f(sample, t, curtask):
            return self.get_task(sample.get_X(t=t), sample.targets, curtask, soft)
        self.agent.reset_to_state(state)
        path = []
        val = 0
        nout = len(self.agent._hyperparams['prim_out_include'])
        l = None
        t = 0
        if eta is not None: self.eta = eta 
        old_eta = self.eta
        debug = np.random.uniform() < 0.1
        while t < max_t and self.agent.feasible_state(state, targets):
            l = self.get_task(state, targets, l, soft)
            if l is None: break
            task_name = self.task_list[l[0]]
            pol = self.agent.policies[task_name]
            s = self.agent.sample_task(pol, 0, state, l, noisy=False, task_f=task_f, skip_opt=True, hor=hor, policies=self.agent.policies)
            val = 1 - self.agent.goal_f(0, s.get_X(s.T-1), targets)
            t += 1
            state = s.end_state # s.get_X(s.T-1)
            path.append(s)
        self.eta = old_eta
        self.log_path(path, lab)
        return val, path

    def check_failed_likelihoods(self):
        cur_t = time.time()
        for buf in [self.suc_trajs, self.failed_trajs]:
            for traj, ts, lls in buf:
                cur_lls = []
                for sample in traj:
                    for t in range(sample.T):
                        task = tuple(sample.get(FACTOREDTASK_ENUM, t=t).astype(int))
                        task_name = self.task_list[task[0]]
                        pol = self.agent.policies[task_name]
                        mu = pol.act(sample.get_X(t=t), sample.get_obs(t=t), t)
                        act = sample.get(ACTION_ENUM, t=t)
                        cur_lls.append(np.sum((mu-act)**2))
                ts.append(cur_t)
                lls.append(np.mean(cur_lls))

    def get_log_info(self):
        info = {
                'time': time.time() - self.start_t,
                }

        wind = 100
        self.check_failed_likelihoods()
        info['dagger_success'] = np.mean(self.postcond_info[-wind:]) if len(self.postcond_info) else 0.
        for key in self.fail_types:
            info[key] = self.fail_types[key] / self.n_fails

        for key in self.postcond_costs:
            if len(self.postcond_costs[key]):
                info[key+'_costs'] = np.mean(self.postcond_costs[key][-wind:])

        for key in self.task_successes:
            if len(self.task_successes[key]):
                info['{0}_successes'.format(key)] = np.mean(self.task_successes[key][-wind:])
            else:
                info['{0}_successes'.format(key)] = 0. 

        info['per_goal_success'] = {goal: np.mean(self.suc_per_goal[goal][-1:]) for goal in self.suc_per_goal}

        return info


    def write_log(self):
        with open(self.dagger_log, 'a+') as f:
            info = self.get_log_info()
            pp_info = pprint.pformat(info, depth=60)
            f.write(str(pp_info))
            f.write('\n\n')

