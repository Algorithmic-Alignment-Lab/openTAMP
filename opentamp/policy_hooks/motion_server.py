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
from scipy.cluster.vq import kmeans2 as kmeans

from opentamp.software_constants import *
from opentamp.core.internal_repr.plan import Plan
import opentamp.core.util_classes.transform_utils as T
from opentamp.policy_hooks.sample import Sample
from opentamp.policy_hooks.sample_list import SampleList
from opentamp.policy_hooks.utils.policy_solver_utils import *
from opentamp.policy_hooks.server import Server
from opentamp.policy_hooks.search_node import *


import torch
import matplotlib.pyplot as plt

LOG_DIR = 'experiment_logs/'

class MotionServer(Server):
    def __init__(self, hyperparams):
        super(MotionServer, self).__init__(hyperparams)
        self.in_queue = self.motion_queue
        self.out_queue = self.task_queue
        self.label_type = 'optimal'
        self.opt_wt = hyperparams['opt_wt']
        self.motion_log = LOG_DIR + hyperparams['weight_dir'] + '/MotionInfo_{0}_log.txt'.format(self.id)
        self.log_infos = []
        self.infos = {'n_ff': 0, 
                      'n_postcond': 0, 
                      'n_precond': 0, 
                      'n_midcond': 0, 
                      'n_explore': 0, 
                      'n_plans': 0}

        self.avgs = {key: [] for key in self.infos}

        self.fail_infos = {'n_fail_ff': 0, 
                           'n_fail_postcond': 0, 
                           'n_fail_precond': 0, 
                           'n_fail_midcond': 0, 
                           'n_fail_explore': 0, 
                           'n_fail_plans': 0}

        self.fail_avgs = {key: [] for key in self.fail_infos}

        self.fail_rollout_infos = {'n_fail_rollout_ff': 0, 
                                   'n_fail_rollout_postcond': 0, 
                                   'n_fail_rollout_precond': 0, 
                                   'n_fail_rollout_midcond': 0, 
                                   'n_fail_rollout_explore': 0}

        self.init_costs = []
        self.rolled_costs = []
        self.final_costs = []
        self.plan_times = []
        self.plan_horizons = []
        self.opt_rollout_info = {'{}_opt_rollout_success'.format(taskname): [] for taskname in self.task_list}
        with open(self.motion_log, 'w+') as f:
            f.write('')


    def gen_plan(self, node):
        node.gen_plan(self.agent.hl_solver, 
                      self.agent.openrave_bodies, 
                      self.agent.ll_solver)
        
        plan = node.curr_plan

        # belief-space planning hook
        if 'observation_model' in self._hyperparams.keys():
            plan.set_observation_model(self._hyperparams['observation_model'])
            # TODO: check functionality with these being outdated
            # plan.set_max_likelihood_obs(self._hyperparams['max_likelihood_obs'])
            # plan.initialize_beliefs()  

        if type(plan) is str: return plan
        if not len(plan.actions): return plan

        for a in range(min(len(plan.actions), plan.start+1)):
            task = self.agent.encode_action(plan.actions[a])
            self.agent.set_symbols(plan, task, a, targets=node.targets)
        
        plan.start = min(plan.start, len(plan.actions)-1)
        ts = (0, plan.actions[plan.start].active_timesteps[0])

        ## NOTE: we allow for failed prefixes of plans, owing to nondeterminisim from random effects
        ##  in belief-space plans (more things may fail silently because of this...)
        # try:
        #     failed_prefix = plan.get_failed_preds(active_ts=ts, tol=1e-3)
        # except Exception as e:
        #     failed_prefix = ['ERROR IN FAIL CHECK', e]

        # if len(failed_prefix) and node.hl:
        #     print('BAD PREFIX! -->', plan.actions[:plan.start], 'FAILED', failed_prefix, node._trace)
        #     plan.start = 0

        # ts = (0, plan.actions[plan.start].active_timesteps[0])
        
        ## TODO populates attribute of start of current update, to the current simulator state...
        # if node.freeze_ts <= 0:
        #     set_params_attrs(plan.params, self.agent.state_inds, node.x0, ts[1])

        plan.freeze_actions(plan.start)
        # cur_t = node.freeze_ts if node.freeze_ts >= 0 else 0

        # breakpoint()

        return plan


    def refine_plan(self, node):
        # breakpoint()
        
        start_t = time.time()
        if node is None: return

        plan = self.gen_plan(node)

        if type(plan) is str or not len(plan.actions): return

        cur_t = node.freeze_ts if node.freeze_ts >= 0 else 0
        cur_step = 2
        self.n_plans += 1

        # breakpoint()

        while cur_t >= 0:
            path, refine_success, replan_success = self.collect_trajectory(plan, node, cur_t)
            
            ## if plan about to fail since not hitting expansion limit
            if node.expansions >= EXPAND_LIMIT or (replan_success and refine_success):
                self.log_node_info(node, replan_success, path)
            
            prev_t = cur_t
            cur_t -= cur_step
            if (refine_success and replan_success) and len(path) and path[-1].success: continue

            # parse the failed predicates for the plan, and push the change to task queue
            if not (refine_success and replan_success): self.parse_failed(plan, node, prev_t)
            while len(plan.get_failed_preds((cur_t, cur_t))) and cur_t > 0:
                cur_t -= 1

            node.freeze_ts = cur_t

            plan = self.gen_plan(node)


    def collect_trajectory(self, plan, node, cur_t):
        x0 = None
        if cur_t < len(node.ref_traj): x0 = node.ref_traj[cur_t]
        if cur_t == 0: x0 = node.x0

        wt = self.explore_wt if node.label.lower().find('rollout') >= 0 or node.nodetype.find('dagger') >= 0 else 1.
        # verbose = self.verbose and (self.id.find('r0') >= 0 or np.random.uniform() < 0.05)
        self.agent.store_hist_info(node.info)
        
        init_t = time.time()
        
        # success  = self.agent.ll_solver.backtrack_solve(plan, 
        #                                             anum=plan.start, 
        #                                             x0=x0,
        #                                             targets=node.targets,
        #                                             n_resamples=self._hyperparams['n_resample'], 
        #                                             rollout=self.rollout_opt, 
        #                                             traj=node.ref_traj, 
        #                                             st=cur_t, 
        #                                             permute=self.permute_hl,
        #                                             label=node.nodetype,
        #                                             backup=self.backup,
        #                                             verbose=verbose,
        #                                             hist_info=node.info)
        
        ## TODO: replace goal instantiation with call to GymAgent
        
        ## preprocessing for beliefs vector (used for certainty constraints, e.g. )
        if plan.start == 0 and len(plan.belief_params) > 0:
            print('Planning on new problem')
            
            planned_obs = {}
            
            for param in plan.belief_params:
                planned_obs[param.name] = param.belief.dist.sample()
                param.pose[:, 0] = planned_obs[param.name].detach().numpy()
            
            plan.observation_model.set_active_planned_observations(planned_obs)

            node.replan_start = 0

            ## get a true belief state, to plan against in the problem
            self.agent.gym_env.resample_belief_true()

            plan.set_mc_lock(self.config['mc_lock'])
        
        ## get the true state of belief variables from sim
        if len(plan.belief_params) > 0:
            goal = self.agent.gym_env.belief_true

        ## disable optimistic predicates for all actions coming before the start of current planning action
        for anum in range(plan.start):
            a = plan.actions[anum]
            for pred_dict in a.preds:
                if pred_dict['pred'].optimistic:
                    ## disable the predicate by using no-eval notation
                    pred_dict['active_timesteps'] = (pred_dict['active_timesteps'][0], pred_dict['active_timesteps'][0]-1)
                    pred_dict['pred'].active_range = (pred_dict['pred'].active_range[0], pred_dict['pred'].active_range[0]-1)
        

        ## reset observations from the beginning
        for param in plan.belief_params:
            param.belief.samples = param.belief.samples[:, :, :plan.actions[plan.start].active_timesteps[0]+1]
                
        refine_success = self.agent.ll_solver._backtrack_solve(plan,
                                                      anum=plan.start,
                                                      n_resamples=5,
                                                      init_traj=node.ref_traj,
                                                      st=cur_t)

        ## for belief-space replanning, only refine if indeed belief-space, and 
        replan_success = True
        if refine_success and len(plan.belief_params) > 0:
            print('Refining from: ', node.replan_start)    

            ## reset beliefs to start of plan
            for param in plan.belief_params:
                param.belief.samples = param.belief.samples[:, :, :plan.actions[node.replan_start].active_timesteps[0]+1]

            ## filter them forward, under the new assumption for the observation
            for anum in range(node.replan_start, len(plan.actions)):
                active_ts = plan.actions[anum].active_timesteps
                
                plan.rollout_beliefs(active_ts)
                
                if plan.actions[anum].non_deterministic:
                    ## perform MCMC to update, using the goal inferred from at the time
                    plan.filter_beliefs(active_ts, provided_goal=goal)
                else:
                    ## just propagate beliefs forward, no inference needed
                    for param in plan.belief_params:
                        new_samp = torch.cat((param.belief.samples, param.belief.samples[:, :, -1:]), dim=2)
                        param.belief.samples = new_samp
                anum += 1
            
            ## see if plan with new beliefs is still valid
            ## reset the observation to new sample if *NOT* solved
            fail_step, fail_pred, _ = node.get_failed_pred()
            if fail_pred:
                ## replanning has now failed
                for anum in range(len(plan.actions)):
                    a = plan.actions[anum]
                    new_assumed_goal = {}

                    ## reset the true state planned to be a random one, with consistent index across samples
                    new_goal_idx = np.random.randint(0, param.belief.samples.shape[0])
                    if a.active_timesteps[0] <= fail_step and fail_step < a.active_timesteps[1]:
                        for param in plan.belief_params:
                            ## set new assumed value for planning to sample from belief -- random choice
                            new_assumed_goal[param.name] = param.belief.samples[new_goal_idx,:,a.active_timesteps[0]]
                            param.pose[:, a.active_timesteps[0]] = new_assumed_goal[param.name]
                            new_assumed_goal[param.name] = torch.tensor(new_assumed_goal[param.name])
                        node.replan_start = anum

                    ## populate with new goal
                    plan.observation_model.set_active_planned_observations(new_assumed_goal)

                    for pred_dict in a.preds:
                        ## disable optimistic actions for planning (ones that can change with random effects)
                        if (pred_dict['active_timesteps'][0] < fail_step and fail_step <= pred_dict['active_timesteps'][1]) and pred_dict['pred'].optimistic:
                            ## disable the predicate by using no-eval notation
                            pred_dict['active_timesteps'] = (pred_dict['active_timesteps'][0], pred_dict['active_timesteps'][0]-1)
                            pred_dict['pred'].active_range = (pred_dict['pred'].active_range[0], pred_dict['pred'].active_range[0]-1)

                replan_success = False


        if replan_success and refine_success:
            print('Success')

            ## if plan only, invoke a breakpoint and inspect the plan statistics
            if self.plan_only:
                ## TODO add a general wrapper here
                print(plan.params['pr2'].pose)
                print(plan.params['target1'].pose)
                print(plan.params['target1'].belief.samples.mean(axis=0))
                print(plan.params['target1'].belief.samples.std(axis=0))
                for idx in range(plan.params['target1'].belief.samples.shape[2]):
                    plt.scatter(plan.params['target1'].belief.samples[:,0,idx], plan.params['target1'].belief.samples[:,1,idx], alpha=0.5)
                    plt.scatter([3.0], [3.0], alpha=1.0, marker='x')
                    plt.savefig('samps'+str(idx)+'.pdf')
                    plt.clf()
                breakpoint()


        # TODO: reactivate when reintegrating RolloutServer + Policy stuff
        path = []

        if not self.plan_only and (refine_success and replan_success):
            ## populate the sample with the entire plan
            a_num = 0
            st = plan.actions[a_num].active_timesteps[0]
            tasks = self.agent.encode_plan(plan)

            for a_num in range(len(plan.actions)):
                new_path, x0 = self.agent.run_action(plan, 
                            a_num, 
                            x0,
                            self.agent.target_vecs[0], 
                            tasks[a_num], 
                            st,
                            reset=True,
                            save=True, 
                            record=True)
                
                path.extend(new_path)

            end_t = time.time()
            for step in path:
                step.wt = wt

            if (refine_success and replan_success):
                self.plan_horizons.append(plan.horizon)
                self.plan_horizons = self.plan_horizons[-5:]
                self.plan_times.append(end_t-init_t)
                self.plan_times = self.plan_times[-5:]

            self.agent.add_task_paths([path])  ## add the given history of tasks from this successful rollout

        ## TODO debug logging
        # self._log_solve_info(path, replan_success, node, plan)
        return path, refine_success, replan_success


    def parse_failed(self, plan, node, prev_t):
        try:
            fail_step, fail_pred, fail_negated = node.get_failed_pred(st=prev_t)
        except:
            fail_pred = None

        if fail_pred is None:
            print('WARNING: Failure without failed constr?')
            return

        # NOTE: refines also done if linear constraints fail, owing to replanning
        failed_preds = plan.get_failed_preds((prev_t, fail_step+fail_pred.active_range[1]), priority=-1)

        if len(failed_preds):
            print('Refine failed with linear constr. viol.', 
                   node._trace, 
                   plan.actions, 
                   failed_preds, 
                   len(node.ref_traj), 
                   node.label,)

            # return ## NOTE: altered return logic here, so all failures hit expansion limit

        print('Refine failed:', 
              plan.get_failed_preds((0, fail_step+fail_pred.active_range[1])), 
              fail_pred, 
              fail_step, 
              plan.actions, 
              node.label, 
              node._trace, 
              prev_t,)

        if not node.hl and not node.gen_child(): return

        n_problem = node.get_problem(fail_step, fail_pred, fail_negated)
        abs_prob = self.agent.hl_solver.translate_problem(n_problem, goal=node.concr_prob.goal)
        prefix = node.curr_plan.prefix(fail_step)
        # breakpoint()
        hlnode = HLSearchNode(abs_prob,
                             node.domain,
                             n_problem,
                             priority=node.priority+1,
                             prefix=prefix,
                             llnode=node,
                             x0=node.x0,
                             targets=node.targets,
                             expansions=node.expansions+1,
                             label=self.id,
                             nodetype=node.nodetype,
                             info=node.info, 
                             replan_start=node.replan_start)
        self.push_queue(hlnode, self.task_queue)
        print(self.id, 'Failed to refine, pushing to task node.')


    def run(self):
        step = 0
        # breakpoint()
        while not self.stopped:
            node = self.pop_queue(self.in_queue)
            if node is None:
                time.sleep(0.01)
                if self.debug or self.plan_only:
                    break # stop iteration after one loop

                continue

            # turn abstract HL plan into motion plan, and collect trajectory samples + populate opt_sample buffers
            self.set_policies()
            self.write_log()
            self.refine_plan(node)

            inv_cov = self.agent.get_inv_cov()
                        
            if not self.plan_only:
                ## send LL samples to policy server, if training imitation
                for task in self.agent.task_list:
                    data = self.agent.get_opt_samples(task, clear=True)
                    opt_samples = [sample for sample in data if not len(sample.source_label) or sample.source_label.find('opt') >= 0]
                    expl_samples = [sample for sample in data if len(sample.source_label) and sample.source_label.find('opt') < 0]
                    
                    if len(opt_samples):
                        self.update_policy(opt_samples, label='optimal', inv_cov=inv_cov, task=task)

                    if len(expl_samples):
                        self.update_policy(expl_samples, label='dagger', inv_cov=inv_cov, task=task)

                # send HL samples to policy server
                self.run_hl_update()
                
                # add cont sample TODO where does this apply?
                cont_samples = self.agent.get_cont_samples()
                if len(cont_samples):
                    self.update_cont_network(cont_samples)

                step += 1

            if self.debug or self.plan_only:
                break # stop iteration after one loop


    def _log_solve_info(self, path, success, node, plan):
        self.n_failed += 0. if success else 1.
        n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_plans']
        with n_plans.get_lock():
            n_plans.value += 1

        if self.verbose and len(path):
            if node.nodetype.find('dagger') >= 0 and np.random.uniform() < 0.05:
                self.save_video(path, path[-1]._postsuc, lab='_suc_{}_dgr'.format(success))
            elif np.random.uniform() < 0.05:
                self.save_video(path, path[-1]._postsuc, lab='_suc_{}_opt'.format(success), annotate=True)
            elif not success and np.random.uniform() < 0.5:
                self.save_video(path, path[-1]._postsuc, lab='_suc_{}_opt_fail'.format(success), annotate=True)

        if self.verbose and self.render:
            for ind, batch in enumerate(info['to_render']):
                for next_path in batch:
                    if len(next_path):
                        print('BACKUP VIDEO:', next_path[-1].task)
                        self.save_video(next_path, next_path[-1]._postsuc, lab='_{}_backup_solve'.format(ind))

        self.log_path(path, 10)
        for step in path: step.source_label = node.nodetype

        if success and len(path):
            print(self.id, 
                  'succ. refine:', 
                  node.label, 
                  plan.actions[0].name, 
                  'rollout succ:', 
                  path[-1]._postsuc, 
                  path[-1].success, 
                  'goal:', 
                  self.agent.goal(0, path[-1].targets), )

        if len(path) and path[-1].success:
            n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_total']
            with n_plans.get_lock():
                n_plans.value += 1

        n_plan = self._hyperparams['policy_opt']['buffer_sizes']['n_plan_{}'.format(node.nodetype)]
        with n_plan.get_lock():
            n_plan.value += 1

        if not success:
            print('Opt failure from', node.label, node.nodetype)
            n_fail = self._hyperparams['policy_opt']['buffer_sizes']['n_plan_{}_failed'.format(node.nodetype)]
            with n_fail.get_lock():
                n_fail.value += 1


    def update_expert_demos(self, demos):
        for path in demos:
            for key in self.expert_demos:
                self.expert_demos[key].append([])
            for s in path:
                for t in range(s.T):
                    if not s.use_ts[t]: continue
                    self.expert_demos['acs'][-1].append(s.get(ACTION_ENUM, t=t))
                    self.expert_demos['obs'][-1].append(s.get_prim_obs(t=t))
                    self.expert_demos['ep_rets'][-1].append(1)
                    self.expert_demos['rews'][-1].append(1)
                    self.expert_demos['tasks'][-1].append(s.get(FACTOREDTASK_ENUM, t=t))
                    self.expert_demos['use_mask'][-1].append(s.use_ts[t])
        if self.cur_step % 5:
            np.save(self.expert_data_file, self.expert_demos)


    def log_node_info(self, node, success, path):
        key = 'n_ff'
        if node.label.find('post') >= 0:
            key = 'n_postcond'
        elif node.label.find('pre') >= 0:
            key = 'n_precond'
        elif node.label.find('mid') >= 0:
            key = 'n_midcond'
        elif node.label.find('rollout') >= 0:
            key = 'n_explore'

        self.infos[key] += 1
        self.infos['n_plans'] += 1
        for altkey in self.avgs:
            if altkey != key:
                self.avgs[altkey].append(0)
            else:
                self.avgs[altkey].append(1)

        failkey = key.replace('n_', 'n_fail_')
        if not success:
            self.fail_infos[failkey] += 1
            self.fail_infos['n_fail_plans'] += 1
            self.fail_avgs[failkey].append(0)
        else:
            self.fail_avgs[failkey].append(1)

        with self.policy_opt.buf_sizes[key].get_lock():
            self.policy_opt.buf_sizes[key].value += 1


    def get_log_info(self):
        info = {
                'time': time.time() - self.start_t,
                'optimization time': np.mean(self.plan_times),
                'plan length': np.mean(self.plan_horizons),
                'opt duration per ts': np.mean(self.plan_times) / np.mean(self.plan_horizons),
                }

        for key in self.infos:
            info[key] = self.infos[key]

        for key in self.fail_infos:
            info[key] = self.fail_infos[key]

        for key in self.fail_rollout_infos:
            info[key] = self.fail_rollout_infos[key]

        wind = 100
        for key in self.avgs:
            if len(self.avgs[key]):
                info[key+'_avg'] = np.mean(self.avgs[key][-wind:])

        for key in self.fail_avgs:
            if len(self.fail_avgs[key]):
                info[key+'_avg'] = np.mean(self.fail_avgs[key][-wind:])

        for key in self.opt_rollout_info:
            if len(self.opt_rollout_info[key]):
                info[key] = np.mean(self.opt_rollout_info[key][-wind:])

        if len(self.init_costs): info['mp initial costs'] = np.mean(self.init_costs[-wind:])
        if len(self.rolled_costs): info['mp rolled out costs'] = np.mean(self.rolled_costs[-wind:])
        if len(self.final_costs): info['mp optimized costs'] = np.mean(self.final_costs[-wind:])
        return info #self.log_infos


    def write_log(self):
        with open(self.motion_log, 'a+') as f:
            info = self.get_log_info()
            pp_info = pprint.pformat(info, depth=60)
            f.write(str(pp_info))
            f.write('\n\n')

