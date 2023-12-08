import pickle as pickle
from datetime import datetime
import numpy as np
import os
import pprint
import queue
import random
import sys
import time

from opentamp.software_constants import *
from opentamp.core.internal_repr.plan import Plan
from opentamp.policy_hooks.sample import Sample
from opentamp.policy_hooks.sample_list import SampleList
from opentamp.policy_hooks.utils.policy_solver_utils import *
from opentamp.policy_hooks.server import Server
from opentamp.policy_hooks.search_node import *

class TaskServer(Server):
    def __init__(self, hyperparams):
        os.nice(1)

        super(TaskServer, self).__init__(hyperparams)
        self.in_queue = self.task_queue
        self.out_queue = self.motion_queue
        self.prob_queue = []
        self.max_labels = self._hyperparams.get('max_label', -1)


    def run(self):
        while not self.stopped:
            self.find_task_plan()
            time.sleep(0.01)
            if self.debug or self.plan_only:
                break # stop iteration after one loop


    def find_task_plan(self):
        node = self.pop_queue(self.task_queue)
        if node is None or node.expansions > EXPAND_LIMIT:
            # if node is None:
            #     breakpoint()
            
            # if node is not None:
            #     breakpoint()
            
            if len(self.prob_queue):
                x, targets = self.prob_queue.pop()
                node = self.spawn_problem(x, targets)
                node.nodetype = 'queued'
                node.label = 'queued'
            elif node is None and self.motion_queue.full():
                return ## don't overwhelm the motion servers, MCMC updates are slow
            else:
                node = self.spawn_problem()
        
        plan_str = self.agent.hl_solver.run_planner(node.abs_prob, 
                                                    node.domain, 
                                                    node.prefix, 
                                                    label='{}_{}'.format(self.id, self.exp_id),)
        # try:
        #     plan_str = self.agent.hl_solver.run_planner(node.abs_prob, 
        #                                                 node.domain, 
        #                                                 node.prefix, 
        #                                                 label='{}_{}'.format(self.id, self.exp_id),)
        # except OSError as e:
        #     print('OSError in hl solve:', e)
        #     plan_str = Plan.IMPOSSIBLE

        if plan_str == Plan.IMPOSSIBLE:
            n_plan = self._hyperparams['policy_opt']['buffer_sizes']['n_plan_{}'.format(node.nodetype)]
            with n_plan.get_lock():
                n_plan.value += 1

            n_fail = self._hyperparams['policy_opt']['buffer_sizes']['n_plan_{}_failed'.format(node.nodetype)]
            with n_fail.get_lock():
                n_fail.value += 1

            with open(self.log_file, 'a+') as f:
                state_info = {(pname, aname): node.x0[self.agent.state_inds[pname, aname]] for (pname, aname) in self.agent.state_inds}
                info = '\n\n{} Task server could not plan for: {}\n{}\nExpansions: {}\n\n'.format(node.label, node.abs_prob, state_info, node.expansions)
                f.write(str(info))

            return
        
        # breakpoint()
 
        new_node = LLSearchNode(plan_str, 
                                prob=node.concr_prob, 
                                domain=node.domain,
                                initial=node.concr_prob.initial,
                                priority=node.priority,
                                ref_plan=node.ref_plan,
                                targets=node.targets,
                                x0=node.x0,
                                expansions=node.expansions+1,
                                label=node.label,
                                refnode=node,
                                nodetype=node.nodetype,
                                info=node.info,
                                replan_start=node.replan_start)


        # if self.config['seq']:
        #     import pma.backtrack_ll_solver_OSQP as bt_ll
        #     visual = len(os.environ.get('DISPLAY', '')) > 0
        #     if visual: self.agent.add_viewer()
        #     bt_ll.DEBUG = True
        #     plan = new_node.gen_plan(self.agent.hl_solver, self.agent.openrave_bodies, self.agent.ll_solver)
        #     if 'observation_model' in self._hyperparams.keys():
        #         plan.set_observation_model(self._hyperparams['observation_model'])
        #         plan.set_max_likelihood_obs(self._hyperparams['max_likelihood_obs'])            

        #     success, opt_suc, path, info = self.agent.backtrack_solve(plan, anum=0, x0=node.x0, targets=node.targets, permute=False, label='seq')
        #     #if not success or not opt_suc:
        #     #    import ipdb; ipdb.set_trace()
        #     new_init = self.agent.hl_solver.apply_action(plan.prob.initial, plan.actions[0])
        #     import ipdb; ipdb.set_trace()
        
        self.push_queue(new_node, self.motion_queue)


