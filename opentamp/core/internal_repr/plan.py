from .action import Action
import numpy as np
import argparse
import logging

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS

MAX_PRIORITY = 3

class Plan(object):
    """
    A plan has the following.

    params: dictionary of plan parameters, mapping name to object
    actions: list of Actions
    horizon: total number of timesteps for plan

    This class also defines methods for executing actions in simulation using the chosen viewer.
    """
    IMPOSSIBLE = "Impossible"

    def __init__(self, params, actions, horizon, env, determine_free=True, observation_model=None, sess=None):
        self.params = params
        self.backup = params
        self.actions = actions
        self.horizon = horizon
        self.observation_model = observation_model
        self.time = np.zeros((1, horizon))
        self.env = env
        self.initialized = False
        self._free_attrs = {}
        self._saved_free_attrs = {}
        self.sampling_trace = []
        self.hl_preds = []
        self.start = 0
        self.prior = None
        self.observation_model = None
        if determine_free:
            self._determine_free_attrs()


    @staticmethod
    def create_plan_for_preds(preds, env):
        ## preds is a list of pred, negated
        ## constructs a plan with a single action that
        ## enforces all the preds
        p_dicts = []
        params = set()
        for p, neg in preds:
            p_dicts.append({'pred': p, 'hl_info': 'pre', 'active_timesteps': (0, 0),
                            'negated': neg})
            params = params.union(p.params)
        params = list(params)
        a = Action(0, 'dummy', (0,0), params, p_dicts)
        param_dict = dict([(p.name, p) for p in params])
        return Plan(param_dict, [a], 1, env, determine_free=False)

    def _determine_free_attrs(self):
        for p in self.params.values():
            for k, v in list(p.__dict__.items()):
                if type(v) == np.ndarray and k not in p._free_attrs:
                    ## free variables are indicated as numpy arrays of NaNs
                    arr = np.zeros(v.shape, dtype='int32')
                    arr[np.isnan(v)] = 1
                    p._free_attrs[k] = arr

    def has_nan(self, active_ts = None):
        if not active_ts:
            active_ts = (0, self.horizon-1)

        for p in self.params.values():
            for k, v in list(p.__dict__.items()):
                if type(v) == np.ndarray:
                    if p.is_symbol() and np.any(np.isnan(v)):
                        print('Nan found in', p.name, k, v)
                        return True
                    if not p.is_symbol() and np.any(np.isnan(v[:, active_ts[0]:active_ts[1]+1])):
                        print('Nan found in', p.name, k, v)
                        return True
        return False

    def backup_params(self):
        for p in self.params:
            self.backup[p] = self.params[p].copy(self.horizon)

    def restore_params(self):
        for p in self.params.values():
            for attr in p._free_attrs:
                p._free_attrs[attr][:] = self.backup[p.name]._free_attrs[attr][:]
                getattr(p, attr)[:] = getattr(self.backup[p.name], attr)[:]

    def save_free_attrs(self):
        for p in self.params.values():
            p.save_free_attrs()

    def restore_free_attrs(self):
        for p in self.params.values():
            p.restore_free_attrs()

    def get_free_attrs(self):
        free_attrs = {}
        for p in self.params.values():
            free_attrs[p] = p.get_free_attrs()
        return free_attrs

    def store_free_attrs(self, attrs):
        for p in self.params.values():
            p.store_free_attrs(attrs[p])

    def freeze_up_to(self, t, exclude_types=[]):
        for p in self.params.values():
            skip = False
            for excl in exclude_types:
                if excl in p.get_type(True):
                    skip = True
                    continue
            if skip: continue
            p.freeze_up_to(t)

    def freeze_actions(self, anum):
        for i in range(anum):
            st, et = self.actions[i].active_timesteps
            for param in self.actions[i].params:
                if param.is_symbol():
                    for attr in param._free_attrs:
                        param._free_attrs[attr][:,0] = 0.
            for param in list(self.params.values()):
                if param.is_symbol(): continue
                for attr in param._free_attrs:
                    param._free_attrs[attr][:,st:et+1] = 0.
        for param in list(self.params.values()):
            if param.is_symbol(): continue
            for attr in param._free_attrs:
                param._free_attrs[attr][:,0] = 0.

    def execute(self):
        raise NotImplementedError

    def get_param(self, pred_type, target_ind, partial_assignment = None,
                  negated=False, return_preds=False):
        """
        get all target_ind parameters of the given predicate type
        partial_assignment is a dict that maps indices to parameter
        """
        if partial_assignment is None:
            partial_assignment = {}
        res = []
        if return_preds:
            preds = []
        for p in self.get_preds(incl_negated = negated):
            has_partial_assignment = True
            if p.get_type() != pred_type: continue
            for idx, v in list(partial_assignment.items()):
                if p.params[idx] != v:
                    has_partial_assignment = False
                    break
            if has_partial_assignment:
                res.append(p.params[target_ind])
                if return_preds: preds.append(p)
        res = np.unique(res)
        if return_preds:
            return res, np.unique(preds)
        return res

    def get_preds(self, incl_negated=True):
        res = []
        for a in self.actions:
            if incl_negated:
                res.extend([p['pred'] for p in a.preds if p['hl_info'] != 'hl_state'])
            else:
                res.extend([p['pred'] for p in a.preds if p['hl_info'] != 'hl_state' and not p['negated']])

        return res

    #@profile
    def get_failed_pred(self, active_ts=None, priority = MAX_PRIORITY, tol=1e-3, incl_negated=True, hl_ignore=False):
        #just return the first one for now
        t_min = self.horizon+1
        pred = None
        negated = False
        for action in self.actions:
            if active_ts is None:
                st, et = action.active_timesteps[0], action.active_timesteps[1]
            else:
                st, et = max(action.active_timesteps[0], active_ts[0]), min(action.active_timesteps[1], active_ts[1])

            a_st, a_et = action.active_timesteps
            if et > st:
                if a_st >= et: continue
                if a_et <= st: continue

            for pr in range(priority+1):
                for n, p, t in self.get_failed_preds(active_ts=(st,et), priority=pr, tol=tol, incl_negated=incl_negated):
                    if t < t_min and (not hl_ignore or not p.hl_ignore):
                        t_min = t
                        pred = p
                        negated = n
                if pred is not None:
                    return negated, pred, t_min
        return negated, pred, t_min

    #@profile
    def get_failed_preds(self, active_ts=None, priority = MAX_PRIORITY, tol=1e-3, incl_negated=True):
        if active_ts == None:
            active_ts = (0, self.horizon-1)
        failed = []
        for a in self.actions:
            st, et = a.active_timesteps
            if active_ts[1] > active_ts[0]:
                if st >= active_ts[1]: continue
                if et <= active_ts[0]: continue
            failed.extend(a.get_failed_preds(active_ts, priority, tol=tol, incl_negated=incl_negated))
        return failed

    def get_failed_preds_by_action(self, active_ts=None, priority = MAX_PRIORITY, tol=1e-3):
        if active_ts == None:
            active_ts = (0, self.horizon-1)
        failed = []
        for a in self.actions:
            failed.append(a.get_failed_preds(active_ts, priority, tol=tol))
        return failed

    def get_failed_preds_by_type(self, active_ts=None, priority = MAX_PRIORITY, tol=1e-3):
        if active_ts == None:
            active_ts = (0, self.horizon-1)
        failed = []
        for a in self.actions:
            failed.extend(a.get_failed_preds_by_type(active_ts, priority, tol=tol))
        return failed

    def satisfied(self, active_ts=None):
        if active_ts == None:
            active_ts = (0, self.horizon-1)
        success = True
        for a in self.actions:
            success &= a.satisfied(active_ts)
        return success

    def get_pr_preds(self, ts, priority):
        res = []
        for t in ts:
            for a in self.actions:
                res.extend(a.get_pr_preds(ts, priority))
        return res

    def get_active_preds(self, t):
        res = []
        for a in self.actions:
            start, end = a.active_timesteps
            if start <= t and end >= t:
                res.extend(a.get_active_preds(t))
        return res

    def check_cnt_violation(self, active_ts=None, priority = MAX_PRIORITY, tol = 1e-3):
        if active_ts is None:
            active_ts = (0, self.horizon-1)
        preds = [(negated, pred, t) for negated, pred, t in self.get_failed_preds(active_ts=active_ts, priority = priority, tol = tol)]
        cnt_violations = []
        for negated, pred, t in preds:
            viol = np.max(pred.check_pred_violation(t, negated=negated, tol=tol))
            cnt_violations.append(viol)
            if np.isnan(viol):
                print((negated, pred, t, 'NAN viol'))
            # print ("{}-{}\n".format(pred.get_type(), t), cnt_violations[-1])

        return cnt_violations

    def check_total_cnt_violation(self, active_ts=None, tol=1e-3):
        if active_ts is None:
            active_ts = (0, self.horizon-1)
        failed_preds = self.get_failed_preds(active_ts=active_ts, priority=3, tol=tol)
        cost = 0
        for failed in failed_preds:
            for t in range(active_ts[0], active_ts[1]+1):
                if t + failed[1].active_range[1] > active_ts[1]:
                    break

                try:
                    viol = failed[1].check_pred_violation(t, negated=failed[0], tol=tol)
                    # if np.any(np.isnan(viol)):
                    #     print('Nan constr violation for {0} at ts {1}'.format(failed, t))

                    if viol is not None:
                        cost += np.max(viol)
                except:
                    pass
        return cost

    def prefix(self, fail_step):
        """
            returns string representation of actions prior to faid_step
        """
        pre = []
        for act in self.actions:
            if act.active_timesteps[1] < fail_step:
                act_str = str(act).split()
                act_str = " ".join(act_str[:2] + act_str[4:]).upper()
                pre.append(act_str)
        return pre

    def get_plan_str(self):
        """
            return the corresponding plan str
        """
        plan_str = []
        for a in self.actions:
            plan_str.append(str(a))
        return plan_str

    def find_pred(self, pred_name):
        res = []
        for a in self.actions:
            res.extend([p['pred'] for p in a.preds if p['hl_info'] != 'hl_state' and p['pred'].get_type() == pred_name])
        return res

    def fill(self, plan, amin=0, amax=None):
        """
            fill self with trajectory from plan
        """
        if amax < 0 : return
        if amax is None:
            amax = len(self.actions)-1
        active_ts = self.actions[amin].active_timesteps[0], self.actions[amax].active_timesteps[1]
        for pname, param in self.params.items():
            if pname not in plan.params:
                raise AttributeError('Reference plan does not contain {0}'.format(pname))
            param.fill(plan.params[pname], active_ts)
        self.start = amax

    def get_values(self):
        vals = {}
        for pname, param in self.params.items():
            for attr in param._free_attrs:
                vals[pname, attr] = getattr(param, attr).copy()
        return vals

    def store_values(self, vals):
        for param, attr in vals:
            getattr(self.params[param], attr)[:] = vals[param, attr]

    def set_to_time(self, ts):
        for param in self.params.values():
            param.set_to_time(ts)

    def set_observation_model(self, observation_model):
        self.observation_model = observation_model

    # based off of hmm example from pyro docs
    def gen_samples(self, param, plan):
        # create unconditional or conditional model, depending
        if not plan:
            kernel = NUTS(self.observation_model)
        else:
            # create a conditioned model on the plan
            obs_dict = {'obs_'+str(i): torch.tensor(param.value.item()) for i in range(len(plan))}
            conditional_model = poutine.condition(self.observation_model, data=obs_dict)
            kernel = NUTS(conditional_model)

        # defaults taken from hmm.py script
        mcmc = MCMC(
            kernel,
            num_samples=1000,
            warmup_steps=5000,
            num_chains=10,
        )

        mcmc.run(plan)
        mcmc.summary(prob=0.95)  # for diagnostics

        return mcmc.get_samples()

    def initialize_beliefs(self):
        for param_key, param in self.params.items():
            if hasattr(param, 'samples'):
                # add samples by generating from prior
                param.samples = self.gen_samples(param, None)['belief_dist'].detach().numpy().reshape((-1, 1))

    # observation_models is an input dict matching belief parameters to
    def filter_beliefs(self, ll_plan):
        # construct a model object, over which we do inference, starting with uniform prior
        for param_key, param in self.params.items():
            if hasattr(param, 'samples'):
                param.samples = self.gen_samples(param, ll_plan)['belief_dist'].detach().numpy().reshape((-1, 1))
