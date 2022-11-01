import copy
import numpy as np

import opentamp.main as main
from opentamp.core.parsing import parse_domain_config, parse_problem_config
from opentamp.pma.hl_solver import FDSolver


def get_hl_solver(domain_fname):
    d_c = main.parse_file_to_dict(domain_fname)
    return FDSolver(d_c)

def plan_from_str(ll_plan_str, prob, domain, env, openrave_bodies, params=None, sess=None, use_tf=False, d_c=None, p_c=None):
    '''Convert a plan string (low level) into a plan object.'''
    domain_fname = domain
    if d_c is None:
        d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = FDSolver(d_c)
    if p_c is None:
        p_c = main.parse_file_to_dict(prob)
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, env, openrave_bodies, reuse_params=params, use_tf=use_tf, sess=sess, visual=False)
    plan = hls.get_plan(ll_plan_str, domain, problem, reuse_params=params)
    #plan = hls.get_plan(ll_plan_str, domain, problem)
    plan.d_c = d_c
    return plan

def get_tasks(file_name):
    with open(file_name, 'r+') as f:
        task_info = f.read().split(';')
        task_map = eval(task_info[0])
    return task_map

def get_task_durations(file_name):
    with open(file_name, 'r+') as f:
        task_info = f.read().split(';')
        task_durations = eval(task_info[1])
    return task_durations

def fill_params(plan_str, values):
    for i in range(len(plan_str)):
        plan_str[i] = plan_str[i].format(*values.append(i))
    return plan_str

def get_task_encoding(task_list):
    encoding = {}
    for i in range(len(task_list)):
        encoding[task_list[i]] = np.zeros((len(task_list)))
        encoding[task_list[i]][i] = 1

    return encoding

def compare_task_states(state1, state2):
    num_states = 0.
    num_states_match = 0.
    for key in state1:
        if key not in state2: continue
        num_states += 1.
        if state1[key] -- state2[key]:
            num_states_match += 1.

    return num_states_match / num_states

def get_hl_plan(prob, domain_file):
    with open(domain_file, 'r+') as f:
        domain = f.read()
    hl_solver = FFSolver(abs_domain=domain)
    return hl_solver._run_planner(domain, prob)

def parse_hl_plan(hl_plan):
    for i in range(len(hl_plan)):
        action = hl_plan[i].split()
        task = action[1]
        params = []
        for param in action[2:]:
            params.append(param.lower())

def parse_state(plan, failed_preds, ts, all_preds=[]):
    new_preds = [p for p in failed_preds if p is not None]
    reps = [p.get_rep() for p in new_preds]
    for a in plan.actions:
        a_st, a_et = a.active_timesteps
        if a_st > ts: break
        preds = copy.copy(a.preds)
        for p in all_preds:
            if type(p) is dict:
                preds.append(p)
            else:
                preds.append({'pred': p, 'active_timesteps':(0,0), 'hl_info':'hl_state', 'negated':False})

        for p in preds:
            if p['pred'].get_rep() in reps:
                continue
            reps.append(p['pred'].get_rep())
            st, et = p['active_timesteps']
            if p['pred'].hl_include: 
                new_preds.append(p['pred'])
                continue
            if p['pred'].hl_ignore:
                continue
            # Only check before the failed ts, previous actions fully checked while current only up to priority
            # TODO: How to handle negated?
            check_ts = ts - p['pred'].active_range[1]
            if st <= ts and check_ts >= 0 and et >= st:
                # hl_state preds aren't tied to ll state
                if p['pred'].hl_include:
                    new_preds.append(p['pred'])
                elif p['hl_info'] == 'hl_state':
                    if p['pred'].active_range[1] > 0: continue
                    old_vals = {}
                    for param in p['pred'].attr_inds:
                        for attr, _ in p['pred'].attr_inds[param]:
                            if param.is_symbol():
                                aval = getattr(plan.params[param.name], attr)[:,0]
                            else:
                                aval = getattr(plan.params[param.name], attr)[:,check_ts]
                            old_vals[param, attr] = getattr(param, attr)[:,0].copy()
                            getattr(param, attr)[:,0] = aval
                    if p['negated'] and not p['pred'].hl_test(0, tol=1e-3, negated=True):
                        new_preds.append(p['pred'])
                    elif not p['negated'] and p['pred'].hl_test(0, tol=1e-3):
                        new_preds.append(p['pred'])

                    for param, attr in old_vals:
                        getattr(param, attr)[:,0] = old_vals[param, attr]
                elif not p['negated'] and p['pred'].hl_test(check_ts, tol=1e-3):
                    new_preds.append(p['pred'])
                elif p['negated'] and not p['pred'].hl_test(check_ts, tol=1e-3, negated=True):
                    new_preds.append(p['pred'])
    return new_preds

