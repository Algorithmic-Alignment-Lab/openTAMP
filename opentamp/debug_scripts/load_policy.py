import argparse
import copy
import imp
import importlib
import os
import pickle
import random
import shutil
import sys
import time

from opentamp.policy_hooks.multiprocess_main import MultiProcessMain
from opentamp.policy_hooks.utils.file_utils import LOG_DIR, load_config
# from ..policy_hooks.run_training import argsparser
from policy_hooks.rollout_server import RolloutServer
from policy_hooks.multiprocess_main import MultiProcessMain
from policy_hooks.utils.policy_solver_utils import *


# load args and hyperparams file automatically from the saved rollouts
with open(LOG_DIR+'namo_objs1_1/test_crash_save_14'+'/args.pkl', 'rb') as f:
    args = pickle.load(f)

exps = None
if args.file == "":
    exps = [[args.config]]

print(('LOADING {0}'.format(args.file)))
if exps is None:
    exps = []
    with open(args.file, 'r+') as f:
        exps = eval(f.read())
        
exps_info = exps
n_objs = args.nobjs if args.nobjs > 0 else None
n_targs = args.nobjs if args.nobjs > 0 else None
#n_targs = args.ntargs if args.ntargs > 0 else None
# if len(args.test):
#     sys.path.insert(1, LOG_DIR+args.test)
#     exps_info = [['hyp']]
#     old_args = args
#     with open(LOG_DIR+args.test+'/args.pkl', 'rb') as f:
#         args = pickle.load(f)
#     args.soft_eval = old_args.soft_eval
#     args.test = old_args.test
#     args.use_switch = old_args.use_switch
#     args.ll_policy = args.test
#     args.hl_policy = args.test
#     args.load_render = old_args.load_render
#     args.eta = old_args.eta
#     args.descr = old_args.descr
#     args.easy = old_args.easy
#     var_args = vars(args)
#     old_vars = vars(old_args)
#     for key in old_vars:
#         if key not in var_args: var_args[key] = old_vars[key]

if args.hl_retrain:
    sys.path.insert(1, LOG_DIR+args.hl_data)
    exps_info = [['hyp']]

config, config_module = load_config(args)

print('\n\n\n\n\n\nLOADING NEXT EXPERIMENT\n\n\n\n\n\n')
old_dir = config['weight_dir_prefix']
old_file = config['task_map_file']
config = {'args': args, 
            'task_map_file': old_file}
config.update(vars(args))
config['source'] = args.config
config['weight_dir_prefix'] = old_dir
current_id = 14
config['group_id'] = current_id
config['weight_dir'] = config['weight_dir_prefix']+'_{0}'.format(current_id)

mp_main = MultiProcessMain(config, load_at_spawn=False)

mp_main.config['run_mcts_rollouts'] = False
mp_main.config['run_alg_updates'] = False
mp_main.config['run_hl_test'] = True
mp_main.config['check_precond'] = False
mp_main.config['share_buffers'] = False
mp_main.config['load_render'] = True
#hyperparams['agent']['image_height']  = 256
#hyperparams['agent']['image_width']  = 256
descr = mp_main.config.get('descr', '')
# hyperparams['weight_dir'] = hyperparams['weight_dir'].replace('exp_id0', 'rerun_{0}'.format(descr))
mp_main.config['id'] = 'test'
mp_main.allocate_shared_buffers(mp_main.config)
mp_main.allocate_queues(mp_main.config)
mp_main.config['policy_opt']['share_buffer'] = True
mp_main.config['policy_opt']['buffers'] = mp_main.config['buffers']
mp_main.config['policy_opt']['buffer_sizes'] = mp_main.config['buffer_sizes']
server = RolloutServer(mp_main.config)

server.set_policies()
server.agent.replace_cond(0)
server.agent.reset(0)
server.agent._eval_mode = True
server.policy_opt.restore_ckpts(None)

init_t = time.time()
server.agent.debug = False
prim_opts = server.agent.prob.get_prim_choices(server.agent.task_list)
n_targs = list(range(len(prim_opts[OBJ_ENUM])))
res = []
ns = [server.config['num_targs']]
if server.config['curric_thresh'] > 0:
    ns = list(range(1, server.config['num_targs']+1))
n = np.random.choice(ns)
s = []
x0 = server.agent.x0[0]
targets = server.agent.target_vecs[0].copy()
print(server.agent.mjc_env.get_item_pos('pr2'))
print(server.agent.mjc_env.get_item_pos('can0'))
print(server.agent.mjc_env.physics.data.xpos)
print(server.agent.mjc_env.physics.model.name2id('pr2', 'body'))
print(server.agent.mjc_env.physics.model.name2id('can0', 'body'))
print(server.agent.policies)

# server.agent.mjc_env.set_item_pos('pr2', np.array([0.0, 0.0, 0.5]))
# server.agent.mjc_env.set_item_pos('can0', np.array([1.0, 0.0, 0.5]))

print('Worked!')

for t in range(n, n_targs[-1]):
    obj_name = prim_opts[OBJ_ENUM][t]
    targ_name = '{0}_end_target'.format(obj_name)
    # print(obj_name)
    # print(targ_name)
    if (targ_name, 'value') in server.agent.target_inds:
        targets[server.agent.target_inds[targ_name, 'value']] = x0[server.agent.state_inds[obj_name, 'pose']]

# if rlen is None: rlen = server.agent.rlen
# hor = server.agent.hor
# nt = 500 # rlen * hor

goal = server.agent.goal(0, targets)
# val, path = server.test_run(x0, targets, 20, hl=True, soft=server.config['soft_eval'], eta=None, lab=-5, hor=25)
# if goal not in server.suc_per_goal:
#     server.suc_per_goal[goal] = []
# server.suc_per_goal[goal].append(val)

state = server.agent.x0[0]
print(state)
max_t = 20
eta=None
def task_f(sample, t, curtask):
    return server.get_task(sample.get_X(t=t), sample.targets, curtask, soft)
server.agent.reset_to_state(state)
path = []
val = 0
nout = len(server.agent._hyperparams['prim_out_include'])
l = None
t = 0
if eta is not None: server.eta = eta 
old_eta = server.eta
debug = np.random.uniform() < 0.1
while t < max_t and val < 1-1e-2 and server.agent.feasible_state(state, targets):
    l = server.get_task(state, targets, l, False)
    task_name = server.task_list[l[0]]
    pol = server.agent.policies[task_name]
    s = server.agent.sample_task(pol, 0, state, l, noisy=False, task_f=task_f, skip_opt=True, hor=server.agent.hor, policies=server.agent.policies)
    val = 1 - server.agent.goal_f(0, s.get_X(s.T-1), targets)
    t += 1
    state = s.end_state # s.get_X(s.T-1)
    print(state)

#print(targets)
#print(goal)
#print(val)
#for p in path:
    #print(p.get_X()[server.agent.state_inds['can0', 'pose']])
#print(path[-1].get_X()[server.agent.action_inds[]])
#print(path[-1].agent.action_inds['pr2', 'pose'])

# mp_main.run_test(mp_main.config)
