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


# load args and hyperparams file automatically from the saved rollouts
with open(LOG_DIR+'namo_objs1_1/mac_test_49'+'/args.pkl', 'rb') as f:
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
current_id = 49
config['group_id'] = current_id
config['weight_dir'] = config['weight_dir_prefix']+'_{0}'.format(current_id)

mp_main = MultiProcessMain(config, load_at_spawn=False)

config['run_mcts_rollouts'] = False
config['run_alg_updates'] = False
config['run_hl_test'] = True
config['check_precond'] = False
config['share_buffers'] = False
config['load_render'] = True
#hyperparams['agent']['image_height']  = 256
#hyperparams['agent']['image_width']  = 256
descr = config.get('descr', '')
# hyperparams['weight_dir'] = hyperparams['weight_dir'].replace('exp_id0', 'rerun_{0}'.format(descr))
config['id'] = 'test'
mp_main.allocate_shared_buffers(config)
mp_main.allocate_queues(config)
config['policy_opt']['share_buffer'] = True
config['policy_opt']['buffers'] = config['buffers']
config['policy_opt']['buffer_sizes'] = config['buffer_sizes']
server = RolloutServer(config)
print(server)

# mp_main.run_test(mp_main.config)