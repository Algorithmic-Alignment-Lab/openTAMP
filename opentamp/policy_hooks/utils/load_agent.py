from collections import OrderedDict
import imp
import os
import os.path
import psutil
import sys
import copy
import argparse
from datetime import datetime
from threading import Thread
import psutil
import sys
import time
import traceback

import numpy as np

from opentamp.policy_hooks.sample import Sample
from opentamp.policy_hooks.utils.policy_solver_utils import *
import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.policy_hooks.sample import Sample
from opentamp.policy_hooks.utils.load_task_definitions import *

from opentamp.core.util_classes.openrave_body import OpenRAVEBody


def load_agent(config):
    prob = config['prob']
    use_grb = config.get('use_grb', False)

    if use_grb:
        import opentamp.pma.backtrack_ll_solver_gurobi as bt_ll
    else:
        import opentamp.pma.backtrack_ll_solver_OSQP as bt_ll

    bt_ll.COL_COEFF = config.get('col_coeff', 0.)
    time_limit = config.get('time_limit', 14400)
    task_list = tuple(sorted(list(get_tasks(config['task_map_file']).keys())))
    cur_n_rollout = 0
    config['task_list'] = task_list
    task_encoding = get_task_encoding(task_list)
    plans = {}
    task_breaks = []
    goal_states = []

    domain, problem, hl_solver = get_planner_objs(config['meta_file'], config['acts_file'], config['prob_file'])

    plans, openrave_bodies, env = prob.get_plans()
    state_vector_include, action_vector_include, target_vector_include = config['get_vector'](config)
    print(state_vector_include)
    
    if plans:
        # using nonvacuous get_plans method
        dX, state_inds, dU, action_inds, symbolic_bound = utils.get_state_action_inds(list(plans.values())[0], 
                                                                                config['robot_name'], 
                                                                                config['attr_map'], 
                                                                                state_vector_include, 
                                                                                action_vector_include)
        target_dim, target_inds = utils.get_target_inds(list(plans.values())[0], 
                                                config['attr_map'], 
                                                target_vector_include)
    else:
        horizon = max([act.horizon for act in domain.action_schemas.values()])
        params = hl_solver._spawn_plan_params(problem, horizon)

        _, openrave_bodies, _ = prob.get_plans(params=params)

        # using vacuous get_plans method
        dX, state_inds, dU, action_inds, symbolic_bound = utils.get_state_action_inds_from_params(params, 
                                                                                config['robot_name'], 
                                                                                config['attr_map'], 
                                                                                state_vector_include, 
                                                                                action_vector_include)
        target_dim, target_inds = utils.get_target_inds_from_params(params, 
                                                config['attr_map'], 
                                                target_vector_include)
        
    
    x0, targets = prob.get_random_initial_state_vec(config, False, dX, state_inds, 0)

    im_h = config.get('image_height', utils.IM_H)
    im_w = config.get('image_width', utils.IM_W)
    im_c = config.get('image_channels', utils.IM_C)
    config['image_height'] = im_h
    config['image_width'] = im_w
    config['image_channels'] = im_c
    config['domain'] = domain
    config['problem'] = problem
    # for plan in list(plans.values()):
    #     plan.state_inds = state_inds
    #     plan.action_inds = action_inds
    #     plan.dX = dX
    #     plan.dU = dU
    #     plan.symbolic_bound = symbolic_bound
    #     plan.target_dim = target_dim
    #     plan.target_inds = target_inds

    sensor_dims = {
        utils.DONE_ENUM: 1,
        utils.TASK_DONE_ENUM: 1,
        utils.STATE_ENUM: symbolic_bound,
        utils.ACTION_ENUM: dU,
        utils.TRAJ_HIST_ENUM: int(dU*config['hist_len']),
        utils.STATE_DELTA_ENUM: int(symbolic_bound*config['hist_len']),
        utils.STATE_HIST_ENUM: int((1+symbolic_bound)*config['hist_len']),
        utils.TASK_ENUM: len(task_list),
        utils.TARGETS_ENUM: target_dim,
        utils.ONEHOT_TASK_ENUM: len(list(plans.keys())),
        utils.IM_ENUM: im_h * im_w * im_c,
        utils.LEFT_IMAGE_ENUM: im_h * im_w * im_c,
        utils.RIGHT_IMAGE_ENUM: im_h * im_w * im_c,
    }

    for enum in config['sensor_dims']:
        sensor_dims[enum] = config['sensor_dims'][enum]

    if 'cont_obs_include' not in config:
        config['cont_obs_include'] = copy.copy(config['prim_obs_include'])

    if config.get('add_action_hist', False):
        config['prim_obs_include'].append(utils.TRAJ_HIST_ENUM)

    if config.get('add_obs_delta', False):
        config['prim_obs_include'].append(utils.STATE_DELTA_ENUM)

    if config.get('add_task_hist', False):
        config['prim_obs_include'].append(utils.TASK_HIST_ENUM)

    if config.get('add_hl_image', False):
        config['prim_obs_include'].append(utils.IM_ENUM)
        config['load_render'] = True

    if config.get('add_cont_image', False):
        config['cont_obs_include'].append(utils.IM_ENUM)
        config['load_render'] = True

    if config.get('add_image', False):
        config['obs_include'].append(utils.IM_ENUM)
        config['load_render'] = True

    options = prob.get_prim_choices(task_list)
    if config.get('flat', False):
        obs_include = config['obs_include']
        config['obs_include'] = []
        for enum in obs_include:
            if enum not in [END_POSE_ENUM, END_ROT_ENUM] and enum not in options.keys():
                config['obs_include'].append(enum)

        for enum in config['prim_obs_include']:
            if enum not in config['obs_include']:
                config['obs_include'].append(enum)

    prim_dims = OrderedDict({})
    config['prim_dims'] = prim_dims
    ind = len(task_list)
    for enum in options:
        if enum == utils.TASK_ENUM: continue
        if hasattr(options[enum], '__len__'):
            n_options = len(options[enum])
        else:
            n_options = options[enum]

        next_ind = ind+n_options
        prim_dims[enum] = n_options
        ind = next_ind

    for enum in prim_dims:
        sensor_dims[enum] = prim_dims[enum]

    prim_ind = 0
    cont_ind = 0
    prim_bounds = []
    cont_bounds = []
    prim_out = []
    cont_out = []
    prim_bounds.append((0, len(task_list)))
    prim_out.append(utils.TASK_ENUM)
    prim_ind = len(task_list)
    for enum in options:
        if enum == utils.TASK_ENUM: continue
        if hasattr(options[enum], '__len__'):
            prim_bounds.append((prim_ind, prim_ind+sensor_dims[enum]))
            prim_ind += sensor_dims[enum]
            prim_out.append(enum)
        else:
            cont_bounds.append((cont_ind, cont_ind+sensor_dims[enum]))
            cont_ind += sensor_dims[enum]
            cont_out.append(enum)

    for enum in config['prim_out_include']:
        if enum == utils.TASK_ENUM or enum in options: continue
        prim_bounds.append((prim_ind, prim_ind+sensor_dims[enum]))
        prim_ind += sensor_dims[enum]
        prim_out.append(enum)

    sensor_dims[utils.TASK_HIST_ENUM] = int(config.get('task_hist_len', 0) * prim_ind)

    aux_bounds = []
    if len(prim_bounds) > len(options.keys()):
        aux_bounds = prim_bounds[len(options.keys()):]
        prim_bounds = prim_bounds[:len(options.keys())]

    config['prim_bounds'] = prim_bounds
    config['cont_bounds'] = cont_bounds
    config['prim_dims'] = prim_dims
    config['aux_bounds'] = aux_bounds

    # breakpoint()

    config['target_f'] = None # prob.get_next_target
    config['encode_f'] = None # prob.sorting_state_encode

    agent_config = {
        'num_objs': config['num_objs'],
        'num_targs': config['num_targs'],
        'type': config['agent_type'],
        'x0': x0,
        'targets': targets,
        'task_list': task_list,
        # 'plans': plans,
        'task_breaks': task_breaks,
        'task_encoding': task_encoding,
        'state_inds': state_inds,
        'action_inds': action_inds,
        'target_inds': target_inds,
        'dU': dU,
        'dX': symbolic_bound,
        'symbolic_bound': symbolic_bound,
        'target_dim': target_dim,
        'get_plan': None, # prob.get_plan,
        'sensor_dims': sensor_dims,
        'state_include': config['state_include'],
        'obs_include': config['obs_include'],
        'prim_obs_include': config['prim_obs_include'],
        'prim_out_include': prim_out,
        'cont_obs_include': config['cont_obs_include'],
        'cont_out_include': cont_out,
        'solver': None,
        'rollout_seed': config.get('rollout_seed', False),
        'num_objs': config['num_objs'],
        'obj_list': [],
        'image_width': im_w,
        'image_height': im_h,
        'image_channels': im_c,
        'hist_len': config.get('hist_len', 0),
        'T': 1,
        'viewer': config.get('viewer', False),
        'model': None,
        'env': env,
        'openrave_bodies': openrave_bodies,
        'n_dirs': config.get('n_dirs', 1),
        'prob': prob,
        'attr_map': config['attr_map'],
        'prim_dims': prim_dims,
        'mp_solver_type': config['mp_solver_type'],
        'robot_name': config['robot_name'],
        'split_nets': config.get('split_nets', True),
        'master_config': config,
        'gym_env_type': config.get('gym_env_type', None),
        'domain': domain,
        'problem': problem
    }

    agent_config['agent_load'] = True
    return agent_config


def build_agent(config):
    agent = config['type'](config)
    return agent

