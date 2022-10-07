"""
Defines utility functions for planning in the sorting domain
"""
import copy
from collections import OrderedDict
import itertools
import numpy as np
import random
import time

from opentamp.core.internal_repr.plan import Plan
from opentamp.core.util_classes.namo_predicates import dsafe
from opentamp.core.util_classes.openrave_body import *
from pma.hl_solver import FFSolver
from opentamp.src.policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str
from opentamp.src.policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils

NO_COL = True
NUM_OBJS = 4
NUM_TARGS = 4
SORT_CLOSET = False
USE_PERTURB = False
OPT_MCTS_FEEDBACK = True
N_GRASPS = 4
FIX_TARGETS = False

CONST_TARGETS = False
CONST_ORDER = False

domain_file = "../domains/robot_wiping_domain/right_wipe_onlytable.domain"
mapping_file = "policy_hooks/wiping/robot_task_mapping"

def prob_file(descr=None):
    return "../domains/robot_wiping_domain/probs/simple_move_onlytable.prob"


def get_prim_choices(task_list=None):
    out = OrderedDict({})
    if task_list is None:
        out[utils.TASK_ENUM] = sorted(list(get_tasks(mapping_file).keys()))
    else:
        out[utils.TASK_ENUM] = sorted(list(task_list))
    out[utils.REG_ENUM] = [] #TODO: Add all region poses here.
    return out


def get_vector(config):
    state_vector_include = {
        'sawyer': ['right', 'right_ee_pos', 'right_ee_rot', 'right_gripper', 'pose']
    }
    # TODO: Add dirt objects and locations.
    # TODO: write code to read dirt positions and reset the simulator
    # to a state where the dirt is at these poses. Do this in the 
    # robo_wiping.py.
    state_vector_include['table'] = ['pose']

    action_vector_include = {
        'sawyer': ['right', 'right_gripper']
    }

    target_vector_include = {}

    return state_vector_include, action_vector_include, target_vector_include


# Will define this later in agent class.
def get_random_initial_state_vec(config, plans, dX, state_inds, conditions):
# Information is track by the environment
    return [np.zeros(dX)], [{'cereal_end_target': np.zeros(3)}]
    x0s = []
    targ_maps = []
    robot = list(plans.values())[0].params['sawyer']
    body = robot.openrave_body
    for i in range(conditions):
        x0 = np.zeros((dX,))

        ee_sol = None
        quat = (0,1,0,0)
        while ee_sol is None:
            ee_x = np.random.uniform(0.2, 0.8)
            ee_y = np.random.uniform(0.2, 0.8)
            ee_z = np.random.uniform(0.15, 0.45)
            body.set_dof({'right': np.zeros(7)})
            ee_sol = body.get_ik_from_pose((ee_x, ee_y, ee_z), quat, 'right')
            ee_info = body.fwd_kinematics('right', dof_map={'right': ee_sol})
            if np.any(np.abs(np.array(ee_info['quat']) - np.array(quat)) > 1e-2):
                ee_sol = None

        x0[state_inds['sawyer', 'right']] = ee_sol
        x0[state_inds['sawyer', 'right_ee_pos']] = ee_info['pos']
        can_locs = copy.deepcopy(possible_can_locs)
        locs = []
        spacing = 0.04
        valid = [1 for _ in range(len(can_locs))]
        while len(locs) < NUM_OBJS:
            locs = []
            random.shuffle(can_locs)
            valid = [1 for _ in range(len(can_locs))]
            for j in range(config['num_objs']):
                for n in range(0, len(can_locs)):
                    if valid[n]:
                        locs.append(can_locs[n])
                        valid[n] = 0
                        for m in range(n+1, len(can_locs)):
                            if not valid[m]: continue
                            if np.linalg.norm(np.array(can_locs[n]) - np.array(can_locs[m])) < spacing:
                                valid[m] = 0
                        break
            spacing -= 0.01

        spacing = 0.04
        targs = []
        can_targs = [can_locs[i] for i in range(len(can_locs)) if (valid[i] or not NO_COL)]
        old_valid = copy.copy(valid)
        while not FIX_TARGETS and len(targs) < config['num_targs']:
            targs = []
            pr2_loc = locs[0]
            random.shuffle(can_targs)
            valid = [1 for _ in range(len(can_targs))]
            for m in range(0, len(can_targs)):
                if np.linalg.norm(np.array(pr2_loc) - np.array(can_targs[m])) < spacing:
                    valid[m] = 0
            for j in range(config['num_targs']):
                for n in range(0, len(can_targs)):
                    if valid[n]:
                        targs.append(can_targs[n])
                        valid[n] = 0
                        for m in range(n+1, len(can_targs)):
                            if not valid[m]: continue
                            if np.linalg.norm(np.array(can_targs[n]) - np.array(can_targs[m])) < spacing:
                                valid[m] = 0
                        break

            spacing -= 0.1

        for l in range(len(locs)):
            locs[l] = np.array(locs[l])

        x0[state_inds['sawyer', 'right']] = R_ARM_INIT
        x0[state_inds['sawyer', 'right_ee_pos']] = RIGHT_INIT_EE
        x0[state_inds['sawyer', 'right_gripper']] = 0.02
        x0 = x0.round(4)

        for o in range(NUM_OBJS):
            x0[state_inds['cloth{0}'.format(o), 'pose']] = locs[o]
        x0s.append(x0)
        if FIX_TARGETS:
            targ_range = list(range(config['num_objs'] - config['num_targs']))
            inds = list(range(len(EMD_TARGETS))) if CONST_TARGETS else np.random.permutation(list(range(len(END_TARGETS))))
            next_map = {'cloth{0}_end_target'.format(no): END_TARGETS[o] for no, o in enumerate(inds[:config['num_objs']])}
            inplace = targ_range if CONST_ORDER else np.random.choice(list(range(config['num_objs'])), len(targ_range), replace=False)
            for n in targ_range:
                x0[state_inds['cloth{0}'.format(inplace[n]), 'pose']] = END_TARGETS[inds[inplace[n]]]
            next_map.update({'end_target_{0}'.format(i): END_TARGETS[i] for i in range(len(END_TARGETS))})
        else:
            inds = np.random.permutation(list(range(config['num_objs'])))
            next_map = {'can{0}_end_target'.format(o): targs[no] for no, o in enumerate(inds[:config['num_targs']])}
            if config['num_targs'] < config['num_objs']:
                next_map.update({'can{0}_end_target'.format(o): locs[o+1] for o in inds[config['num_targs']:config['num_objs']]})
        for a in range(n_aux):
            if a == 0:
                next_map['aux_target_{0}'.format(a)] = (0, 0)
            elif a % 2:
                next_map['aux_target_{0}'.format(a)] = (-int(a/2.+0.5), 0)
            else:
                next_map['aux_target_{0}'.format(a)] = (int(a/2.+0.5), 0)
        targ_maps.append(next_map)
    return x0s, targ_maps


def get_plans(use_tf=False):
    tasks = get_tasks(mapping_file)
    task_ids = sorted(list(get_tasks(mapping_file).keys()))
    prim_options = get_prim_choices()
    plans = {}
    openrave_bodies = {}
    env = None
    params = None
    sess = None
    st = time.time()
    for task in task_ids:
        params = None
        next_task_str = copy.deepcopy(tasks[task])
        for i in range(len(prim_options[utils.OBJ_ENUM])):
            for j in range(len(prim_options[utils.TARG_ENUM])):
                obj = prim_options[utils.OBJ_ENUM][i]
                targ = prim_options[utils.TARG_ENUM][j]
                new_task_str = []
                for step in next_task_str:
                    new_task_str.append(step.format(obj, targ))
                plan = plan_from_str(new_task_str, prob_file(), domain_file, env, openrave_bodies, params=params, sess=sess, use_tf=use_tf)
                params = plan.params
                plans[(task_ids.index(task), i, j)] = plan
                if env is None:
                    env = plan.env
                    for param in list(plan.params.values()):
                        if hasattr(param, 'geom'):
                            if not hasattr(param, 'openrave_body') or param.openrave_body is None:
                                param.openrave_body = OpenRAVEBody(env, param.name, param.geom)
                            openrave_bodies[param.name] = param.openrave_body
    return plans, openrave_bodies, env

