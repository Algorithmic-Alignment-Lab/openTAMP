"""
Defines utility functions for planning in the sorting domain
"""
import copy
from collections import OrderedDict
import itertools
import numpy as np
import random
import time

import opentamp
from opentamp.core.internal_repr.plan import Plan
from opentamp.core.util_classes.namo_predicates import dsafe
from opentamp.core.util_classes.openrave_body import *
from opentamp.policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str
from opentamp.policy_hooks.utils.policy_solver_utils import *
import opentamp.policy_hooks.utils.policy_solver_utils as utils

NO_COL = True
NUM_OBJS = 4
NUM_TARGS = 4
N_HUMAN = 0#6
SORT_CLOSET = False
USE_PERTURB = False
OPT_MCTS_FEEDBACK = True
N_GRASPS = 4
FIX_TARGETS = True

CONST_TARGETS = False
CONST_ORDER = False

domain_file = opentamp.__path__._path[0] + "/domains/namo_domain/namo_current_holgrip.domain"
mapping_file = "policy_hooks/namo/grip_task_mapping"

descriptor = 'namo_{0}_obj_sort_closet_{1}_perturb_{2}_feedback_to_tree_{3}'.format(NUM_OBJS, SORT_CLOSET, USE_PERTURB, OPT_MCTS_FEEDBACK)

END_TARGETS =[(0., 5.8), (0., 5.), (0., 4.)] if SORT_CLOSET else []
END_TARGETS.extend([(1., 2.),
                   (-1., 2.),
                   (2.8, 2.),
                   (-2.8, 2.),
                   (-4.6, 2.),
                   (4.6, 2.),
                   (6.4, 2.),
                   ])


#END_TARGETS.extend([
#                   (6.4, 2.2),
#                   (-6.4, 2.2),
#                   (6.4, -8.2),
#                   (-6.4, -8.2),
#                   (3.2, 2.2),
#                   (-3.2, 2.2),
#                   (3.2, -8.2),
#                   (-3.2, -8.2),
#                   ])

n_aux = 0
possible_can_locs = [(0, 57), (0, 50), (0, 43), (0, 35)] if SORT_CLOSET else []
MAX_Y = 25
#possible_can_locs.extend(list(itertools.product(list(range(-45, 45, 4)), list(range(-40, -10, 2)))))
#possible_can_locs.extend(list(itertools.product(list(range(-70, 70, 2)), list(range(-75, 0, 2)))))
possible_can_locs.extend(list(itertools.product(list(range(-80, 80, 2)), list(range(-60, 0, 2)))))


for i in range(len(possible_can_locs)):
    loc = list(possible_can_locs[i])
    loc[0] *= 0.1
    loc[1] *= 0.1
    possible_can_locs[i] = tuple(loc)


def prob_file(descr=None):
    return opentamp.__path__._path[0] + "/domains/belief_space_domain/probs/namo_mod.prob"


def get_prim_choices(task_list=None):
    out = OrderedDict({})
    if task_list is None:
        out[utils.TASK_ENUM] = sorted(list(get_tasks(mapping_file).keys()))
    else:
        out[utils.TASK_ENUM] = sorted(list(task_list))
    out[utils.OBJ_ENUM] = ['can{0}'.format(i) for i in range(NUM_OBJS)]
    out[utils.TARG_ENUM] = []
    for i in range(n_aux):
        out[utils.TARG_ENUM] += ['aux_target_{0}'.format(i)]
    for i in range(len(END_TARGETS)):
        out[utils.TARG_ENUM] += ['end_target_{0}'.format(i)]
    #out[utils.GRASP_ENUM] = ['grasp{0}'.format(i) for i in range(N_GRASPS)]
    #out[utils.ABS_POSE_ENUM] = 2
    return out


def get_vector(config):
    state_vector_include = {
        'pr2': ['pose', 'gripper', 'theta', 'vel']
    }
    for i in range(config['num_objs']):
        state_vector_include['can{0}'.format(i)] = ['pose']

    for i in range(N_HUMAN):
        state_vector_include['human{}'.format(i)] = ['pose']

    action_vector_include = {
        'pr2': ['pose', 'gripper', 'theta']
    }

    target_vector_include = {
        'can{0}_end_target'.format(i): ['value'] for i in range(config['num_objs'])
    }
    for i in range(n_aux):
        target_vector_include['aux_target_{0}'.format(i)] = ['value']
    for i in range(len(END_TARGETS)):
        target_vector_include['end_target_{0}'.format(i)] = ['value']

    return state_vector_include, action_vector_include, target_vector_include


def get_random_initial_state_vec(config, plans, dX, state_inds, conditions):
# Information is track by the environment
    x0s = []
    targ_maps = []
    conditions = max(1, conditions)
    for i in range(conditions):
        x0 = np.zeros((dX,))

        # x0[state_inds['pr2', 'pose']] = np.random.uniform([-3, -4], [3, -2]) # [0, -2]
        # x0[state_inds['pr2', 'pose']] = np.random.uniform([-3, -1], [3, 1])
        can_locs = copy.deepcopy(possible_can_locs)
        # can_locs = copy.deepcopy(END_TARGETS)
        locs = []
        pr2_loc = None
        spacing = 2.4
        valid = [1 for _ in range(len(can_locs))]
        while len(locs) < config['num_objs'] + N_HUMAN + 1:
            locs = []
            random.shuffle(can_locs)
            pr2_loc = can_locs[0]
            locs.append(pr2_loc)
            valid = [1 for _ in range(len(can_locs))]
            valid[0] = 0
            for m in range(1, len(can_locs)):
                if np.linalg.norm(np.array(pr2_loc) - np.array(can_locs[m])) < spacing:
                    valid[m] = 0

            for j in range(config['num_objs']):
                for n in range(1, len(can_locs)):
                    if valid[n]:
                        locs.append(can_locs[n])
                        valid[n] = 0
                        for m in range(n+1, len(can_locs)):
                            if not valid[m]: continue
                            if np.linalg.norm(np.array(can_locs[n]) - np.array(can_locs[m])) < spacing:
                                valid[m] = 0
                        break

            for j in range(N_HUMAN):
                for n in range(1, len(can_locs)):
                    if valid[n]:
                        locs.append(can_locs[n])
                        valid[n] = 0
                        for m in range(n+1, len(can_locs)):
                            if not valid[m]: continue
                            if np.linalg.norm(np.array(can_locs[n]) - np.array(can_locs[m])) < spacing:
                                valid[m] = 0
                        break
            spacing -= 0.1

        targs = []
        old_valid = copy.copy(valid)

        for l in range(len(locs)):
            locs[l] = np.array(locs[l])

        x0[state_inds['pr2', 'pose']] = locs[0]
        x0[state_inds['pr2', 'theta']] = np.random.uniform(-np.pi, np.pi)
        for o in range(config['num_objs']):
            x0[state_inds['can{0}'.format(o), 'pose']] = locs[o+1]

        for h in range(N_HUMAN):
            x0[state_inds['human{}'.format(h), 'pose']] = locs[h+config['num_objs']+1]

        x0[state_inds['pr2', 'gripper']] = -0.1
        x0s.append(x0)
        inds = np.random.permutation(list(range(len(END_TARGETS))))
        next_map = {'can{0}_end_target'.format(no): END_TARGETS[o] for no, o in enumerate(inds[:config['num_objs']])}

        next_map.update({'end_target_{0}'.format(i): END_TARGETS[i] for i in range(len(END_TARGETS))})
        for a in range(n_aux):
            if a == 0:
                next_map['aux_target_{0}'.format(a)] = (0, 0)
            elif a % 2:
                next_map['aux_target_{0}'.format(a)] = (-int(a/2.+0.5), 0)
            else:
                next_map['aux_target_{0}'.format(a)] = (int(a/2.+0.5), 0)
        targ_maps.append(next_map)
    return x0s, targ_maps

def parse_hl_plan(hl_plan):
    plan = []
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        task = act_params[1].lower()
        next_params = [p.lower() for p in act_params[2:]]
        plan.append((task, next_params))
    return plan

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
        next_task_str = copy.deepcopy(tasks[task])
        if task.find('move') >= 0:
            for i in range(len(prim_options[utils.OBJ_ENUM])):
                obj = prim_options[utils.OBJ_ENUM][i]
                new_task_str = []
                for step in next_task_str:
                    new_task_str.append(step.format(obj, '', ''))
                plan = plan_from_str(new_task_str, prob_file(), domain_file, env, openrave_bodies, params=params, sess=sess, use_tf=use_tf)
                params = plan.params
                if env is None:
                    env = plan.env
                    for param in list(plan.params.values()):
                        if hasattr(param, 'geom'):
                            if not hasattr(param, 'openrave_body') or param.openrave_body is None:
                                param.openrave_body = OpenRAVEBody(env, param.name, param.geom)
                            openrave_bodies[param.name] = param.openrave_body

                for j in range(len(prim_options[utils.TARG_ENUM])):
                    plans[(task_ids.index(task), i, j)] = plan

        else:
            for i in range(len(prim_options[utils.OBJ_ENUM])):
                for j in range(len(prim_options[utils.TARG_ENUM])):
                    obj = prim_options[utils.OBJ_ENUM][i]
                    targ = prim_options[utils.TARG_ENUM][j]
                    new_task_str = []
                    for step in next_task_str:
                        new_task_str.append(step.format(obj, targ, 'grasp0'))
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

def get_end_targets(num_cans=NUM_OBJS, num_targs=NUM_OBJS, targs=None, randomize=False, possible_locs=END_TARGETS):
    raise Exception('Bad method call')
    target_map = {}
    inds = list(range(NUM_TARGS)) # np.random.permutation(range(num_targs))
    for n in range(num_cans):
        if n > num_targs and targs is not None:
            target_map['can{0}_end_target'.format(n)] = np.array(targs[n])
        else:
            if randomize:
                ind = inds[n]
            else:
                ind = n

            target_map['can{0}_end_target'.format(n)] = np.array(possible_locs[ind])

    if SORT_CLOSET:
        target_map['middle_target'] = np.array([0., 0.])
        target_map['left_target_1'] = np.array([-1., 0.])
        target_map['right_target_1'] = np.array([1., 0.])
        # target_map['left_target_2'] = np.array([-2., 0.])
        # target_map['right_target_2'] = np.array([2., 0.])
    return target_map


def setup(hyperparams):
    wall_dims = OpenRAVEBody.get_wall_dims('closet')
    config = {
        'obs_include': ['overhead_camera'],
        'include_files': [],
        'include_items': [
            {'name': 'pr2', 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.6, 1.), 'rgba': (1, 1, 1, 1)},
        ],
        'view': False,
        'image_dimensions': (hyperparams['image_width'], hyperparams['image_height'])
    }

    self.main_camera_id = 0
    colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0.5, 0.75, 0.25, 1], [0.75, 0.5, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]

    items = config['include_items']
    prim_options = get_prim_choices()
    for name in prim_options[OBJ_ENUM]:
        if name =='pr2': continue
        cur_color = colors.pop(0)
        items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.4, 1.), 'rgba': tuple(cur_color)})
        # items.append({'name': '{0}_end_target'.format(name), 'type': 'cylinder', 'is_fixed': False, 'pos': (10, 10, 0.5), 'dimensions': (0.8, 0.2), 'rgba': tuple(cur_color)})
    for i in range(len(wall_dims)):
        dim, next_trans = wall_dims[i]
        next_trans[0,3] -= 3.5
        next_dim = dim # [dim[1], dim[0], dim[2]]
        pos = next_trans[:3,3] # [next_trans[1,3], next_trans[0,3], next_trans[2,3]]
        items.append({'name': 'wall{0}'.format(i), 'type': 'box', 'is_fixed': True, 'pos': pos, 'dimensions': next_dim, 'rgba': (0.2, 0.2, 0.2, 1)})

    generate_xml(BASE_XML, ENV_XML, include_files=config['include_files'], include_items=config['include_items'])
