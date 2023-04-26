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
from opentamp.pma.hl_solver import FFSolver
from opentamp.policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str
from opentamp.policy_hooks.utils.policy_solver_utils import *
import opentamp.policy_hooks.utils.policy_solver_utils as utils


domain_file = opentamp.__path__._path[0] + "/domains/robot_manipulation_domain/right_desk.domain"
mapping_file =  opentamp.__path__._path[0] + "/policy_hooks/robodesk/robot_task_mapping"

GOAL_OPTIONS = [
                '(SlideDoorClose shelf_handle shelf)',
                '(SlideDoorOpen drawer_handle drawer)',
                '(Lifted upright_block panda)',
                '(Lifted ball panda)',
                '(Near upright_block off_desk_target)',
                '(InSlideDoor flat_block shelf)',
                '(Near flat_block bin_target)',
                '(Stacked upright_block flat_block)',
                '(InGripperRight panda green_button)',
                ]
INVARIANT_GOALS = []

def prob_file(descr=None):
    return opentamp.__path__._path[0] + "/domains/robot_manipulation_domain/probs/robodesk_prob.prob"


def get_prim_choices(task_list=None):
    out = OrderedDict({})
    if task_list is None:
        out[utils.TASK_ENUM] = sorted(list(get_tasks(mapping_file).keys()))
    else:
        out[utils.TASK_ENUM] = sorted(list(task_list))

    out[utils.OBJ_ENUM] = ['ball', 'upright_block', 'flat_block', 'green_button']
    out[utils.TARG_ENUM] = ['bin_target', 'off_desk_target']
    out[utils.DOOR_ENUM] = ['drawer', 'shelf']
    for door in ['drawer', 'shelf']:
        out[utils.OBJ_ENUM].append('{}_handle'.format(door))
    #out[utils.END_POSE_ENUM] = 3
    #out[utils.END_ROT_ENUM] = 3
    return out


def get_vector(config):
    state_vector_include = {
        'panda': ['right', 'right_ee_pos', 'right_ee_rot', 'right_gripper', 'pose', 'rotation']
    }

    for item in ['ball', 'upright_block', 'flat_block', 'green_button', 'red_button', 'blue_button']:
        state_vector_include[item] = ['pose', 'rotation']

    for door in ['drawer', 'shelf']:
        state_vector_include[door] = ['pose', 'rotation', 'hinge']
        state_vector_include['{}_handle'.format(door)] = ['pose', 'rotation']

    action_vector_include = {
        'panda': ['right', 'right_gripper']
    }

    target_vector_include = {
        'bin_target': ['value', 'rotation'],
        'off_desk_target': ['value', 'rotation'],
    }
    for goal in GOAL_OPTIONS + INVARIANT_GOALS:
        target_vector_include[goal] = ['value']

    return state_vector_include, action_vector_include, target_vector_include


def get_plans(use_tf=False):
    tasks = get_tasks(mapping_file)
    task_ids = sorted(list(get_tasks(mapping_file).keys()))
    prim_options = get_prim_choices()
    plans = {}
    openrave_bodies = {}
    env = None
    params = None
    sess = None

    for task_ind, task in enumerate(task_ids):
        params = None
        # SlideDoor actions don't need obj or targ
        if task.lower().find('slide') >= 0:
            for door_ind, door in enumerate(prim_options[DOOR_ENUM]):
                next_task_str = copy.deepcopy(tasks[task])
                new_task_str = []
                for step in next_task_str:
                    new_task_str.append(step.format('', '', door))
                plan = plan_from_str(new_task_str, prob_file(), domain_file, env, openrave_bodies, params=params, sess=sess, use_tf=use_tf)
                params = plan.params
                if env is None:
                    env = plan.env
                    for param in list(plan.params.values()):
                        if hasattr(param, 'geom'):
                            if not hasattr(param, 'openrave_body') or param.openrave_body is None:
                                param.openrave_body = OpenRAVEBody(env, param.name, param.geom)
                            openrave_bodies[param.name] = param.openrave_body

                for obj_ind, obj in enumerate(prim_options[OBJ_ENUM]):
                    for targ_ind, targ in enumerate(prim_options[TARG_ENUM]):
                        plans[task_ind, obj_ind, targ_ind, door_ind] = plan

        # Lift/Hold/MoveToGrasp only need obj
        elif task.lower().find('hold') >= 0 or \
             task.lower().find('lift') >= 0 or \
             task.lower().find('grasp') >= 0:
            for obj_ind, obj in enumerate(prim_options[OBJ_ENUM]):
                next_task_str = copy.deepcopy(tasks[task])
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

                for door_ind, door in enumerate(prim_options[DOOR_ENUM]):
                    for targ_ind, targ in enumerate(prim_options[TARG_ENUM]):
                        plans[task_ind, obj_ind, targ_ind, door_ind] = plan

        # PlaceInDoor actions don't need targ
        elif task.lower().find('place_in') >= 0:
            for obj_ind, obj in enumerate(prim_options[OBJ_ENUM]):
                for door_ind, door in enumerate(prim_options[DOOR_ENUM]):
                    next_task_str = copy.deepcopy(tasks[task])
                    new_task_str = []
                    for step in next_task_str:
                        new_task_str.append(step.format(obj, '', door))
                    plan = plan_from_str(new_task_str, prob_file(), domain_file, env, openrave_bodies, params=params, sess=sess, use_tf=use_tf)
                    params = plan.params
                    if env is None:
                        env = plan.env
                        for param in list(plan.params.values()):
                            if hasattr(param, 'geom'):
                                if not hasattr(param, 'openrave_body') or param.openrave_body is None:
                                    param.openrave_body = OpenRAVEBody(env, param.name, param.geom)
                                openrave_bodies[param.name] = param.openrave_body

                    for targ_ind, targ in enumerate(prim_options[TARG_ENUM]):
                        plans[task_ind, obj_ind, targ_ind, door_ind] = plan

        else:
            for obj_ind, obj in enumerate(prim_options[OBJ_ENUM]):
                for targ_ind, targ in enumerate(prim_options[TARG_ENUM]):
                    next_task_str = copy.deepcopy(tasks[task])
                    new_task_str = []
                    for step in next_task_str:
                        new_task_str.append(step.format(obj, targ, ''))
                    plan = plan_from_str(new_task_str, prob_file(), domain_file, env, openrave_bodies, params=params, sess=sess, use_tf=use_tf)
                    for door_ind, door in enumerate(prim_options[DOOR_ENUM]):
                        plans[task_ind, obj_ind, targ_ind, door_ind] = plan

                    params = plan.params
                    if env is None:
                        env = plan.env
                        for param in list(plan.params.values()):
                            if hasattr(param, 'geom'):
                                if not hasattr(param, 'openrave_body') or param.openrave_body is None:
                                    param.openrave_body = OpenRAVEBody(env, param.name, param.geom)
                                openrave_bodies[param.name] = param.openrave_body
    return plans, openrave_bodies, env


def get_random_initial_state_vec(config, plans, dX, state_inds, conditions):
    return [np.zeros(dX)], [{'bin_target': np.zeros(3)}]

