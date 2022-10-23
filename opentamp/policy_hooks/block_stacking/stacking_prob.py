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


domain_file = opentamp.__path__._path[0] + "/domains/robot_manipulation_domain/right_stack.domain"

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
TASK_MAPPING = {
    'move_to_grasp_right': [
        '0: MOVE_TO_GRASP_RIGHT PANDA {0} BLOCK0_TARGET',
    ],

   'lift_right': [
        '0: LIFT_RIGHT PANDA {0} BLOCK0_TARGET',
    ],

    'stack_right': [
        '0: STACK_RIGHT PANDA {1} {0}'
    ],

}

N_OBJS = 3
ROBOT_NAME = 'panda'

def prob_file(descr=None):
    return opentamp.__path__._path[0] + "/domains/robot_manipulation_domain/probs/robodesk_prob.prob"


def get_prim_choices(task_list=None):
    out = OrderedDict({})
    if task_list is None:
        out[utils.TASK_ENUM] = sorted(list(get_tasks(mapping_file).keys()))
    else:
        out[utils.TASK_ENUM] = sorted(list(task_list))

    out[utils.OBJ_ENUM] = ['block{}'.format(i) for i in range(N_OBJ)]
    out[utils.TARG_ENUM] = ['block{}'.format(i) for i in range(N_OBJ)]
    return out


def get_vector(config):
    state_vector_include = {
        ROBOT_NAME: ['right', 'right_ee_pos', 'right_ee_rot', 'right_gripper', 'pose', 'rotation']
    }

    for item in ['block{}'.format(i) for i in range(N_OBJS)]:
        state_vector_include[item] = ['pose', 'rotation']

    action_vector_include = {
        'panda': ['right', 'right_gripper']
    }

    target_vector_include = {}

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

