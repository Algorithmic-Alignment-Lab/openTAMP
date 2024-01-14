from collections import OrderedDict
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.policy_hooks.utils.policy_solver_utils import *
from opentamp.policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str

# random constants, todo deprecate
mapping_file = "policy_hooks/namo/pointer_taskmapping"  # TODO alter

END_TARGETS = [(1)]
n_aux=0
NUM_OBJS = 1


## DUMMY WRAPPER FOR NOW

def get_prim_choices(task_list=None):
    out = OrderedDict({})
    # if task_list is None:
    #     out[utils.TASK_ENUM] = sorted(list(get_tasks(mapping_file).keys()))
    # else:
    #     out[utils.TASK_ENUM] = sorted(list(task_list))
    # out[utils.OBJ_ENUM] = ['can{0}'.format(i) for i in range(NUM_OBJS)]
    # out[utils.TARG_ENUM] = []
    # for i in range(n_aux):
    #     out[utils.TARG_ENUM] += ['aux_target_{0}'.format(i)]
    # for i in range(len(END_TARGETS)):
    #     out[utils.TARG_ENUM] += ['end_target_{0}'.format(i)]
    #out[utils.GRASP_ENUM] = ['grasp{0}'.format(i) for i in range(N_GRASPS)]
    #out[utils.ABS_POSE_ENUM] = 2
    out[utils.TASK_ENUM] = sorted(list(get_tasks(mapping_file).keys()))
    # out[utils.ANG_ENUM] = 1  ## TODO: HARDCODED FOR NOW, change
    return out


def get_vector(config):
    # construct state_include, action_include, target_include vectors
    concr_gym_env = config['gym_env_type']()
    return concr_gym_env.get_vector()


def get_random_initial_state_vec(config, plans, dX, state_inds, conditions):
    # empty targ maps, just pass the observation itself
    concr_gym_env = config['gym_env_type']()
    return [concr_gym_env.get_random_init_state()], [{'target': END_TARGETS[0]}]


def get_plans(use_tf=False, params=None):
    # do vacuously for now

    openrave_bodies = {}

    # TODO FIX THE OPENRAVE WEIRDNESS
    if params:
        for param in list(params.values()):
            if hasattr(param, 'geom'):
                if not hasattr(param, 'openrave_body') or param.openrave_body is None:
                    param.openrave_body = OpenRAVEBody(0, param.name, param.geom)
                openrave_bodies[param.name] = param.openrave_body

    return {}, openrave_bodies, None
