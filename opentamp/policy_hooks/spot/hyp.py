from datetime import datetime
import os
import os.path

import numpy as np

import opentamp
import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.core.util_classes.namo_grip_predicates import ATTRMAP
import opentamp.policy_hooks.spot.spot_prob as prob

from opentamp.policy_hooks.spot.spot_agent import SpotAgent
from opentamp.pma.robot_solver import RobotSolverOSQP

BASE_DIR = opentamp.__path__._path[0] + '/policy_hooks/'
EXP_DIR = BASE_DIR + 'experiments/'

NUM_CONDS = 1 # Per rollout server
NUM_PRETRAIN_STEPS = 20
NUM_PRETRAIN_TRAJ_OPT_STEPS = 1
NUM_TRAJ_OPT_STEPS = 1
N_SAMPLES = 10
N_TRAJ_CENTERS = 1
HL_TIMEOUT = 600
OPT_WT_MULT = 5e2
N_ROLLOUT_SERVERS = 34 # 58
N_ALG_SERVERS = 0
N_OPTIMIZERS = 0
N_DIRS = 16
N_GRASPS = 4
TIME_LIMIT = 14400

def refresh_config(no=1, nt=1):
    prob.NUM_OBJS = no
    prob.NUM_TARGS = nt
    opts = prob.get_prim_choices()
    discr_opts = [opt for opt in opts if not np.isscalar(opts[opt])]
    cont_opts = [opt for opt in opts if np.isscalar(opts[opt])]

    prob.n_aux = 0
    config = {
        'num_samples': N_SAMPLES,
        'num_conds': NUM_CONDS,
        'solver_type': 'adam', #'rmsprop',
        'base_weight_dir': 'spot_',
        'task_map_file': prob.mapping_file,
        'prob': prob,
        'get_vector': prob.get_vector,
        'num_objs': no,
        'num_targs': nt,
        'attr_map': ATTRMAP,
        'agent_type': SpotAgent,
        'mp_solver_type': RobotSolverOSQP,
        'll_solver_type': RobotSolverOSQP,
        'domain': 'spot',
        'share_buffer': True,
        'split_nets': False,
        'robot_name': 'spot',
        'ctrl_mode': 'joint_angle',
        'visual_cameras': [0],

        'state_include': [utils.STATE_ENUM],
        'obs_include': [utils.TASK_ENUM,
                        utils.POS_DELTA_ENUM,
                        utils.THETA_DELTA_ENUM,
                        utils.IMAGE_ENUM,
                        ],
        'prim_obs_include': [
                             utils.ONEHOT_GOAL_ENUM,
                             utils.POS_ENUM,
                             utils.THETA_ENUM,
                             utils.IMAGE_ENUM,
                             ],
        'prim_out_include': discr_opts,
        'sensor_dims': {
                utils.TASK_ENUM: 1,
                utils.POSE_DELTA: 2,
                utils.THETA_DELTA_ENUM: 1,
                utils.IMAGE_ENUM: IM_W * IM_H * IM_C,
            },
        'num_filters': [32, 32, 16],
        'filter_sizes': [7, 5, 3],
        'prim_filters': [16,16,16], # [16, 32],
        'prim_filter_sizes': [7,5,5], # [7, 5],
    }

    return config

config = refresh_config()

