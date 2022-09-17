from datetime import datetime
import os
import os.path

import numpy as np

import opentamp
import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.core.util_classes.namo_grip_predicates import ATTRMAP
import opentamp.policy_hooks.robodesk.desk_prob as prob

from opentamp.policy_hooks.robodesk.robot_agent import RobotAgent
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
    #prob.GOAL_OPTIONS = [
    #            '(Stacked upright_block flat_block)',
    #            #'(InGripperRight panda green_button)',
    #            ]
    prob.NUM_OBJS = no
    prob.NUM_TARGS = nt
    prob.N_GRASPS = N_GRASPS
    prob.FIX_TARGETS = True
    opts = prob.get_prim_choices()
    discr_opts = [opt for opt in opts if not np.isscalar(opts[opt])]
    cont_opts = [opt for opt in opts if np.isscalar(opts[opt])]

    prob.n_aux = 0
    config = {
        'num_samples': N_SAMPLES,
        'num_conds': NUM_CONDS,
        'solver_type': 'adam', #'rmsprop',
        'base_weight_dir': 'panda_',
        'max_sample_queue': 5e2,
        'max_opt_sample_queue': 10,
        'task_map_file': prob.mapping_file,
        'prob': prob,
        'get_vector': prob.get_vector,
        'num_objs': no,
        'num_targs': nt,
        'attr_map': ATTRMAP,
        'agent_type': RobotAgent,
        'mp_solver_type': RobotSolverOSQP,
        'll_solver_type': RobotSolverOSQP,
        'domain': 'panda',
        'share_buffer': True,
        'split_nets': False,
        'robot_name': 'panda',
        'ctrl_mode': 'joint_angle',
        'visual_cameras': [0],

        'state_include': [utils.STATE_ENUM],
        'obs_include': [utils.TASK_ENUM,
                        #utils.END_POSE_ENUM,
                        #utils.END_ROT_ENUM,
                        utils.RIGHT_ENUM,
                        #utils.RIGHT_VEL_ENUM,
                        utils.RIGHT_EE_POS_ENUM,
                        utils.RIGHT_GRIPPER_ENUM,
                        utils.GRIP_CMD_ENUM,
                        utils.OBJ_ENUM,
                        utils.TARG_ENUM,
                        utils.DOOR_ENUM,
                        ],
        'prim_obs_include': [
                             utils.ONEHOT_GOAL_ENUM,
                             utils.RIGHT_EE_POS_ENUM,
                             #utils.RIGHT_EE_ROT_ENUM,
                             utils.RIGHT_ENUM,
                             #utils.RIGHT_VEL_ENUM,
                             utils.RIGHT_GRIPPER_ENUM,
                             utils.GRIP_CMD_ENUM,
                             ],
        'prim_out_include': discr_opts,
        'cont_obs_include': [opt for opt in discr_opts],
        'sensor_dims': {
                utils.OBJ_POSE_ENUM: 3,
                utils.TARG_POSE_ENUM: 3,
                utils.LIDAR_ENUM: N_DIRS,
                utils.EE_ENUM: 3,
                utils.RIGHT_EE_POS_ENUM: 3,
                utils.RIGHT_EE_ROT_ENUM: 3,
                utils.END_POSE_ENUM: 3,
                utils.ABS_POSE_ENUM: 3,
                utils.END_ROT_ENUM: 3,
                utils.TRUE_POSE_ENUM: 3,
                utils.TRUE_ROT_ENUM: 3,
                utils.GRIPPER_ENUM: 1,
                utils.GOAL_ENUM: 3*no,
                utils.ONEHOT_GOAL_ENUM: 12 + len(prob.GOAL_OPTIONS),
                utils.INGRASP_ENUM: no,
                utils.TRUETASK_ENUM: 2,
                utils.TRUEOBJ_ENUM: no,
                utils.ATGOAL_ENUM: no,
                utils.FACTOREDTASK_ENUM: len(list(prob.get_prim_choices().keys())),
                utils.RIGHT_ENUM: 7,
                utils.RIGHT_VEL_ENUM: 7,
                utils.RIGHT_GRIPPER_ENUM: 2,
                utils.GRIP_CMD_ENUM: 2,
                utils.QPOS_ENUM: 38,
                # utils.INIT_OBJ_POSE_ENUM: 2,
            },
        'time_limit': TIME_LIMIT,
        'num_filters': [32, 32, 16],
        'filter_sizes': [7, 5, 3],
        'prim_filters': [16,16,16], # [16, 32],
        'prim_filter_sizes': [7,5,5], # [7, 5],
        'cont_filters': [32, 16],
        'cont_filter_sizes': [7, 5],
    }

    for o in range(no):
        config['sensor_dims'][utils.OBJ_DELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.OBJ_ROTDELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.TARG_ROTDELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.OBJ_ENUMS[o]] = 3
        config['sensor_dims'][utils.TARG_ENUMS[o]] = 3
        #config['prim_obs_include'].append(utils.OBJ_DELTA_ENUMS[o])
        #config['prim_obs_include'].append(utils.TARG_ENUMS[o])
        #config['prim_obs_include'].append(utils.OBJ_ROTDELTA_ENUMS[o])
        # config['prim_obs_include'].append(utils.TARG_ROTDELTA_ENUMS[o])
    return config

config = refresh_config()

