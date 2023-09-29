

NUM_OBJS = 1
NUM_TARGS = 1

from datetime import datetime
import os
import os.path

import numpy as np

# from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd

# from opentamp.policy_hooks.algorithm_impgps import AlgorithmIMPGPS
# from opentamp.policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
import opentamp.policy_hooks.utils.policy_solver_utils as utils
# from opentamp.policy_hooks.traj_opt_pi2 import TrajOptPI2
from opentamp.core.util_classes.namo_grip_predicates import ATTRMAP
import opentamp.policy_hooks.robosuite.jnt_pick_prob as prob
prob.NUM_OBJS = NUM_OBJS
prob.NUM_TARGS = NUM_TARGS
# from opentamp.policy_hooks.policy_mp_prior_gmm import PolicyMPPriorGMM
# from opentamp.policy_hooks.policy_prior_gmm import PolicyPriorGMM

from opentamp.policy_hooks.robosuite.robot_agent import RobotAgent
from opentamp.pma.robosuite_solver import RobotSolverOSQP

BASE_DIR = os.getcwd() + '/policy_hooks/'
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


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': NUM_CONDS,
}

# algorithm = {
#     # 'type': AlgorithmIMPGPS,
#     'conditions': common['conditions'],
#     'policy_sample_mode': 'add',
#     'sample_on_policy': True,
#     'iterations': 1e3, #5e4,
#     'max_ent_traj': 0.0,
#     'fit_dynamics': False,
#     'stochastic_conditions': True,
#     'policy_inf_coeff': 1e2,
#     'policy_out_coeff': 1e1,
#     'kl_step': 1.,
#     'min_step_mult': 0.05,
#     'max_step_mult': 5.0,
#     'sample_ts_prob': 1.0,
#     'opt_wt': OPT_WT_MULT,
#     'fail_value': 50,
#     'use_centroids': True,
#     'n_traj_centers': N_TRAJ_CENTERS,
#     'num_samples': N_SAMPLES,
#     'mp_opt': True,
#     'her': False,
#     'rollout_opt': False,
# }

# algorithm['init_traj_distr'] = {
#     # 'type': init_pd,
#     'init_var': 0.01,
#     'pos_gains': 0.00,
# }

# algorithm['traj_opt'] = {
#     # 'type': TrajOptPI2,
#     'kl_threshold': 1.,
#     'covariance_damping': 0.00,
#     'min_temperature': 1e-3,
# }

# # algorithm['policy_prior'] = {
# #     'type': PolicyPrior,
# # }

# # algorithm = {
# #     'type': AlgorithmMDGPS,
# #     'conditions': common['conditions'],
# #     'iterations': 10,
# #     'kl_step': 0.1,
# #     'min_step_mult': 0.5,
# #     'max_step_mult': 3.0,
# #     'policy_sample_mode': 'replace',
# # }

# # algorithm['init_traj_distr'] = {
# #     'type': init_pd,
# #     'pos_gains':  1e-5,
# # }

# # algorithm['init_traj_distr'] = {
# #     'type': init_lqr,
# #     'init_var': 0.001,
# #     'stiffness': 10.0,
# #     'stiffness_vel': 0.5,
# #     'final_weight': 5.0,
# # }

# # algorithm = {
# #     'type': AlgorithmTrajOptPILQR,
# #     'conditions': common['conditions'],
# #     'iterations': 20,
# #     'step_rule': 'res_percent',
# #     'step_rule_res_ratio_dec': 0.2,
# #     'step_rule_res_ratio_inc': 0.05,
# #     'kl_step': np.linspace(0.6, 0.2, 100),
# # }

# # algorithm['dynamics'] = {
# #     'type': DynamicsLRPrior,
# #     'regularization': 1e-6,
# #     'prior': {
# #         'type': DynamicsPriorGMM,
# #         'max_clusters': 20,
# #         'min_samples_per_cluster': 60,
# #         'max_samples': 30,
# #     },
# # }

# # algorithm['traj_opt'] = {
# #     'type': TrajOptPILQR,
# # }

# # algorithm['traj_opt'] = {
# #     'type': TrajOptLQRPython,
# # }

# algorithm['policy_prior'] = {
#     # 'type': PolicyPriorGMM,
#     'max_clusters': 20,
#     'min_samples_per_cluster': 40,
#     'max_samples': 50,
# }

# algorithm['mp_policy_prior'] = {
#     # 'type': PolicyMPPriorGMM,
#     'max_clusters': 20,
#     'min_samples_per_cluster': 40,
#     'max_samples': 50,
# }

def refresh_config(no=NUM_OBJS, nt=NUM_TARGS):
    cost_wp_mult = np.ones((3 + 2 * NUM_OBJS))
    prob.NUM_OBJS = no
    prob.NUM_TARGS = nt
    prob.N_GRASPS = N_GRASPS
    prob.FIX_TARGETS = True

    prob.END_TARGETS = prob.END_TARGETS[:8]
    prob.n_aux = 0
    config = {
        'gui_on': False,
        # 'iterations': algorithm['iterations'],
        'verbose_trials': 1,
        'verbose_policy_trials': 1,
        'common': common,
        # 'algorithm': algorithm,
        # 'num_samples': algorithm['num_samples'],
        'num_distilled_samples': 0,
        'num_conds': NUM_CONDS,
        'mode': 'position',
        # 'stochastic_conditions': algorithm['stochastic_conditions'],
        'policy_coeff': 1e0,
        'sample_on_policy': True,
        'hist_len': 3,
        'take_optimal_sample': True,
        'num_rollouts': 10,
        'max_tree_depth': 5 + no*2,
        'branching_factor': 4,
        # 'opt_wt': algorithm['opt_wt'],
        # 'fail_value': algorithm['fail_value'],
        'lr': 1e-3,
        'solver_type': 'adam', #'rmsprop',
        'cost_wp_mult': cost_wp_mult,

        'train_iterations': 50,
        'weight_decay': 1e-3,
        'prim_weight_decay': 1e-3,
        'val_weight_decay': 1e-3,
        'batch_size': 500,
        'n_layers': 2,
        'prim_n_layers': 1,
        'val_n_layers': 1,
        'dim_hidden': [32, 32],
        'prim_dim_hidden': [32],
        'val_dim_hidden': [32],
        # 'n_traj_centers': algorithm['n_traj_centers'],
        'traj_opt_steps': NUM_TRAJ_OPT_STEPS,
        'pretrain_steps': NUM_PRETRAIN_STEPS,
        'pretrain_traj_opt_steps': NUM_PRETRAIN_TRAJ_OPT_STEPS,
        'on_policy': True,

        # New for multiprocess, transfer to sequential version as well.

        'n_optimizers': N_OPTIMIZERS,
        'n_rollout_servers': N_ROLLOUT_SERVERS,
        'n_alg_servers': N_ALG_SERVERS,
        'base_weight_dir': 'sawyer_',
        # 'policy_out_coeff': algorithm['policy_out_coeff'],
        # 'policy_inf_coeff': algorithm['policy_inf_coeff'],
        'max_sample_queue': 5e2,
        'max_opt_sample_queue': 10,
        'task_map_file': prob.mapping_file,
        'prob': prob,
        'get_vector': prob.get_vector,
        'obj_type': 'can',
        'num_objs': no,
        'num_targs': nt,
        'attr_map': ATTRMAP,
        'agent_type': RobotAgent,
        'mp_solver_type': RobotSolverOSQP,
        'll_solver_type': RobotSolverOSQP,
        'update_size': 2000,
        'prim_update_size': 5000,
        'val_update_size': 1000,
        'use_local': True,
        'n_dirs': N_DIRS,
        'domain': 'sawyer',
        'perturb_steps': 3,
        'mcts_early_stop_prob': 0.5,
        'hl_timeout': HL_TIMEOUT,
        'multi_policy': False,
        'image_width': 107,
        'image_height': 80,
        'image_channels': 3,
        'opt_prob': 1.,
        'opt_smooth': False,
        'share_buffer': True,
        'split_nets': False,
        'split_mcts_alg': True,
        'robot_name': 'sawyer',
        'ctrl_mode': 'joint_angle',
        'visual_cameras': [0],

        'state_include': [utils.STATE_ENUM],
        'obs_include': [utils.TASK_ENUM,
                        #utils.OBJ_POSE_ENUM,
                        #utils.TARG_POSE_ENUM,
                        utils.END_POSE_ENUM,
                        utils.END_ROT_ENUM,
                        utils.RIGHT_ENUM,
                        #utils.RIGHT_GRIPPER_ENUM,
                        # utils.DONE_ENUM,
                        ],
        'prim_obs_include': [
                            #  utils.DONE_ENUM,
                            #  utils.STATE_ENUM,
                            #  utils.GOAL_ENUM,
                             utils.ONEHOT_GOAL_ENUM,
                             ],
        'val_obs_include': [utils.ONEHOT_GOAL_ENUM,
                            ],
        'prim_out_include': list(prob.get_prim_choices().keys()),
        'sensor_dims': {
                utils.OBJ_POSE_ENUM: 3,
                utils.TARG_POSE_ENUM: 3,
                utils.LIDAR_ENUM: N_DIRS,
                utils.EE_ENUM: 3,
                utils.RIGHT_EE_POS_ENUM: 3,
                utils.RIGHT_EE_ROT_ENUM: 3,
                utils.END_POSE_ENUM: 3,
                utils.END_ROT_ENUM: 3,
                utils.GRIPPER_ENUM: 1,
                utils.GOAL_ENUM: 3*no,
                utils.ONEHOT_GOAL_ENUM: no*(prob.n_aux + len(prob.END_TARGETS)),
                utils.INGRASP_ENUM: no,
                utils.TRUETASK_ENUM: 2,
                utils.TRUEOBJ_ENUM: no,
                utils.TRUETARG_ENUM: len(prob.END_TARGETS),
                utils.ATGOAL_ENUM: no,
                utils.FACTOREDTASK_ENUM: len(list(prob.get_prim_choices().keys())),
                utils.RIGHT_ENUM: 7,
                utils.RIGHT_GRIPPER_ENUM: 2,
                # utils.INIT_OBJ_POSE_ENUM: 2,
            },
        'visual': False,
        'time_limit': TIME_LIMIT,
        'success_to_replace': 1,
        'steps_to_replace': no * 50,
        'curric_thresh': -1,
        'n_thresh': -1,
        'expand_process': False,
        'her': False,
        'prim_decay': 0.95,
        'prim_first_wt': 1e1,
    }

    for o in range(no):
        config['sensor_dims'][utils.OBJ_DELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.OBJ_ROTDELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.TARG_ROTDELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.OBJ_ENUMS[o]] = 3
        config['sensor_dims'][utils.TARG_ENUMS[o]] = 3
        config['prim_obs_include'].append(utils.OBJ_DELTA_ENUMS[o])
        config['prim_obs_include'].append(utils.TARG_ENUMS[o])
        config['prim_obs_include'].append(utils.OBJ_ROTDELTA_ENUMS[o])
        config['prim_obs_include'].append(utils.TARG_ROTDELTA_ENUMS[o])
    return config

config = refresh_config()
