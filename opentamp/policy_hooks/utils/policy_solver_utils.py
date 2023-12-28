import numpy as np
import scipy.stats as stats

# Don't change these
ACTION_ENUM = 0
STATE_ENUM = 1
IM_ENUM = 2
NOISE_ENUM = 3
EE_ENUM = 4
GRIPPER_ENUM = 5
TRAJ_HIST_ENUM = 6
COLORS_ENUM = 7
TASK_ENUM = 8
TARGETS_ENUM = 9
OBJ_ENUM = 10
TARG_ENUM = 11
OBJ_POSE_ENUM = 12
TARG_POSE_ENUM = 13
LIDAR_ENUM = 14
EE_2D_ENUM = 15
EE_POS_ENUM = 16
EE_QUAT_ENUM = 17
RIGHT_EE_POS_ENUM = 18
RIGHT_EE_QUAT_ENUM = 19
LEFT_EE_POS_ENUM = 20
LEFT_EE_QUAT_ENUM = 21
CLOTH_POINTS_ENUM = 22
LEFT_TARG_POSE_ENUM  = 23
RIGHT_TARG_POSE_ENUM = 24
LEFT_TARG_ENUM = 25
RIGHT_TARG_ENUM = 26
CLOTH_JOINTS_ENUM = 27
OVERHEAD_IMAGE_ENUM = 28
LEFT_IMAGE_ENUM = 29
RIGHT_IMAGE_ENUM = 30
JOINTS_ENUM = 31
POS_ENUM = 32
LEFT_GRIPPER_ENUM = 33
RIGHT_GRIPPER_ENUM = 34
INIT_OBJ_POSE_ENUM = 35
END_POSE_ENUM = 36
GOAL_ENUM = 37
GRASP_ENUM = 38
ONEHOT_GOAL_ENUM = 39
ONEHOT_TASK_ENUM = 40
ATGOAL_ENUM = 41
INGRASP_ENUM = 42
TRUETASK_ENUM = 43
TRUEOBJ_ENUM = 44
TRUETARG_ENUM = 45
TRUEGRASP_ENUM = 46
FACTOREDTASK_ENUM = 47
DONE_ENUM = 48
STATE_DELTA_ENUM = 49
MJC_SENSOR_ENUM = 50
STATE_HIST_ENUM = 51
VEL_ENUM = 52
THETA_ENUM = 53
THETA_VEC_ENUM = 54
TASK_DONE_ENUM = 55
TASK_HIST_ENUM = 56
OBJ_LIDAR_ENUM = 57
ARM_ENUM = 58
LEFT_ENUM = 59
RIGHT_ENUM = 60
ABS_POSE_ENUM = 61
REL_POSE_ENUM = 62
TRUE_POSE_ENUM = 63
DOOR_ENUM = 64
DOOR_THETA_ENUM = 65
RIGHT_EE_ROT_ENUM = 66
RIGHT_EE_QUAT_ENUM = 67
END_ROT_ENUM = 68
OBJ_ROT_ENUM = 69
TARG_ROT_ENUM = 70
TRUE_ROT_ENUM = 71
GRIP_CMD_ENUM = 72
RIGHT_VEL_ENUM = 73
LEFT_VEL_ENUM = 74
QPOS_ENUM = 75
BELIEF_GLOBAL_ENUM = 76
PAST_COUNT_ENUM = 77
PAST_TARG_ENUM = 78
PAST_POINT_ENUM = 79
PAST_TASK_ENUM = 80

OBJ_ENUMS = {}
for i in range(100):
    OBJ_ENUMS[i] = 100 + i

TARG_ENUMS = {}
for i in range(50):
    TARG_ENUMS[i] = 200 + i

TARG_DELTA_ENUMS = {}
for i in range(50):
    TARG_DELTA_ENUMS[i] = 250 + i

GOAL_ENUMS = {}
for i in range(100):
    GOAL_ENUMS[i] = 300 + i

OBJ_DELTA_ENUMS = {}
for i in range(50):
    OBJ_DELTA_ENUMS[i] = 400 + i

OBJ_ROTDELTA_ENUMS = {}
for i in range(50, 100):
    OBJ_ROTDELTA_ENUMS[i-50] = 400 + i

TARG_ROTDELTA_ENUMS = {}
for i in range(50):
    TARG_ROTDELTA_ENUMS[i] = 500 + i

IM_H = 64
IM_W = 64
IM_C = 3

# FAIL_LABEL = (1, 0)
# SUCCESS_LABEL = (0, 1)
FAIL_LABEL = 0.
SUCCESS_LABEL = 1.

GPS_RATIO = 1e3

MUJOCO_STEPS_PER_SECOND = 200
POLICY_STEPS_PER_SECOND = 1

# Use these to help delineate modes
POSE_DELTA = 0
JOINT_DELTA = 1
POSE = 2
VEL = 3
ACC = 4
JOINT = 5

## Misc Constants
EXPAND_LIMIT = 30


def get_state_action_inds(plan, robot_name, attr_map, x_params={}, u_params={}):
    '''
    Maps the parameters of the plan actions to indices in the policy state and action vectors, and returns the dimensions of those vectors.
    This mapping should apply to any plan with the given actions. Replaces get_plan_to_policy_mapping.
    '''
    # assert all(map(lambda a: a.train_policy, plan.actions))
    if not len(plan.actions):
        return 0, {}, 0, {}

    params_to_x_inds, params_to_u_inds = {}, {}
    cur_x_ind, cur_u_ind = 0, 0

    # robot_x_attrs = x_params[robot_name]
    # robot_u_attrs = u_params[robot_name]
    # robot_attr_map = attr_map['Robot']
    # ee_all_attrs = ['ee_pose']
    # ee_pos_attrs = ['ee_left_pos', 'ee_right_pos']
    # ee_rot_attrs = ['ee_left_rot', 'ee_right_rot']
    '''
    for attr in robot_x_attrs:
        # if attr in ee_pos_attrs:
        #     x_inds = np.array([0, 1, 2]) + cur_x_ind
        #     cur_x_ind = x_inds[-1] + 1
        #     params_to_x_inds[(robot_name, attr)] = x_inds
        #     continue

        # if attr in ee_rot_attrs:
        #     x_inds = np.array([0, 1, 2, 3]) + cur_x_ind
        #     cur_x_ind = x_inds[-1] + 1
        #     params_to_x_inds[(robot_name, attr)] = x_inds
        #     continue

        # if attr in ee_all_attrs:
        #     x_inds = np.array(range(7)) + cur_x_ind
        #     cur_x_ind = x_inds[-1] + 1
        #     params_to_x_inds[(robot_name, attr)] = x_inds
        #     continue
        print(robot_attr_map, attr)
        inds = next(filter(lambda p: p[0]==attr, robot_attr_map))[1]
        x_inds = inds + cur_x_ind
        cur_x_ind = x_inds[-1] + 1
        params_to_x_inds[(robot_name, attr)] = x_inds

    for attr in robot_u_attrs:
        # if attr in ee_pos_attrs:
        #     u_inds = np.array([0, 1, 2]) + cur_u_ind
        #     cur_u_ind = u_inds[-1] + 1
        #     params_to_u_inds[(robot_name, attr)] = u_inds
        #     continue

        # if attr in ee_rot_attrs:
        #     u_inds = np.array([0, 1, 2, 3]) + cur_u_ind
        #     cur_u_ind = u_inds[-1] + 1
        #     params_to_u_inds[(robot_name, attr)] = u_inds
        #     continue

        # if attr in ee_all_attrs:
        #     u_inds = np.array(range(7)) + cur_u_ind
        #     cur_u_ind = u_inds[-1] + 1
        #     params_to_u_inds[(robot_name, attr)] = u_inds
        #     continue

        inds = next(filter(lambda p: p[0]==attr, robot_attr_map))[1]
        u_inds = inds + cur_u_ind
        cur_u_ind = u_inds[-1] + 1
        params_to_u_inds[(robot_name, attr)] = u_inds
    '''
    for param_name in x_params:
        # if param_name not in plan.params: continue
        param = plan.params[param_name]
        #param_attr_map = attr_map[param._type]
        for attr in x_params[param_name]:
            if (param_name, attr) in params_to_x_inds: continue
            inds = np.arange(len(getattr(param, attr)[:,0])) + cur_x_ind
            cur_x_ind = inds[-1] + 1
            params_to_x_inds[(param.name, attr)] = inds

    for param_name in u_params:
        # if param_name not in plan.params: continue
        param = plan.params[param_name]
        #param_attr_map = attr_map[param._type]
        for attr in u_params[param_name]:
            if (param_name, attr) in params_to_u_inds: continue
            inds = np.arange(len(getattr(param, attr)[:,0])) + cur_u_ind
            cur_u_ind = inds[-1] + 1
            params_to_u_inds[(param.name, attr)] = inds

    symbolic_boundary = cur_x_ind # Used to differentiate parameters from symbols in the state vector

    '''
    for param in plan.params.values():
        if not param.is_symbol(): continue
        param_attr_map = attr_map[param._type]
        for attr in param_attr_map:
            if (param.name, attr[0]) in params_to_x_inds: continue
            inds = attr[1] + cur_x_ind
            cur_x_ind = inds[-1] + 1
            params_to_x_inds[(param.name, attr[0])] = inds
    '''

    # dX, state index map, dU, (policy) action map
    return cur_x_ind, params_to_x_inds, cur_u_ind, params_to_u_inds, symbolic_boundary

def get_state_action_inds_from_params(params, robot_name, attr_map, x_params={}, u_params={}):
    '''
    Maps the parameters of the plan actions to indices in the policy state and action vectors, and returns the dimensions of those vectors.
    This mapping should apply to any plan with the given actions. Replaces get_plan_to_policy_mapping.
    '''
    # assert all(map(lambda a: a.train_policy, plan.actions))
    if not len(params):
        return 0, {}, 0, {}

    params_to_x_inds, params_to_u_inds = {}, {}
    cur_x_ind, cur_u_ind = 0, 0

    # robot_x_attrs = x_params[robot_name]
    # robot_u_attrs = u_params[robot_name]
    # robot_attr_map = attr_map['Robot']
    # ee_all_attrs = ['ee_pose']
    # ee_pos_attrs = ['ee_left_pos', 'ee_right_pos']
    # ee_rot_attrs = ['ee_left_rot', 'ee_right_rot']
    '''
    for attr in robot_x_attrs:
        # if attr in ee_pos_attrs:
        #     x_inds = np.array([0, 1, 2]) + cur_x_ind
        #     cur_x_ind = x_inds[-1] + 1
        #     params_to_x_inds[(robot_name, attr)] = x_inds
        #     continue

        # if attr in ee_rot_attrs:
        #     x_inds = np.array([0, 1, 2, 3]) + cur_x_ind
        #     cur_x_ind = x_inds[-1] + 1
        #     params_to_x_inds[(robot_name, attr)] = x_inds
        #     continue

        # if attr in ee_all_attrs:
        #     x_inds = np.array(range(7)) + cur_x_ind
        #     cur_x_ind = x_inds[-1] + 1
        #     params_to_x_inds[(robot_name, attr)] = x_inds
        #     continue
        print(robot_attr_map, attr)
        inds = next(filter(lambda p: p[0]==attr, robot_attr_map))[1]
        x_inds = inds + cur_x_ind
        cur_x_ind = x_inds[-1] + 1
        params_to_x_inds[(robot_name, attr)] = x_inds

    for attr in robot_u_attrs:
        # if attr in ee_pos_attrs:
        #     u_inds = np.array([0, 1, 2]) + cur_u_ind
        #     cur_u_ind = u_inds[-1] + 1
        #     params_to_u_inds[(robot_name, attr)] = u_inds
        #     continue

        # if attr in ee_rot_attrs:
        #     u_inds = np.array([0, 1, 2, 3]) + cur_u_ind
        #     cur_u_ind = u_inds[-1] + 1
        #     params_to_u_inds[(robot_name, attr)] = u_inds
        #     continue

        # if attr in ee_all_attrs:
        #     u_inds = np.array(range(7)) + cur_u_ind
        #     cur_u_ind = u_inds[-1] + 1
        #     params_to_u_inds[(robot_name, attr)] = u_inds
        #     continue

        inds = next(filter(lambda p: p[0]==attr, robot_attr_map))[1]
        u_inds = inds + cur_u_ind
        cur_u_ind = u_inds[-1] + 1
        params_to_u_inds[(robot_name, attr)] = u_inds
    '''
    for param_name in x_params:
        # if param_name not in plan.params: continue
        param = params[param_name]
        #param_attr_map = attr_map[param._type]
        for attr in x_params[param_name]:
            if (param_name, attr) in params_to_x_inds: continue
            inds = np.arange(len(getattr(param, attr)[:,0])) + cur_x_ind
            cur_x_ind = inds[-1] + 1
            params_to_x_inds[(param.name, attr)] = inds

    for param_name in u_params:
        # if param_name not in plan.params: continue
        param = params[param_name]
        #param_attr_map = attr_map[param._type]
        for attr in u_params[param_name]:
            if (param_name, attr) in params_to_u_inds: continue
            inds = np.arange(len(getattr(param, attr)[:,0])) + cur_u_ind
            cur_u_ind = inds[-1] + 1
            params_to_u_inds[(param.name, attr)] = inds

    symbolic_boundary = cur_x_ind # Used to differentiate parameters from symbols in the state vector

    '''
    for param in plan.params.values():
        if not param.is_symbol(): continue
        param_attr_map = attr_map[param._type]
        for attr in param_attr_map:
            if (param.name, attr[0]) in params_to_x_inds: continue
            inds = attr[1] + cur_x_ind
            cur_x_ind = inds[-1] + 1
            params_to_x_inds[(param.name, attr[0])] = inds
    '''

    # dX, state index map, dU, (policy) action map
    return cur_x_ind, params_to_x_inds, cur_u_ind, params_to_u_inds, symbolic_boundary

def get_target_inds(plan, attr_map, include):
    cur_ind = 0
    target_inds = {}
    for param_name in include:
        if param_name in plan.params:
            param = plan.params[param_name]
            for attr in include[param_name]:
                inds = np.arange(len(getattr(param, attr)[:,0])) + cur_ind
                cur_ind = inds[-1] + 1
                target_inds[param.name, attr] = inds
        else:
            dim = include[param_name] if type(include[param_name]) is int else len(include[param_name])
            inds = np.arange(dim) + cur_ind
            target_inds[param_name, 'value'] = inds
            cur_ind = inds[-1] + 1

    return cur_ind, target_inds


def get_target_inds_from_params(params, attr_map, include):
    cur_ind = 0
    target_inds = {}
    for param_name in include:
        if param_name in params:
            param = params[param_name]
            for attr in include[param_name]:
                inds = np.arange(len(getattr(param, attr)[:,0])) + cur_ind
                cur_ind = inds[-1] + 1
                target_inds[param.name, attr] = inds
        else:
            dim = include[param_name] if type(include[param_name]) is int else len(include[param_name])
            inds = np.arange(dim) + cur_ind
            target_inds[param_name, 'value'] = inds
            cur_ind = inds[-1] + 1

    return cur_ind, target_inds


def get_plan_to_policy_mapping(plan, robot_name, x_params=[], u_params=[], x_attrs=[], u_attrs=[]):
    '''
    Maps the parameters of the plan actions to indices in the policy state and action vectors, and returns the dimensions of those vectors.
    This mapping should apply to any plan with the given actions

    Parameters:
        action: The action whose eparameters are being mapped
        x_params: Which parameters to include in the state; if none are specified all are included
        u_params: Which parameters to include in the action; if none are specified all robots are included

    Returns:
        The dimension of the state vector
        Mappings from parameters to indices in the state vector
        The dimension of the action vector
        Mappings from paramters to indices in the action vector
    '''
    # assert all(map(lambda a: a.train_policy, plan.actions))
    if not len(plan.actions):
        return 0, {}, 0, {}

    params_to_x_inds, params_to_u_inds = {}, {}
    cur_x_ind, cur_u_ind = 0, 0
    x_params_init, u_params_init = len(x_params), len(u_params)

    if not x_params_init:
        params = list(plan.params.values())
        x_params_init = len(x_params)
    else:
        params = [plan.params[p] for p in x_params]

    robot = plan.params[robot_name] #TODO: Make this more general
    robot_attr_map = const.ATTR_MAP[robot._type]
    ee_pos_attrs = ['ee_left_pos', 'ee_right_pos']
    ee_rot_attrs = ['ee_left_rot', 'ee_right_rot']
    for attr in robot_attr_map:
        if len(u_attrs) and attr[0] not in u_attrs: continue
        # if attr[0] != 'lArmPose' and attr[0] != 'rArmPose' and attr[0] != 'lGripper' and attr[0] != 'rGripper':
        x_inds = attr[1] + cur_x_ind
        cur_x_ind = x_inds[-1] + 1
        params_to_x_inds[(robot.name, attr[0])] = x_inds
        u_inds = attr[1] + cur_u_ind
        cur_u_ind = u_inds[-1] + 1
        params_to_u_inds[(robot.name, attr[0])] = u_inds
    # for attr in ee_pos_attrs:
    #     if attr not in u_attrs: continue
    #     x_inds = np.array([0, 1, 2]) + cur_x_ind
    #     cur_x_ind = x_inds[-1] + 1
    #     params_to_x_inds[(robot.name, attr)] = x_inds
    #     u_inds = np.array([0, 1, 2]) + cur_u_ind
    #     cur_u_ind = u_inds[-1] + 1
    #     params_to_u_inds[(robot.name, attr)] = u_inds
    # for attr in ee_rot_attrs:
    #     if attr not in u_attrs: continue
    #     x_inds = np.array([0, 1, 2, 3]) + cur_x_ind
    #     cur_x_ind = x_inds[-1] + 1
    #     params_to_x_inds[(robot.name, attr)] = x_inds
    #     u_inds = np.array([0, 1, 2, 3]) + cur_u_ind
    #     cur_u_ind = u_inds[-1] + 1
    #     params_to_u_inds[(robot.name, attr)] = u_inds

    # for attr in robot_attr_map:
    #     if len(u_attrs) and attr[0] not in u_attrs: continue
    #     # if attr[0] != 'lArmPose' and attr[0] != 'rArmPose' and attr[0] != 'lGripper' and attr[0] != 'rGripper':
    #     x_vel_inds = attr[1] + cur_x_ind
    #     cur_x_ind = x_vel_inds[-1] + 1
    #     params_to_x_inds[(robot.name, attr[0]+'__vel')] = x_vel_inds
    # for attr in ee_attrs:
    #     if attr not in u_attrs: continue
    #     x_vel_inds = np.array([0, 1, 2]) + cur_x_ind
    #     cur_x_ind = x_vel_inds[-1] + 1
    #     params_to_x_inds[(robot.name, attr+'__vel')] = x_vel_inds

    # for attr in robot_attr_map:
    #     if (robot.name, attr[0]) in params_to_x_inds: continue
    #     x_inds = attr[1] + cur_x_ind
    #     cur_x_ind = x_inds[-1] + 1
    #     params_to_x_inds[(robot.name, attr[0])] = x_inds

    for param in params:
        param_attr_map = const.ATTR_MAP[param._type]
        # Uses all parameters for state unless otherwise specified

        for attr in param_attr_map:
            # TODO: Remove special case for basket
            if (param.name, attr[0]) in params_to_x_inds or param.is_symbol() or (attr[0] not in x_attrs and param.name != 'basket'): continue
            inds = attr[1] + cur_x_ind
            cur_x_ind = inds[-1] + 1
            params_to_x_inds[(param.name, attr[0])] = inds

    symbolic_boundary = cur_x_ind # Used to differntiate parameters from symbols in the state vector

    for param in list(plan.params.values()):
        if not param.is_symbol(): continue
        param_attr_map = const.ATTR_MAP[param._type]
        for attr in param_attr_map:
            if (param.name, attr[0]) in params_to_x_inds: continue
            inds = attr[1] + cur_x_ind
            cur_x_ind = inds[-1] + 1
            params_to_x_inds[(param.name, attr[0])] = inds

    # dX, state index map, dU, (policy) action map
    return cur_x_ind, params_to_x_inds, cur_u_ind, params_to_u_inds, symbolic_boundary

def fill_vector(params, params_to_inds, vec, t, use_symbols=False):
    for param_name, attr in params_to_inds:
        inds = params_to_inds[(param_name, attr)]
        param = params[param_name]
        if param.is_symbol():
            if not use_symbols: continue
            vec[inds] = getattr(param, attr)[:, 0].copy()
        else:
            vec[inds] = getattr(param, attr)[:, t].copy()

def set_params_attrs(params, params_to_inds, vec, t, use_symbols=False, plan=None):
    for param_name, attr in params_to_inds:
        inds = params_to_inds[(param_name, attr)]
        param = params[param_name]
        if param.is_symbol():
            if not use_symbols: continue
            getattr(param, attr)[:, 0] = vec[params_to_inds[(param.name, attr)]]
        else:
            if t >= getattr(param, attr).shape[1]:
                acts = None if plan is None else plan.actions
                print(param, attr, getattr(param, attr).shape, t, acts, 'BAD SHAPE?')
            getattr(param, attr)[:, t] = vec[params_to_inds[(param.name, attr)]]

def fill_sample_from_trajectory(sample, plan, u_vec, noise, t, dX):
    active_ts, params = get_plan_traj_info(plan)

    sample.set(ACTION_ENUM, u_vec, t-active_ts[0])

    X = np.zeros((plan.dX,))
    fill_vector(params, plan.state_inds, X, t)
    sample.set(STATE_ENUM, X, t-active_ts[0])

    sample.set(NOISE_ENUM, noise, t-active_ts[0])

def fill_trajectory_from_sample(sample, plan, time_interval=1, active_ts=None):
    if active_ts is None:
        active_ts = (0, plan.horizon-1) # (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
    for t in range(active_ts[0], active_ts[1]+1):
        X = sample.get_X(t*time_interval-active_ts[0])
        set_params_attrs(plan.params, plan.state_inds, X, t)
    # X = sample.get_X(active_ts[1]*time_interval-1)
    # set_params_attrs(params, plan.state_inds, X, active_ts[1])

def get_trajectory_cost(plan, init_t, final_t, time_interval=POLICY_STEPS_PER_SECOND):
    '''
    Calculates the constraint violations at the provided timestep for the current trajectory, as well as the first and second order approximations.
    This function handles the hierarchies of mappings from parameters to attributes to indices and translates between how the predicates consturct
    those hierachies and how the policy states & actions

    state_inds & action_inds map (param. attr_name) to the the relevant indices
    '''
    preds = []
    state_inds = plan.state_inds
    action_inds = plan.action_inds
    dX = plan.symbolic_bound
    dU = plan.dU
    for action in plan.actions:
        preds.extend(action.preds)
    active_ts = (init_t, final_t)
    T = active_ts[1] - active_ts[0]

    timestep_costs = np.zeros((T*time_interval, )) # l
    first_order_x_approx = np.zeros((T*time_interval, dX, )) # lx
    first_order_u_approx = np.zeros((T*time_interval, dU, )) # lu
    second_order_xx_approx = np.zeros((T*time_interval, dX, dX)) # lxx
    second_order_uu_approx = np.zeros((T*time_interval, dU, dU)) # luu
    second_order_ux_approx = np.zeros((T*time_interval, dU, dX)) # lux

    pred_param_attr_inds = {}
    for p in preds:
        if p['pred'] in pred_param_attr_inds: continue
        pred_param_attr_inds[p['pred']] = {}
        attr_inds = p['pred'].attr_inds
        cur_ind = 0
        for param in attr_inds:
            pred_param_attr_inds[p['pred']][param.name] = {}
            for attr_name, inds in attr_inds[param]:
                pred_param_attr_inds[p['pred']][param.name][attr_name] = np.array(list(range(cur_ind, cur_ind+len(inds))))
                cur_ind += len(inds)

    for t in range(active_ts[0], active_ts[1]-1):
        active_preds = plan.get_active_preds(t+1)
        preds_checked = []
        for p in preds:
            if p['pred'].__class__ not in INCLUDE_PREDS: continue
            if p['pred'] not in active_preds or p['pred'] in preds_checked: continue
            if p['pred'].active_range[1] > active_ts[1]: continue
            attr_inds = p['pred'].attr_inds
            comp_expr = p['pred'].get_expr(negated=p['negated'])

            # Constant terms
            expr = comp_expr.expr if comp_expr else None
            if not expr: continue
            param_vector = p['pred'].get_param_vector(t+1)
            param_vector[np.where(np.isnan(param_vector))] = 0
            # if np.any(np.isnan(param_vector)):
            #     import ipdb; ipdb.set_trace()

            cost_vector = expr.eval(param_vector)
            param_attr_inds = pred_param_attr_inds[p['pred']]
            time_ind = (t-active_ts[0])*time_interval
            timestep_costs[time_ind:time_ind+time_interval] -= np.sum(cost_vector)

            # if np.any(np.isnan(cost_vector)):
            #     import ipdb; ipdb.set_trace()

            # if hasattr(expr, '_grad') and expr._grad:
            #     # Linear terms
            #     first_degree_convexification = expr._grad(param_vector) # expr.convexify(param_vector, degree=1).eval(param_vector)
            #     try:
            #         for param in param_attr_inds:
            #             if plan.params[param].is_symbol(): continue
            #             for attr_name in param_attr_inds[param]:
            #                 if (param, attr_name) in state_inds:
            #                     first_order_x_approx[t-active_ts[0], state_inds[(param, attr_name)]] += np.sum(first_degree_convexification[:, param_attr_inds[param][attr_name]], axis=0)
            #                 if (param, attr_name) in action_inds:
            #                     first_order_u_approx[t-active_ts[0], action_inds[(param, attr_name)]] += np.sum(first_degree_convexification[:, param_attr_inds[param][attr_name]], axis=0)
            #     except Exception as e:
            #         import ipdb; ipdb.set_trace()
            # if hasattr(expr, '_hess') and expr._hess:
            #     # Quadratic terms
            #     try:
            #         second_degree_convexification = expr._hess(param_vector) # expr.convexify(param_vector, degree=2).eval(param_vector)
            #         for param_1 in param_attr_inds:
            #             for param_2 in param_attr_inds:
            #                 if plan.params[param_1].is_symbol() or plan.params[param_2].is_symbol(): continue
            #                 for attr_name_1 in param_attr_inds[param_1]:
            #                     for attr_name_2 in param_attr_inds[param2]:
            #                         if (param_1, attr_name_1) in state_inds and (param_2, attr_name_2) in state_inds:
            #                             x_inds_1 = state_inds[(param_1, attr_name_1)]
            #                             x_inds_2 = state_inds[(param_2, attr_name_2)]
            #                             pred_inds_1 = param_attr_inds[param_1][attr_name_1]
            #                             pred_inds_2 = param_attr_inds[param_2][attr_name_2]
            #                             assert len(x_inds_1) == len(pred_inds_1) and len(x_inds_2) == len(pred_inds_2)
            #                             second_order_xx_approx[t-active_ts[0], x_inds_1, x_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

            #                         if (param_1, attr_name_1) in action_inds and (param_2, attr_name_2) in action_inds:
            #                             u_inds_1 = action_inds[(param_1, attr_name_1)]
            #                             u_inds_2 = action_inds[(param_2, attr_name_2)]
            #                             pred_inds_1 = param_attr_inds[param_1][attr_name_1]
            #                             pred_inds_2 = param_attr_inds[param_2][attr_name_2]
            #                             assert len(u_inds_1) == len(pred_inds_1) and len(u_inds_2) == len(pred_inds_2)
            #                             second_order_uu_approx[t-active_ts[0], u_inds_1, u_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

            #                         if (param_1, attr_name_1) in action_inds and (param_2, attr_name_2) in state_inds:
            #                             u_inds_1 = action_inds[(param_1, attr_name_1)]
            #                             x_inds_2 = state_inds[(param_2, attr_name_2)]
            #                             pred_inds_1 = param_attr_inds[param_1][attr_name_1]
            #                             pred_inds_2 = param_attr_inds[param_2][attr_name_2]
            #                             assert len(u_inds_1) == len(pred_inds_1) and len(x_inds_2) == len(pred_inds_2)
            #                             second_order_ux_approx[t-active_ts[0], u_inds_1, x_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]
            #     except:
            #         import ipdb; ipdb.set_trace()
            preds_checked.append(p['pred'])

    return timestep_costs, first_order_x_approx, first_order_u_approx, second_order_xx_approx, second_order_uu_approx, second_order_ux_approx

# def map_trajectory_to_vel_acc(plan):
#     '''
#         Perform basic kienmatic calculations to find the velocity and acceleration of each joint at each timestep
#     '''
#     params = get_action_params(plan)
#     active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
#     T = active_ts[1] - active_ts[0] + 1
#     vels = np.zeros((plan.dU, T))
#     accs = np.zeros((plan.dU, T))

#     a = np.zeros((plan.dU,))
#     v = np.zeros((plan.dU,))
#     for t in range(active_ts[0], active_ts[1]):
#         U_0 = np.zeros((plan.dU,))
#         U = np.zeros((plan.dU,))
#         fill_vector(params, plan.action_inds, U_0, t)
#         fill_vector(params, plan.action_inds, U, t+1)
#         real_t = plan.time[0, t]
#         vels[:, t-active_ts[0]] = v
#         a = 2*(U-U_0-v*real_t) / (real_t**2)
#         accs[:, t-active_ts[0]] = a
#         v = v + a*real_t
#     vels[:, active_ts[1]-active_ts[0]] = v + a*plan.time[0, active_ts[1]]

#     return vels, accs


# def timestep_vel_acc(plan, t, int_vel, real_ts_offset):
#     '''
#         Perform basic kienmatic calculations to find the velocity and acceleration of each joint at each timestep
#     '''
#     assert t < plan.T
#     params = get_action_params(plan)
#     a = np.zeros((plan.dU,))
#     v = int_vel.copy()
#     U_0 = np.zeros((plan.dU,))
#     U = np.zeros((plan.dU,))
#     fill_vector(params, plan.action_inds, U_0, t)
#     fill_vector(params, plan.action_inds, U, t+1)
#     real_t = plan.time[0, t] - real_ts_offset
#     vels[:, t-active_ts[0]] = v
#     a = 2*(U-U_0-v*real_t) / (real_t**2)

#     return a

# def get_plan_traj_info(plan):
#     '''
#         Extract active timesteps and active parameters from the plan
#     '''
#     active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
#     # if hasattr(plan, 'state_inds'):
#     #     params = list(set(map(lambda k: k[0], plan.state_inds.keys())))
#     # else:
#     #     params = plan.params.values()
#     params = get_state_params(plan)
#     return active_ts, params

def create_sub_plans(plan, action_sequence):
    next_plan_acts = []
    cur_seq_ind = 0
    plans = []
    for i in range(len(plan.actions)):
        act = plan.actions[i]
        if act.name == action_sequnce[cur_seq_ind]:
            next_plan_acts.append(act)
            cur_seq_ind += 1
            if cur_seq_ind >= len(action_sequnce):
                plans.append(Plan(plan.params, next_plan_acts, plan.horizon, plan.env, False))
                next_plan_acts = []
                cur_seq_ind = 0
        else:
            next_plan_acts = []
            cur_seq_ind = 0

    return plans

def get_state_params(plan):
    assert hasattr(plan, 'state_inds')
    params = [plan.params[k[0]] for k in list(plan.state_inds.keys())]
    return list(set(params))

def get_action_params(plan):
    assert hasattr(plan, 'action_inds')
    params = [plan.params[k[0]] for k in list(plan.action_inds.keys())]
    return list(set(params))

def closest_arm_pose(arm_poses, cur_arm_pose):
    min_change = np.inf
    chosen_arm_pose = None
    for arm_pose in arm_poses:
        change = sum((arm_pose - cur_arm_pose)**2)
        if change < min_change:
            chosen_arm_pose = arm_pose
            min_change = change
    return chosen_arm_pose



def check_traj_cost(plan, solver, traj, task):
    if np.any(np.isnan(np.array(traj))):
        raise Exception('Nans in trajectory passed')
    old_free_attrs = plan.get_free_attrs()
    for t in range(0, len(traj)):
        for param_name, attr in plan.state_inds:
            param = plan.params[param_name]
            if param.is_symbol(): continue
            if hasattr(param, attr):
                getattr(param, attr)[:, t] = traj[t, plan.state_inds[param_name, attr]]
                param.fix_attr(attr, (t,t))
                if attr == 'pose' and (param_name, 'rotation') not in plan.state_inds and hasattr(param, 'rotation'):
                    param.rotation[:, t] = 0

                if attr == 'pose':
                    init_target = '{0}_init_target'.format(param_name)
                    if init_target in plan.params:
                        plan.params[init_target].value[:, 0] = param.pose[:, 0]

    for p in list(plan.params.values()):
        if p.is_symbol():
            p.free_all_attr((t,t))
    # Take a quick solve to resolve undetermined symbolic values
    solver._backtrack_solve(plan, time_limit=2, max_priority=-1, task=task)
    plan.store_free_attrs(old_free_attrs)

    viols = plan.check_cnt_violation()
    for i in range(len(viols)):
        if np.isnan(viols[i]):
            viols[i] = 1e3
    plan_total_violation = np.sum(viols) / plan.horizon
    plan_failed_constrs = plan.get_failed_preds_by_type()
    return plan_total_violation, plan_failed_constrs


def plan_to_traj(plan, inds, dX, ts=None, base_t=0):
    if ts is None:
        ts = (0, plan.horizon)
    T = ts[1]-ts[0]
    traj = np.zeros((T, dX))
    for pname, aname in inds:
        if plan.params[pname].is_symbol(): continue
        traj[ts[0]:ts[1]][:, inds[pname, aname]] = getattr(plan.params[pname], aname)[:,ts[0]-base_t:ts[1]-base_t]
    return traj


def find_task_breaks(path, d=5, cont=False, tol=0):
    breaks = [(0,0)]
    tasks = path[0].get(FACTOREDTASK_ENUM)
    cur_task = stats.mode(tasks[:d], axis=0)[0]
    for i, s in enumerate(path):
        if cont:
            raise NotImplementedError
        else:
            tasks = s.get(FACTOREDTASK_ENUM)
            for t in range(0, s.T-d, d):
                next_task = stats.mode(tasks[t:t+d], axis=0)[0]
                if np.any(cur_task != next_task):
                    breaks.append((i, t))
                    cur_task = next_task
    breaks.append((len(path)-1, path[-1].T-1))
    return breaks


def first_failed_state(cost_f, task_breaks, path, cont=False):
    if cont:
        raise NotImplementedError
    else:
        for i, (ind, ts) in enumerate(task_breaks):
            if i < len(task_breaks) - 1:
                next_ind, next_ts = task_breaks[i+1]
            else:
                next_ind, next_ts = len(path)-1, path[-1].T-1
            ts_ran = range(max(0,next_ts-2), min(next_ts+3, path[next_ind].T))
            cost = min([cost_f(path[ind].get_X(t=cur_ts), tuple(path[next_ind].get(FACTOREDTASK_ENUM, t=cur_ts))) for cur_ts in ts_ran])
            if cost > 0:
                print('First failed index:', ind, ts, task_breaks)
                return (ind, ts)
    return None


def get_values(plan):
    vals = {}
    for pname, param in plan.params.items():
        for attr in param._free_attrs:
            vals[pname, attr] = getattr(param, attr).copy()
    return vals


def store_values(plan, vals):
    for param, attr in vals:
        getattr(plan.params[param], attr)[:] = vals[param, attr]

