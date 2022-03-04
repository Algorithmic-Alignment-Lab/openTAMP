import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import itertools
import os
from opentamp.envs import MJCEnv
import argparse
from core.parsing import parse_domain_config, parse_problem_config
import main
import policy_hooks.namo.sorting_prob_11 as prob_gen
from pma.namo_grip_solver import NAMOSolverOSQP, NAMOSolverGurobi
from pma.hl_solver import *
from pma.pr_graph import *
from pma import backtrack_ll_solver_OSQP as bt_ll_osqp
from pma import backtrack_ll_solver_gurobi as bt_ll_gurobi
from core.util_classes.openrave_body import OpenRAVEBody
from opentamp.src.policy_hooks.utils.policy_solver_utils import *
import os


prob_gen.NUM_OBJS = 2
prob_gen.FIX_TARGETS = True
prob_gen.n_aux = 0
prob_gen.END_TARGETS = prob_gen.END_TARGETS[:8]
prob_gen.domain_file = os.getcwd() + '/opentamp' + "/domains/namo_domain/namo_current_holgrip.domain"
bt_ll_gurobi.DEBUG = True 
bt_ll_gurobi.COL_COEFF = 0.01
bt_ll_osqp.DEBUG = True
bt_ll_osqp.COL_COEFF = 0.01

visual = len(os.environ.get('DISPLAY', '')) > 0
d_c = main.parse_file_to_dict(prob_gen.domain_file)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
prob_file = os.getcwd() + '/opentamp' + "/domains/namo_domain/namo_probs/grip_prob_{}_8end_0aux.prob".format(prob_gen.NUM_OBJS)
goal = '(and '
p_c = main.parse_file_to_dict(prob_file)
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None)

possible_can_locs = list(itertools.product(list(range(-70, 70, 4)), list(range(-60, 0, 4))))
params = problem.init_state.params
inds = [52, 478, 382]
targ_inds = list(range(8))

params['obs0'].pose[:,0] = [-3.5, 0.]
params['pr2'].pose[:,0] = np.array(possible_can_locs[inds[-1]]) / 10.
for n in range(len(prob_gen.END_TARGETS)):
    params['end_target_{}'.format(n)].value[:,0] = prob_gen.END_TARGETS[n]

for n in range(prob_gen.NUM_OBJS):
    params['can{}'.format(n)].pose[:,0] = np.array(possible_can_locs[inds[n]]) / 10.
    goal += '(Near can{} end_target_{})'.format(n, targ_inds[n])
goal += ')'

for pname in params:
    targ = '{}_init_target'.format(pname)
    if targ in params:
        params[targ].value[:,0] = params[pname].pose[:,0]

# NOTE: To use Gurobi instead of OSQP, simply replace the below line with:
# solver = NAMOSolverGurobi()
solver = NAMOSolverOSQP()

hls = FFSolver(d_c)
plan, descr = p_mod_abs(hls, solver, domain, problem, goal=goal, debug=True, n_resamples=5)

if plan is None:
    exit()

fpath = os.getcwd() + '/opentamp'
act_jnts = ['robot_x', 'robot_y', 'robot_theta', 'left_finger_joint', 'right_finger_joint']
items = []
fname = fpath+'/robot_info/lidar_namo.xml'
colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0.7, 0.7, 0.1, 1], [1., 0.1, 0.8, 1], [0.5, 0.95, 0.5, 1], [0.75, 0.4, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]
for n in range(prob_gen.NUM_OBJS):
    cur_color = colors.pop(0)
    targ_color = cur_color[:3] + [1.]
    targ_pos = np.r_[plan.params['can{}'.format(n)].pose[:,-1], -0.15]
    items.append({'name': 'can{}'.format(n), 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 1.5, 0.5), 'dimensions': (0.3, 0.2), 'mass': 40., 'rgba': tuple(cur_color)})
    items.append({'name': 'can{0}_end_target'.format(n), 'type': 'box', 'is_fixed': True, 'pos': targ_pos, 'dimensions': (0.35, 0.35, 0.045), 'rgba': tuple(targ_color), 'mass': 1.})

wall_dims = OpenRAVEBody.get_wall_dims('closet')
for i in range(len(wall_dims)):
    dim, next_trans = wall_dims[i]
    next_trans[0,3] -= 3.5
    next_dim = dim # [dim[1], dim[0], dim[2]]
    pos = next_trans[:3,3] # [next_trans[1,3], next_trans[0,3], next_trans[2,3]]
    items.append({'name': 'wall{0}'.format(i), 'type': 'box', 'is_fixed': True, 'pos': pos, 'dimensions': next_dim, 'rgba': (0.2, 0.2, 0.2, 1)})

config = {'include_files': [fname], 'sim_freq': 50, 'include_items': items, 'act_jnts': act_jnts, 'step_mult': 5e0, 'view': visual, 'timestep': 0.002, 'load_render': True}
env = MJCEnv.load_config(config)
pr2 = plan.params['pr2']
xval, yval = pr2.pose[:,0]
grip = pr2.gripper[0,0]
theta = pr2.theta[0,0]
env.set_joints({'robot_x': xval, 'robot_y': yval, 'left_finger_joint': grip, 'right_finger_joint': grip, 'robot_theta': theta}, forward=False)
for n in range(prob_gen.NUM_OBJS):
    pname = 'can{}'.format(n)
    param = plan.params[pname]
    env.set_item_pos(pname, np.r_[param.pose[:,0], 0.5])

for t in range(plan.horizon-1):
    cur_x, cur_y, _ = env.get_item_pos('pr2')
    cur_theta = env.get_joints(['robot_theta'])['robot_theta'][0]
    cmd_x, cmd_y = pr2.pose[:,t+1] - [cur_x, cur_y]
    cmd_theta = pr2.theta[0,t+1] - cur_theta
    gripper = pr2.gripper[0,t]
    gripper = -0.1 if gripper < 0 else 0.1

    vel_ratio = 0.05
    nsteps = int(max(abs(cmd_x), abs(cmd_y)) / vel_ratio) + 1
    for n in range(nsteps):
        x = cur_x + float(n)/nsteps * cmd_x
        y = cur_y + float(n)/nsteps * cmd_y
        theta = cur_theta + float(n)/nsteps * cmd_theta
        ctrl_vec = np.array([x, y, theta, 5*gripper, 5*gripper])
        env.step(ctrl_vec, mode='velocity', gen_obs=False)
    ctrl_vec = np.array([cur_x+cmd_x, cur_y+cmd_y, cur_theta+cmd_theta, 5*gripper, 5*gripper])
    env.step(ctrl_vec, mode='velocity')
    env.step(ctrl_vec, mode='velocity')
    if visual:
        env.render(camera_id=0, height=256, width=256, view=True)

obj_pos = [env.get_item_pos('can{}'.format(n)) for n in range(prob_gen.NUM_OBJS)]
print('Final obj pos', obj_pos)
