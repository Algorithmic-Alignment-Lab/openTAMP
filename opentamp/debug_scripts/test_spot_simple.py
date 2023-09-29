import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import argparse
import itertools
import os

import opentamp.main as main
from opentamp.core.parsing import parse_domain_config, parse_problem_config
from opentamp.envs import MJCEnv
import opentamp.policy_hooks.spot.spot_prob as prob_gen
from opentamp.pma.robot_solver import RobotSolverOSQP
from opentamp.pma.hl_solver import *
from opentamp.pma.pr_graph import *
from opentamp.pma import backtrack_ll_solver_OSQP as bt_ll_osqp
#from opentamp.pma import backtrack_ll_solver_gurobi as bt_ll_gurobi
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
from opentamp.policy_hooks.utils.policy_solver_utils import *


N_OBJS = 4
N_TARGS = 5
prob_gen.NUM_OBJS = N_OBJS
prob_gen.NUM_TARG = N_TARGS
prob_gen.domain_file = opentamp.__path__._path[0] + "/domains/robot_manipulation_domain/move_robot.domain"
bt_ll_osqp.DEBUG = True
bt_ll_osqp.COL_COEFF = 0.005
visual = False # len(os.environ.get('DISPLAY', '')) > 0
d_c = main.parse_file_to_dict(prob_gen.domain_file)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
prob_file = opentamp.__path__._path[0] + "/domains/robot_manipulation_domain/spot_probs/spot_prob_{}obj_{}targ.prob".format(N_OBJS, N_TARGS)

p_c = main.parse_file_to_dict(prob_file)
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, visual=visual)

possible_can_locs = list(itertools.product(list(range(-50, 50, 4)), list(range(-50, 50, 4))))
params = problem.init_state.params
inds = np.random.choice(range(len(possible_can_locs)), N_OBJS+N_TARGS+1, replace=False)

for n in range(N_TARGS):
    params['spot_pose_{}'.format(n)].position[:,0] = np.array(possible_can_locs[inds[n]]) / 10.
    params['spot_pose_{}'.format(n)].theta[:,0] = np.random.uniform(-np.pi, np.pi)

for n in range(N_OBJS):
    params['can{}'.format(n)].pose[:2,0] = np.array(possible_can_locs[inds[n + N_TARGS]]) / 10.

goal = '(RobotAt spot spot_pose_0)'

for pname in params:
    targ = '{}_init_target'.format(pname)
    if targ in params:
        params[targ].value[:,0] = params[pname].pose[:,0]

# NOTE: To use Gurobi instead of OSQP, simply replace the below line with:
# solver = NAMOSolverGurobi()
solver = RobotSolverOSQP()

USE_FF = False
if USE_FF:
    hls = FFSolver(d_c)
else:
    hls = FDSolver(d_c)
    
print(domain)
print(problem)

plan, descr = p_mod_abs(hls, solver, domain, problem, goal=goal, debug=True, n_resamples=5)

if plan is None:
    exit()

fpath = opentamp.__path__._path[0]
act_jnts = ['robot_x', 'robot_y', 'robot_theta']
items = []
fname = fpath + '/robot_info/spot_simple.xml'
colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0.7, 0.7, 0.1, 1], [1., 0.1, 0.8, 1], [0.5, 0.95, 0.5, 1], [0.75, 0.4, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]
for n in range(N_OBJS):
    cur_color = colors.pop(0)
    targ_color = cur_color[:3] + [1.]
    targ_pos = np.r_[plan.params['can{}'.format(n)].pose[:,-1], -0.15]
    items.append({'name': 'can{}'.format(n), 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 1.5, 0.5), 'dimensions': (0.3, 0.2), 'mass': 40., 'rgba': tuple(cur_color)})


config = {'include_files': [fname], 'sim_freq': 50, 'include_items': items, 'act_jnts': act_jnts, 'step_mult': 5e0, 'view': visual, 'timestep': 0.002, 'load_render': True}
env = MJCEnv.load_config(config)
spot = plan.params['spot']
xval, yval = spot.position[:,0]
theta = spot.theta[0,0]
env.set_joints({'robot_x': xval, 'robot_y': yval, 'robot_theta': theta}, forward=False)
for n in range(N_OBJS):
    pname = 'can{}'.format(n)
    param = plan.params[pname]
    env.set_item_pos(pname, param.pose[:,0])

for t in range(plan.horizon-1):
    cur_jnts = env.get_joints(['robot_x', 'robot_y', 'robot_theta'])
    cur_x, cur_y = cur_jnts['robot_x'][0], cur_jnts['robot_y'][0]
    cur_theta = cur_jnts['robot_theta'][0]

    cmd_x, cmd_y = spot.position[:,t+1] - [cur_x, cur_y]
    cmd_theta = spot.theta[0,t+1] - cur_theta

    vel_ratio = 0.05
    nsteps = int(max(abs(cmd_x), abs(cmd_y)) / vel_ratio) + 1
    for n in range(nsteps):
        x = cur_x + float(n)/nsteps * cmd_x
        y = cur_y + float(n)/nsteps * cmd_y
        theta = cur_theta + float(n)/nsteps * cmd_theta
        ctrl_vec = np.array([x, y, theta])
        env.step(ctrl_vec, mode='velocity', gen_obs=False)
    ctrl_vec = np.array([cur_x+cmd_x, cur_y+cmd_y, cur_theta+cmd_theta])
    env.step(ctrl_vec, mode='velocity')
    env.step(ctrl_vec, mode='velocity')
    if visual:
        env.render(camera_id=0, height=128, width=128, view=True)
    # import ipdb; ipdb.set_trace()
