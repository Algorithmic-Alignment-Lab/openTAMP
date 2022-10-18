import opentamp
from opentamp.envs import MJCEnv
import os
import sys
import time

import numpy as np
import pybullet as P
import scipy as sp
from scipy.spatial.transform import Rotation

import opentamp.core.util_classes.transform_utils as T
import opentamp.main as main
from opentamp.core.parsing import parse_domain_config, parse_problem_config
from opentamp.core.util_classes.openrave_body import *
from opentamp.core.util_classes.transform_utils import *
from opentamp.core.util_classes.viewer import PyBulletViewer
from opentamp.pma import backtrack_ll_solver_gurobi as bt_ll
from opentamp.pma.hl_solver import *
from opentamp.pma.pr_graph import *
from opentamp.pma.robot_solver import RobotSolverGurobi
from sco_py.expr import *
import random


bt_ll.DEBUG = True

const.NEAR_GRIP_COEFF = 4e-2
const.NEAR_GRIP_ROT_COEFF = 7e-3
const.NEAR_APPROACH_COEFF = 7e-3
const.NEAR_RETREAT_COEFF = 8e-3
const.NEAR_APPROACH_ROT_COEFF = 1e-3
const.GRASP_DIST = 0.12
const.PLACE_DIST = 0.12
const.APPROACH_DIST = 0.01 
const.RETREAT_DIST = 0.01
const.EEREACHABLE_COEFF = 3e-1
const.EEREACHABLE_ROT_COEFF = 1e-2
const.EEREACHABLE_STEPS = 5
const.EEATXY_COEFF = 5e-2
const.RCOLLIDES_COEFF = 2e-2
const.OBSTRUCTS_COEFF = 2.5e-2

bt_ll.INIT_TRAJ_COEFF = 3e-1
bt_ll.TRAJOPT_COEFF = 1e3
bt_ll.RS_COEFF = 1e2
bt_ll.OSQP_MAX_ITER = int(4e03)
bt_ll.INIT_TRUST_REGION_SIZE = 1e1

openrave_bodies = None
domain_fname = opentamp.__path__._path[0] + "/domains/robot_manipulation_domain/right_desk.domain"
prob = opentamp.__path__._path[0] + "/domains/robot_block_stacking/probs/stack_3_blocks.prob"
d_c = main.parse_file_to_dict(domain_fname)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
hls = FDSolver(d_c, cleanup_files=False)
p_c = main.parse_file_to_dict(prob)
visual = True
solver = RobotSolverGurobi()
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=visual)
params = problem.init_state.params

GOAL = "(and (stacked block0 block1) (stacked block1 block2))"
plan, descr = p_mod_abs(hls, solver, domain, problem, goal=GOAL, debug=True, n_resamples=5, max_iter=2)

# Setup simulator below here

PANDA_XML = opentamp.__path__._path[0] + "/robot_info/robodesk/franka_panda.xml"
HEADER_XML = opentamp.__path__._path[0] + "/robot_info/robodesk/franka_panda_headers.xml"

n_blocks = 3
view = True
config = {
    "obs_include": ["block{0}".format(i) for i in range(n_blocks)],
    "include_files": [PANDA_XML],
    "include_items": [],
    "items": [('robotview', '<camera mode="fixed" name="robotview" pos="2.0 0 2.4" quat="0.653 0.271 0.271 0.653"/>', {})],
    "view": view,
    "load_render": view,
    "sim_freq": 25,
    "timestep": 0.002,
    "image_dimensions": [1024, 1024],
    "step_mult": 5e0,
    "act_jnts": [
        "panda0_joint1",
        "panda0_joint2",
        "panda0_joint3",
        "panda0_joint4",
        "panda0_joint5",
        "panda0_joint6",
        "panda0_joint7",
        "panda0_finger_joint1",
        "panda0_finger_joint2"
    ],
}

for i in range(n_blocks):
    config["include_items"].append({
            "name": "block{0}".format(i),
            "type": "box",
            "is_fixed": False,
            "pos": plan.params['block{}'.format(i)].pose[:,0] - [0., 0.1, 0.61],
            "dimensions": [0.03, 0.03, 0.03],
            "rgba": (0.2, 0.2, 0.2, 1.0),
        })


config["include_items"].append({
        "name": "table",
        "type": "box",
        "is_fixed": True,
        "pos": [0, 0, 0],
        "dimensions": [3., 3., 0.05],
        "rgba": (1.0, 1.0, 1.0, 1.0),
    })

env = MJCEnv.load_config(config)

env.render(view=view)
env.render(view=view)

import ipdb; ipdb.set_trace()
