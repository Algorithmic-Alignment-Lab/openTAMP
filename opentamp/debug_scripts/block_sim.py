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
import main
from opentamp.core.parsing import parse_domain_config, parse_problem_config
from opentamp.core.util_classes.openrave_body import *
from opentamp.core.util_classes.transform_utils import *
from opentamp.core.util_classes.viewer import PyBulletViewer
from pma import backtrack_ll_solver_OSQP as bt_ll
from pma.hl_solver import *
from pma.pr_graph import *
from pma.robot_solver import RobotSolverOSQP
from sco_py.expr import *
import random


# bt_ll.DEBUG = True
# openrave_bodies = None
# domain_fname = opentamp.__path__._path[0] + "/domains/robot_manipulation_domain/right_desk.domain"
# prob = opentamp.__path__._path[0] + "/domains/robot_block_stacking/probs/stack_3_blocks.prob"
# d_c = main.parse_file_to_dict(domain_fname)
# domain = parse_domain_config.ParseDomainConfig.parse(d_c)
# hls = FDSolver(d_c, cleanup_files=False)
# p_c = main.parse_file_to_dict(prob)
# visual = True
# problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=visual)
# params = problem.init_state.params

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
            "pos": [0.3, i / 10., 0.02],
            "dimensions": [0.01, 0.01, 0.01],
            "rgba": (0.2, 0.2, 0.2, 1.0),
        })


config["include_items"].append({
        "name": "table",
        "type": "box",
        "is_fixed": True,
        "pos": [0, 0, 0],
        "dimensions": [2., 2., 0.01],
        "rgba": (1.0, 1.0, 1.0, 1.0),
    })

env = MJCEnv.load_config(config)

env.render(view=view)
env.render(view=view)

import ipdb; ipdb.set_trace()
