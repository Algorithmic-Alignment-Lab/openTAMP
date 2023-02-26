import os
import sys

import numpy as np
import pybullet as P
import robosuite
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation

import opentamp.core.util_classes.transform_utils as T
import main
from opentamp.core.parsing import parse_domain_config, parse_problem_config
from opentamp.core.util_classes.openrave_body import *
from opentamp.core.util_classes.transform_utils import *
from pma.hl_solver import *
from pma.pr_graph import *
from pma.robosuite_solver import RobotSolverOSQP
from sco_py.expr import *

# TODO: initialize calls to B.S. planner, add paths to relevant folders
domain_fname = os.getcwd() +
prob = os.getcwd() +
d_c = main.parse_file_to_dict(domain_fname)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
hls = FFSolver(d_c)
p_c = main.parse_file_to_dict(prob)
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=visual)

hls = FFSolver(d_c)
solver = RobotSolverOSQP()
goal =   # NEEDS TO BE STRING

# Run planning to obtain a final plan.
plan, descr = p_mod_abs(
    hls, solver, domain, problem, goal=goal, debug=True, n_resamples=10
)
