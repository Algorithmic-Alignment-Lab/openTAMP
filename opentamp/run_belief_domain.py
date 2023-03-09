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
from pma.toy_solver import ToySolver
from sco_py.expr import *

# TODO: initialize calls to B.S. planner, add paths to relevant folders
domain_fname = os.getcwd() + "/opentamp/domains/belief_space_domain/toy_belief.domain"
prob = os.getcwd() + "/opentamp/domains/belief_space_domain/probs/toy_belief.prob"

# configuring task plan
d_c = main.parse_file_to_dict(domain_fname)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
print(domain)
hls = FFSolver(d_c)

# configuing motion plan
p_c = main.parse_file_to_dict(prob)
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=False)
print(problem)
solver = ToySolver(sigma=0.1)

# Run planning to obtain a final plan.
plan, descr = p_mod_abs(
    hls, solver, domain, problem,
    goal="(PointerAtLocation p1 g)", debug=False, n_resamples=10
)

# for now, just prints plan (doesn't try to enact plan, no replanning)
print(plan)
print(descr)

# TODO: implement replan logic when belief-space implemented

# TODO: BCheckSuccess (first pass implement, gets at needed subroutines)

# TODO: observation model details (how will observe actions look *concretely*?)