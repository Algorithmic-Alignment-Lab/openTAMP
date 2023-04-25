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
# from pma.toy_solver import ToySolver
from pma.toy_solver import ToySolver
from sco_py.expr import Expr, AffExpr, EqExpr, LEqExpr
import torch
import copy

import pyro
import pyro.distributions as dist
from core.util_classes.custom_dist import CustomDist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS

# TODO: initialize calls to planner, add paths to relevant folders
domain_fname = os.getcwd() + "/opentamp/domains/belief_space_domain/toy_belief.domain"
prob = os.getcwd() + "/opentamp/domains/belief_space_domain/probs/toy_belief.prob"

# configuring task plan
d_c = main.parse_file_to_dict(domain_fname)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
hls = FFSolver(d_c)

# configuing motion plan
p_c = main.parse_file_to_dict(prob)
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=False)
solver = ToySolver()


# TODO: do this with the implemented geom logic, do belief space in similar way
def is_in_ray(item, belief):
    return np.pi/2 - 0.1/2 - np.arctan(belief/1.0) <= item <= np.pi/2 + 0.1/2 - np.arctan(belief/1.0) and np.arctan(1) <= item <= np.pi - np.arctan(1)


# NOTE: expected names for pyro samples are "belief_"{param-name}+""
def toy_observation(rs_params, belief_mean, belief_cov):
    # LaPlace estimate: todo SVI if needed
    b_global = pyro.sample('belief_global', dist.Normal(belief_mean, belief_cov))

    if rs_params is None:
        return b_global

    # start observations in the first action todo: loop this over actions in the plan
    obs = torch.torch.empty(rs_params[0].pose.shape[1]-1)
    for a in rs_params:
        for i in range(1, rs_params[0].pose.shape[1]):
            # differentially take conditional depending on the ray
            # 1.10714871779
            if is_in_ray(a.pose[0][i], b_global.item()):
                obs[i-1] = pyro.sample('obs'+str(i), dist.Uniform(b_global.item()-0.001, b_global.item()+0.001))
            else:
                obs[i-1] = pyro.sample('obs'+str(i), dist.Uniform(b_global.item()-1, b_global.item()+1))  # no marginal information gotten

    return obs

# Run planning to obtain a final plan.
plan, descr = p_mod_abs(
    hls, solver, domain, problem,
    goal=None, observation_model=toy_observation, max_likelihood_obs=0.5, debug=False, n_resamples=10
)

if plan is not None:
    print(plan.actions)
    print(plan.params['theta'].pose)  # track pose through time
    print(plan.params['g'].value)  # track goal through time (not modified)
    print(plan.params['g'].belief.samples)

print(descr)

# TODO: implement replan logic when belief-space implemented

# TODO: BCheckSuccess (first pass implement, gets at needed subroutines)