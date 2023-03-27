# from opentamp.pma.backtrack_ll_solver_OSQP import *
from sco_py.expr import AffExpr, BoundExpr, QuadExpr
from sco_py.sco_osqp.prob import Prob
from sco_py.sco_osqp.solver import Solver
from sco_py.sco_osqp.variable import Variable

import numpy as np
import torch as

class ToySolver():
    def __init__(self, sigma):
        self.sigma = sigma

    # used for the ToyDomain demo, just randomize around the proposed location
    # with variation sigma
    def _backtrack_solve(self,
        plan,
        callback=None,
        anum=0,
        verbose=False,
        amax=None,
        n_resamples=5,
        init_traj=[],
        st=0,
        debug=False
    ):
        belief_state = plan.actions[0].params[0].value

        # for now, just verifies if a select location
        for act in plan.actions:
            # only nontrivial action is observation plan
            if act.name == 'observe':
                prob = Prob()

                mean_location = act.params[0].value[0].item()
                set_location = act.params[1].value.item()
                if np.abs(set_location - mean_location) < 0.01:
                    return True

        return False
