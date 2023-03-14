# from opentamp.pma.backtrack_ll_solver_OSQP import *

import numpy as np

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
        for act in plan.actions:
            mean_location = act.params[0].value.item()
            set_location = np.random.normal(loc=mean_location, scale=self.sigma)
            if np.abs(set_location - mean_location) < 0.01:
                return True

        return True
