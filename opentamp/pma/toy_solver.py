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
        if plan:
            # for now, just verifies if a select location
            for act in plan.actions:
                if act.name == 'select_location':
                    print('enter solver')
                    print(act.preds)
                    mean_location = act.params[0].value[0].item()
                    set_location = act.params[1].value.item()
                    if np.abs(set_location - mean_location) < 0.01:
                        return True

        return False
