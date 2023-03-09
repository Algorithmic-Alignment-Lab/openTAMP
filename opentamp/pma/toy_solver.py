# from opentamp.pma.backtrack_ll_solver_OSQP import *

class ToySolver():
    def __init__(self, sigma):
        self.sigma = 0

    # used for the ToyDomain demo
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
        print(plan.actions)
        return True