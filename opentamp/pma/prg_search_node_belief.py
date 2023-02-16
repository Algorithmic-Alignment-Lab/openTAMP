from opentamp.core.internal_repr.state import State
from opentamp.core.internal_repr.problem import Problem
from opentamp.core.util_classes.learning import PostLearner
from opentamp.pma.prg_search_node import LLSearchNode

import copy
import functools
import random


class LLBeliefSeachNode(LLSearchNode):
    def __init__(self,
                 domain,
                 prob,
                 initial,
                 plan_str=None,
                 plan=None,
                 priority=1,
                 keep_failed=False,
                 ref_plan=None,
                 expansions=0,
                 tol=1e-3,
                 refnode=None,
                 freeze_ts=-1,
                 hl=True,
                 ref_traj=[],
                 nodetype='optimal',
                 env_state=None,
                 label='',
                 info={},
                 x0=None,
                 targets=None,
                 debug=False):

        super().__init__(domain,
                 prob,
                 initial,
                 plan_str=None,
                 plan=None,
                 priority=1,
                 keep_failed=False,
                 ref_plan=None,
                 expansions=0,
                 tol=1e-3,
                 refnode=None,
                 freeze_ts=-1,
                 hl=True,
                 ref_traj=[],
                 nodetype='optimal',
                 env_state=None,
                 label='',
                 info={},
                 x0=None,
                 targets=None,
                 debug=False)

        def plan(self, solver, n_resamples=5, debug=False):
            pass

        def solved(self):
            pass

        def get_failed_pred(self, forward_only=False, st=0):
            pass