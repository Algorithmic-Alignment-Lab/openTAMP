from opentamp.errors_exceptions import ProblemConfigException

class Problem(object):
    """
    Problem objects are associated with HL search nodes only. Problem objects contain a concrete initial state (see State class)
    and a set of goal predicates (which must hold true in the goal state). Each time an LL search node
    is spawned from the associated HL search node, this initial state is used to initialize parameter trajectories.
    """
    def __init__(self, init_state, goal_preds, env, check_consistent=True, start_action=0, sess=None):
        if not init_state.is_concrete():
            raise ProblemConfigException("Initial state is not concrete. Have all non-symbol parameters been instantiated with a value?")
        if check_consistent and not init_state.is_consistent():
            raise ProblemConfigException("Initial state is not consistent (predicates are violated).")
        self.init_state = init_state
        self.goal_preds = goal_preds
        self.env = env
        self.start_action = start_action
        self.goal = [p.get_rep() for p in goal_preds]
        self.initial = [p.get_rep() for p in init_state.preds]
        self.sess = sess

    def goal_test(self):
        # because problems are associated with HL search nodes,
        # only need to check timestep 0 here
        return all(pred.test(time=0) for pred in self.goal_preds)
