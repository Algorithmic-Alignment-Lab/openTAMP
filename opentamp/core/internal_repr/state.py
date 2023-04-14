class State(object):
    """
    A state is parametrized by parameters, a 0-indexed timestep, and predicates (see Predicate class). It maintains the
    predicates that hold true at that timestep. A concrete state is one in which all the predicates are concrete.

    NOTE: Currently, we only use this class in conjunction with Problem objects' concrete initial states, for
    HL search nodes. At the low level, the state at each timestep is implicit in the parameter trajectory tables.
    """
    def __init__(self, name, params, preds=None, timestep=0, invariants=None):
        self.name = name
        self.params = params
        self.preds = set(preds) if preds else set()
        self.timestep = timestep
        self.invariants = set(invariants) if invariants else set()

    def is_concrete(self):
        for p in self.params.values():
            if not p.is_symbol() and not p.is_defined():
                return False
        return True

    def is_consistent(self):
        preds = list(self.preds) + list(self.invariants)
        for pred in preds:
            if pred.active_range != (0,0): continue
            if pred.is_concrete() and not pred.test(time=self.timestep):
                print("Initial State Not Consistent with predicates {} at time {}".format(pred, self.timestep))
                return False
        return True
