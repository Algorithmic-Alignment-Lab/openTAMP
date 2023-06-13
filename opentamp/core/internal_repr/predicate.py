from opentamp.errors_exceptions import ParamValidationException
import numpy as np

class Predicate(object):
    """
    Predicates hold a set of parameters (see Parameter class) and represent testable relationships among
    these parameters. The test occurs for a particular time (0-indexed). A concrete predicate is one in which all
    the non-symbol parameters have values. Commonly used predicates can be found in the core/util_classes/ folder.
    """
    def __init__(self, name, params, expected_param_types, env=None, active_range=(0,0), priority = 0):
        self.name = name
        self.params = params
        self.validate_params(expected_param_types)
        self.env = env
        self.active_range = active_range
        self.priority = priority

    def get_type(self):
        return self.__class__.__name__

    def is_concrete(self):
        for param in self.params:
            if not param.is_defined():
                return False
        return True

    def test(self, time, negated=False, tol = None):
        if not self.is_concrete():
            return False
        raise NotImplementedError("Override this.")

    def resample(self, negated, time, plan):
        return None, None

    def validate_params(self, expected_param_types):
        try:
            if len(self.params) != len(expected_param_types):
                raise ParamValidationException("Parameter type validation failed for predicate '%s'."%self)
            for i, p in enumerate(self.params):
                if expected_param_types[i] not in p.get_type(True):
                    raise ParamValidationException("Parameter type validation failed for predicate '%s'."%self)
        except Exception as e:
            print(e)

    def check_pred_violation(self, t, negated=False, tol=1e-3):
        expr = self.get_expr(negated=negated)
        # if not self.test(t, negated=negated, tol=tol):
        #     return None
        if expr is None: 
            violation = np.abs(self.expr.expr.eval(self.get_param_vector(t)))
        else:
            violation = np.abs(expr.expr.eval(self.get_param_vector(t)))

        return violation


    def get_rep(self):
        s = "(%s "%(self.get_type())
        for param in self.params[:-1]:
            s += param.name + " "
        s += self.params[-1].name + ") "
        return s

    def __repr__(self):
        s = "%s: (%s "%(self.name, self.get_type())
        for param in self.params[:-1]:
            s += param.name + " "
        s += self.params[-1].name + ")"
        return s
