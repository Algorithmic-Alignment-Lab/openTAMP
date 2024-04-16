from sco_py.expr import LEqExpr, DEFAULT_TOL
import numpy as np

class LEqEpsExpr(LEqExpr):
    def __init__(self, expr, val, conf=0.95):
        super().__init__(expr, val)
        self.conf = conf

    def eval(self, x, tol=DEFAULT_TOL, negated=False):
        """
        Tests whether the expression at x is less than or equal to self.val with
        tolerance tol.
        """
        assert tol >= 0.0
        expr_val = self.expr.eval(x)
        if negated:
            ## need the tolerance to go the other way if its negated
            return not (expr_val <= self.val - tol * np.ones(expr_val.shape)).mean() >= self.conf
        else:
            return (expr_val <= self.val + tol * np.ones(expr_val.shape)).mean() >= self.conf
