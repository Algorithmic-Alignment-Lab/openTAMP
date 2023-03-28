import sys
import traceback
from collections import OrderedDict
from sco_py.expr import Expr, AffExpr, EqExpr, LEqExpr

import numpy as np

from opentamp.core.internal_repr.predicate import Predicate
from opentamp.core.util_classes.common_predicates import ExprPredicate

# class PointerAtTarget(Predicate):
#     def __init__(self,  name, params, expected_param_types, env=None, active_range=(0,0), priority = 0):
#         super().__init__(name, params, expected_param_types)
#
#     def test(self, time, negated=False, tol=None):
#         if not self.is_concrete():
#             return False
#
#         return True
#
#
# class PointerAtGoal(Predicate):
#     def __init__(self,  name, params, expected_param_types, env=None, active_range=(0,0), priority = 0):
#         super().__init__(name, params, expected_param_types)
#
#     def test(self, time, negated=False, tol=None):
#         if not self.is_concrete():
#             return False
#
#         return True


class PointerInRange(ExprPredicate):
    pass


class PointerAtTarget(ExprPredicate):
    pass


class PointerAtLocation(ExprPredicate):
    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        active_range=(0, 0),
        priority=0,
        debug=False,
    ):
        attr_inds = OrderedDict([
            (params[0], [("pose", np.array([0], dtype='int32'))]),
            (params[1], [("value", np.array([0], dtype='int32'))])
        ])

        aff_expr = AffExpr(np.array([[1, -1]]), np.array([0]))  # takes the difference between the input
        e = EqExpr(aff_expr, np.array([0]))

        super().__init__(name, e, attr_inds, params, expected_param_types)

# class PointerAtGoal(Predicate):
#     def __init__(self,  name, params, expected_param_types, env=None, active_range=(0,0), priority = 0, debug=False):
#         super().__init__(name, params, expected_param_types)
#
#     def test(self, time, negated=False, tol=None):
#         if not self.is_concrete():
#             return False
#
#         value_vec = [getattr(param, 'value') for param in self.params]
#
#         return np.abs(value_vec[0].item() - value_vec[1].item()) < 0.01


# class MLPointerAtLocation(ExprPredicate):
#     def __init__(
#         self,
#         name,
#         params,
#         expected_param_types,
#         env=None,
#         active_range=(0, 0),
#         priority=0,
#         debug=False,
#     ):
#         super().__init__()
#
    # def test(self, time, negated=False, tol=None):
    #     if not self.is_concrete():
    #         return False
    #
    #     value_vec = [getattr(param, 'value') for param in self.params]  # these are now individually Gaussians
    #     if negated:
    #         return np.abs(value_vec[0].item() - value_vec[1].mean) >= 0.01
    #     else:
    #         return np.abs(value_vec[0].item() - value_vec[1].mean) < 0.01


class Uncertain(ExprPredicate):
    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        active_range=(0, 0),
        priority=0,
        debug=False,
        sigma=0.1
    ):
        # variance is sufficiently small
        attr_inds = OrderedDict([
            (params[0], [("value", np.array([1], dtype='int32'))]),
        ])

        self.sigma = sigma # the baseline amount of uncertainty
        aff_expr = AffExpr(np.array([[-1]]), np.array([0]))  # trivial constraint
        e = LEqExpr(aff_expr, np.array([-self.sigma]))

        super().__init__(name, e, attr_inds, params, expected_param_types)


class AngleViewingWall(ExprPredicate):
    def __init__(
            self,
            name,
            params,
            expected_param_types,
            env=None,
            active_range=(0, 0),
            priority=0,
            debug=False,
            sigma=0.1
    ):
        attr_inds = OrderedDict([
            (params[0], [("pose", np.array([0], dtype='int32'))]),
        ])

        aff_expr = AffExpr(np.array([[-1],[1]]), np.array([[0],[0]]))  # trivial constraint
        wu = params[1].value
        wd = params[2].value
        d = params[3].value

        e = LEqExpr(aff_expr, np.array([[-np.arctan(d/wu)], [np.pi + np.arctan(d/wd)]]))

        super().__init__(name, e, attr_inds, params, expected_param_types)


class CoordInView(ExprPredicate):
    def __init__(
            self,
            name,
            params,
            expected_param_types,
            env=None,
            active_range=(0, 0),
            priority=0,
            debug=False,
            sigma=0.1
    ):
        attr_inds = OrderedDict([
            (params[0], [("pose", np.array([0], dtype='int32'))]),
        ])

        g = params[1].value
        d = params[2].value
        r = params[3].value

        aff_expr = AffExpr(np.array([[-1],[1]]), np.array([[0],[0]]))  # trivial constraint
        e = LEqExpr(aff_expr, np.array([[-np.pi/2 + r/2 + np.arctan(g/d)], [np.pi/2 + r/2 - np.arctan(g/d)]]))

        super().__init__(name, e, attr_inds, params, expected_param_types)


# class UncertainTest(ExprPredicate):
#     def __init__(
#         self,
#         name,
#         params,
#         expected_param_types,
#         env=None,
#         active_range=(0, 0),
#         priority=0,
#         debug=False,
#     ):
#         attr_inds = OrderedDict([
#             (params[0], [("value", np.array([0], dtype='int32'))]),
#         ])
#
#         aff_expr = AffExpr(np.array([[0]]), np.array([0])) # trivial constraint
#         e = EqExpr(aff_expr, np.array([0]))
#
#         super().__init__(name, e, attr_inds, params, expected_param_types)


# used for vacuous preconditions
class AlwaysTrue(ExprPredicate):
    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        active_range=(0, 0),
        priority=0,
        debug=False,
    ):
        attr_inds = OrderedDict([
            (params[0], [("pose", np.array([0], dtype='int32'))]),
        ])

        aff_expr = AffExpr(np.array([[0]]), np.array([0]))  # trivial constraint
        e = EqExpr(aff_expr, np.array([0]))

        super().__init__(name, e, attr_inds, params, expected_param_types)