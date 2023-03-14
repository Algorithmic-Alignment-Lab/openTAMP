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
            (params[0], ["value", np.array([0], dtype='int32')]),
            (params[1], ["value", np.array([0], dtype='int32')])
        ])

        print([a for a in attr_inds])

        e = EqExpr(getattr(params[0], 'value'), getattr(params[1], 'value'))

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


# class Uncertain(ExprPredicate):
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
#     def test(self, time, negated=False, tol=None):
#         if not self.is_concrete():
#             return False
#
#         value_vec = [getattr(param, 'value') for param in self.params]  # these are now individually Gaussians
#
#         # only expecting one argument here: implement uncertainty logic
#         if negated:
#             return value_vec[0].variance < 0.05
#         else:
#             return value_vec[0].variance >= 0.05


class UncertainTest(Predicate):
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
        super().__init__(name, params, expected_param_types)

    def test(self, time, negated=False, tol=None):
        if not self.is_concrete():
            return False

        return True


# used for vacuous preconditions
class AlwaysTrue(Predicate):
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
        super().__init__(name, params, expected_param_types)

    def test(self, time, negated=False, tol=None):
        if not self.is_concrete():
            return False

        return True
