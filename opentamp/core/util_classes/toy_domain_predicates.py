import sys
import traceback

import numpy as np

from opentamp.core.internal_repr.predicate import Predicate

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


class PointerAtLocation(Predicate):
    def __init__(self,  name, params, expected_param_types, env=None, active_range=(0,0), priority = 0):
        super().__init__(name, params, expected_param_types)

    def test(self, time, negated=False, tol=None):
        if not self.is_concrete():
            return False

        return True

class AlwaysTrue(Predicate):
    def __init__(self,  name, params, expected_param_types, env=None, active_range=(0,0), priority = 0):
        super().__init__(name, params, expected_param_types)

    def test(self, time, negated=False, tol=None):
        if not self.is_concrete():
            return False

        return True