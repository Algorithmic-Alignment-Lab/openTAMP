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
    def __init__(self,  name, params, expected_param_types, env=None, active_range=(0,0), priority = 0, debug=False):
        super().__init__(name, params, expected_param_types)

    def hl_test(self, time, negated=False, tol=None):
        return self.test(time, negated, tol)

    def test(self, time, negated=False, tol=None):
        if not self.is_concrete():
            return False

        value_vec = [getattr(param, 'value') for param in self.params]

        if negated:
            return np.abs(value_vec[0].item() - value_vec[1].item()) >= 0.01
        else:
            return np.abs(value_vec[0].item() - value_vec[1].item()) < 0.01

#
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


class MLPointerAtLocation(Predicate):
    def __init__(self,  name, params, expected_param_types, env=None, active_range=(0,0), priority = 0, debug=False):
        super().__init__(name, params, expected_param_types)

    def hl_test(self, time, negated=False, tol=None):
        return self.test(time, negated, tol)

    def test(self, time, negated=False, tol=None):
        if not self.is_concrete():
            return False

        value_vec = [getattr(param, 'value') for param in self.params]  # these are now individually Gaussians
        if negated:
            return np.abs(value_vec[0].item() - value_vec[1].mean) >= 0.01
        else:
            return np.abs(value_vec[0].item() - value_vec[1].mean) < 0.01


class Uncertain(Predicate):
    def __init__(self,  name, params, expected_param_types, env=None, active_range=(0,0), priority = 0, debug=False):
        super().__init__(name, params, expected_param_types)

    def hl_test(self, time, negated=False, tol=None):
        return self.test(time, negated, tol)

    def test(self, time, negated=False, tol=None):
        if not self.is_concrete():
            return False

        value_vec = [getattr(param, 'value') for param in self.params]  # these are now individually Gaussians

        # only expecting one argument here: implement uncertainty logic
        if negated:
            return value_vec[0].variance < 0.05
        else:
            return value_vec[0].variance >= 0.05


class UncertainTest(Predicate):
    def __init__(self,  name, params, expected_param_types, env=None, active_range=(0,0), priority = 0, debug=False):
        super().__init__(name, params, expected_param_types)

    def hl_test(self, time, negated=False, tol=None):
        return self.test(time, negated, tol)

    def test(self, time, negated=False, tol=None):
        if not self.is_concrete():
            return False

        value_vec = [getattr(param, 'value') for param in self.params]  # NOT individually Gaussians

        # placeholder in debugging
        return True


class AlwaysTrue(Predicate):
    def __init__(self,  name, params, expected_param_types, env=None, active_range=(0,0), priority = 0, debug=False):
        super().__init__(name, params, expected_param_types)

    def hl_test(self, time, negated=False, tol=None):
        return self.test(time, negated, tol)

    def test(self, time, negated=False, tol=None):
        if not self.is_concrete():
            return False

        return not negated
