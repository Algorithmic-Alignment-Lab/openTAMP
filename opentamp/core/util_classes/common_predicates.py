import sys
import traceback

import numpy as np

from opentamp.core.internal_repr.predicate import Predicate
from opentamp.core.util_classes.common_constants import *
from opentamp.core.util_classes.matrix import Vector2d
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
from opentamp.errors_exceptions import PredicateException
from sco_py.expr import AffExpr, EqExpr, Expr, LEqExpr

"""
This file implements the classes for commonly used predicates that are useful in a wide variety of
typical domains.
namo_domain specific predicates can be found in core/util_classes/namo_predicates_gurobi.py
pr2_domain specific predicates can be cound in core/util_classes/pr2_predicates.py
"""

DEFAULT_TOL = 1e-3

def get_param_vector_helper(pred, res_arr, startind, t, attr_inds):
    i = startind
    for p in pred.attr_inds:
        for attr, ind_arr in pred.attr_inds[p]:
            n_vals = len(ind_arr)
            inds = pred.attr_map[p, attr] + startind
            if p.is_symbol():
                # res_arr[i : i + n_vals] = getattr(p, attr)[ind_arr, 0]
                res_arr[inds] = getattr(p, attr)[ind_arr, 0]
            else:
                # res_arr[i : i + n_vals] = getattr(p, attr)[ind_arr, t]
                res_arr[inds] = getattr(p, attr)[ind_arr, t]
            i += n_vals
    return i


class ExprPredicate(Predicate):

    """
    Predicates which are defined by a target value for a set expression.
    """

    def __init__(
        self,
        name,
        expr,
        attr_inds,
        params,
        expected_param_types,
        env=None,
        active_range=(0, 0),
        tol=DEFAULT_TOL,
        priority=0,
        debug=False,
        optimistic=False
    ):
        """
        attr_inds is a dictionary that maps each parameter name to a
        list of (attr, active_inds) pairs. This defines the mapping
        from the primitive predicates of the params to the inputs to
        expr
        """
        super(ExprPredicate, self).__init__(
            name,
            params,
            expected_param_types,
            env=env,
            active_range=active_range,
            priority=priority,
            optimistic=optimistic
        )
        self.expr = expr
        self.attr_inds = attr_inds
        self.tol = tol

        self.x_dim = sum(
            len(active_inds)
            for p_attrs in list(attr_inds.values())
            for (_, active_inds) in p_attrs
        )
        self.set_attr_map(attr_inds)

        start, end = active_range
        self.x_dim *= end + 1 - start
        self.x = np.zeros(self.x_dim)
        self.hl_info = False
        self.hl_ignore = False
        self.hl_include = False
        self.tf_vars = []
        self._rollout = False
        self._nonrollout = False
        self._init_include = True

    def set_attr_map(self, attr_inds):
        self.attr_map = {}
        cur_ind = 0
        for p in attr_inds:
            for attr, inds in attr_inds[p]:
                ninds = len(inds)
                self.attr_map[p, attr] = np.array(range(cur_ind, cur_ind + ninds))
                cur_ind += ninds

    # @profile
    def lazy_spawn_or_body(self, param, name, geom):
        if param.openrave_body is not None:
            assert geom == param.openrave_body._geom
        else:
            param.openrave_body = OpenRAVEBody(self._env, name, geom)
        return param.openrave_body

    def get_expr(self, negated):
        if negated:
            return None
        else:
            return self.expr

    def get_param_vector(self, t):
        end_ind = 0
        start, end = self.active_range
        for rel_t in range(start, end + 1):
            try:
                end_ind = get_param_vector_helper(
                    self, self.x, end_ind, t + rel_t, self.attr_inds
                )
            except IndexError as err:
                traceback.print_exception(*sys.exc_info())
                if end - start >= 1:
                    raise PredicateException(
                        "Insufficient pose trajectory to check dynamic predicate '%s' at the timestep."
                        % self
                    )
                else:
                    raise err
        return self.x.reshape((self.x_dim, 1))

    def hl_test(self, time, negated=False, tol=None):
        return self.test(time, negated, tol)

    # @profile
    def test(self, time, negated=False, tol=None):        
        if self.hl_info:
            return True

        if tol is None:
            tol = self.tol
        if not self.is_concrete():
            return False
        if time < 0:
            print("Out of range time for predicate '%s'." % self)
            #raise PredicateException("Out of range time for predicate '%s'." % self)
        try:
            return self.expr.eval(self.get_param_vector(time), tol=tol, negated=negated)
        except IndexError as err:
            ## this happens with an invalid time
            traceback.print_exception(*sys.exc_info())
            raise PredicateException("Out of range time for predicate '%s'." % self)
        except Exception as err:
            traceback.print_exception(*sys.exc_info())
            print((self, "threw error"))
            raise err

    def unpack(self, y):
        """
        Gradient returned in a similar form to attr_inds
        {param_name: [(attr, (g1,...gi,...gn)]}
        gi are in the same order as the attr_inds list
        """
        res = {}
        i = 0
        for p in self.params:
            res[p.name] = []
            for attr, ind_arr in self.attr_inds[p]:
                n_vals = len(ind_arr)
                res[p.name].append((attr, y[i : i + n_vals]))
                i += n_vals
        return res

    def _grad(self, t):
        return self.expr.grad(self.get_param_vector(t))


# class BeliefExprPredicate(Predicate):
#     def __init__(
#             self,
#             name,
#             expr,
#             attr_inds,
#             params,
#             expected_param_types,
#             env=None,
#             active_range=(0, 0),
#             tol=DEFAULT_TOL,
#             priority=0,
#             debug=False,
#     ):
#         super(BeliefExprPredicate, self).__init__(
#             name,
#             expr,
#             attr_inds,
#             params,
#             expected_param_types,
#             env=None,
#             active_range=(0, 0),
#             tol=DEFAULT_TOL,
#             priority=0,
#             debug=False,
#         )
#
#         self.belief = None
#
#     def refresh_belief(self):
#
#
#     def test(self, time, negated=False, tol = None):
