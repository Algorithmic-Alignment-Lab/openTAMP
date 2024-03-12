import itertools
import random

import numpy as np

from opentamp.core.util_classes import common_predicates
from opentamp.core.util_classes.matrix import Vector, Vector2d

from sco_py.expr import AffExpr, BoundExpr, EqExpr, QuadExpr
from sco_py.sco_osqp.osqp_utils import OSQPLinearConstraint, OSQPVar
from sco_py.sco_osqp.prob import Prob
from sco_py.sco_osqp.solver import Solver
from sco_py.sco_osqp.variable import Variable

MAX_PRIORITY = 5
WIDTH = 7
HEIGHT = 2
TRAJOPT_COEFF = 1e0
dsafe = 1e-1


class LLSolverOSQP(object):
    """
    LLSolver solves the underlying optimization problem using motion planning.
    This is where different refinement strategies (e.g. backtracking,
    randomized), different motion planners, and different optimization
    strategies (global, sequential) are implemented.
    """

    def solve(self, plan):
        raise NotImplementedError("Override this.")

    # NOTE: Probably want to keep this, but just have it return the relevant portions of matrix, not
    # make the mapping
    def _spawn_sco_var_for_pred(self, pred, t):
        x = np.empty(pred.x_dim, dtype=object)
        v = np.empty(pred.x_dim)
        i = 0
        start, end = pred.active_range
        for rel_t in range(start, end + 1):
            for p in pred.attr_inds:
                for attr, ind_arr in pred.attr_inds[p]:
                    n_vals = len(ind_arr)
                    ll_p = self._param_to_ll[p]
                    if p.is_symbol():
                        x[i : i + n_vals] = getattr(ll_p, attr)[ind_arr, 0]
                        v[i : i + n_vals] = getattr(p, attr)[ind_arr, 0]
                    else:
                        #breakpoint()

                        x[i : i + n_vals] = getattr(ll_p, attr)[
                            ind_arr, t + rel_t - self.ll_start
                        ]
                        v[i : i + n_vals] = getattr(p, attr)[ind_arr, t + rel_t]
                    i += n_vals

        assert i >= pred.x_dim
        x = x.reshape((pred.x_dim, 1))
        v = v.reshape((pred.x_dim, 1))
        return Variable(x, v)


class LLParamOSQP(object):
    """
    LLParamOSQP keeps track of the low-level representation of parameters
    that is used in the optimization. For every numerical attribute in the
    original parameter, LLParam keeps track of how the corresponding attribute
    is represented in the constructed OSQP problem (i.e, what row of the matrix
    this particular attribute is in). LLParam updates its parameter based off of
    the returned value of these variables after OSQP solving.
    """

    def __init__(self, param, horizon, active_ts):
        self._param = param
        self._horizon = horizon
        self._num_attrs = []
        self.active_ts = active_ts

    def create_osqp_vars(self):
        """
        Creates OSQPVars for attributes of certain types.
        This is analogous to the 'create_grb_vars' function from the original
        LLParam class.
        The created variable names will be useful when creating sco_osqp
        Variables.
        """
        for k, _ in list(self._param.__dict__.items()):
            rows = None
            attr_type = self._param.get_attr_type(k)
            if issubclass(attr_type, Vector):
                rows = attr_type.dim

            if rows is not None:
                self._num_attrs.append(k)

                shape = None
                value = None
                if self._param.is_symbol():
                    shape = (rows, 1)
                else:
                    shape = (rows, self._horizon)
                x = np.empty(shape, dtype=object)
                name = "({}-{}-{})"
                for index in np.ndindex(shape):
                    # NOTE: it is easier for the sco code and update_param to
                    # # handle things if everything is a variable
                    x[index] = OSQPVar(name.format(self._param.name, k, index))

                setattr(self, k, x)

    def batch_add_cnts(self):
        """
        Adds all the linear equality constraints.
        """
        linear_eq_cnts_list = []
        if self._param.is_defined():
            for attr in self._num_attrs:
                osqp_vars = getattr(self, attr)
                value = self.get_param_val(attr)
                free_vars = self.get_free_vars(attr)
                for index, val in np.ndenumerate(value):
                    if not free_vars[index] and not np.any(np.isnan(val)):
                        linear_eq_cnts_list.append(
                            OSQPLinearConstraint(
                                np.array([osqp_vars[index]]),
                                np.array([1]),
                                np.array([val]),
                                np.array([val]),
                            )
                        )
        return linear_eq_cnts_list

    def _get_attr_val(self, attr):
        osqp_vars = getattr(self, attr)
        value = np.zeros(osqp_vars.shape)
        for index, var in np.ndenumerate(osqp_vars):
            value[index] = var.val
        return value

    def update_param(self):
        """
        Updates the numerical attributes in the original parameter based off of
        the attribute's corresponding OSQP solution variables.
        """
        for attr in self._num_attrs:
            value = self._get_attr_val(attr)

            # if attr == "pose":
            #     import pdb

            #     pdb.set_trace()

            if np.any(np.isnan(value)):
                continue
            self.set_param_val(attr, value)

    def get_param_val(self, attr):
        if self._param.is_symbol():
            return getattr(self._param, attr)[:, 0][:, None]
        return getattr(self._param, attr)[:, self.active_ts[0] : self.active_ts[1] + 1]

    def get_free_vars(self, attr):
        if self._param.is_symbol():
            return self._param._free_attrs[attr][:, 0][:, None]
        return self._param._free_attrs[attr][
            :, self.active_ts[0] : self.active_ts[1] + 1
        ]

    def set_param_val(self, attr, value):
        assert not np.any(np.isnan(value))
        if self._param.is_symbol():
            setattr(self._param, attr, value)
        getattr(self._param, attr)[:, self.active_ts[0] : self.active_ts[1] + 1] = value

        # if attr == "pose":
        #     import pdb

        #     pdb.set_trace()

        assert np.allclose(self.get_param_val(attr), value)


class NAMOSolverOSQP(LLSolverOSQP):
    def __init__(self, early_converge=True, transfer_norm="min-vel"):
        self.transfer_coeff = 1e1
        self.rs_coeff = 1e6
        self.init_penalty_coeff = 1e2
        self.child_solver = None
        self._param_to_ll = {}
        self._failed_groups = []

        self.early_converge = early_converge
        self.transfer_norm = transfer_norm

    def backtrack_solve(self, plan, callback=None, verbose=False):
        plan.save_free_attrs()
        success = self._backtrack_solve(plan, callback, anum=0, verbose=verbose)
        plan.restore_free_attrs()
        return success

    def _backtrack_solve(self, plan, callback=None, anum=0, verbose=False):
        if anum > len(plan.actions) - 1:
            return True
        a = plan.actions[anum]
        active_ts = a.active_timesteps
        inits = {}
        if a.name == "moveto":
            ## find possible values for the final pose
            rs_param = a.params[2]
        elif a.name == "movetoholding":
            ## find possible values for the final pose
            rs_param = a.params[2]
        elif a.name == "grasp":
            ## sample the grasp/grasp_pose
            rs_param = a.params[4]
        elif a.name == "putdown":
            ## sample the end pose
            rs_param = a.params[4]
        else:
            raise NotImplemented

        def recursive_solve():
            ## don't optimize over any params that are already set
            old_params_free = {}
            for p in plan.params.values():
                if p.is_symbol():
                    if p not in a.params:
                        continue
                    old_params_free[p] = p._free_attrs["value"].copy()
                    p._free_attrs["value"][:] = 0
                else:
                    old_params_free[p] = p._free_attrs["pose"][:, active_ts[1]].copy()
                    p._free_attrs["pose"][:, active_ts[1]] = 0
            self.child_solver = NAMOSolver()
            if self.child_solver._backtrack_solve(
                plan, callback=callback, anum=anum + 1, verbose=verbose
            ):
                return True
            ## reset free_attrs
            for p in a.params:
                if p.is_symbol():
                    if p not in a.params:
                        continue
                    p._free_attrs["value"] = old_params_free[p]
                else:
                    p._free_attrs["pose"][:, active_ts[1]] = old_params_free[p]
            return False

        if not np.all(rs_param._free_attrs["value"]):
            ## this parameter is fixed
            if callback is not None:
                callback_a = lambda: callback(a)
            else:
                callback_a = None
            self.child_solver = NAMOSolver()
            success = self.child_solver.solve(
                plan,
                callback=callback_a,
                n_resamples=0,
                active_ts=active_ts,
                verbose=verbose,
                force_init=True,
            )

            if not success:
                ## if planning fails we're done
                return False
            ## no other options, so just return here
            return recursive_solve()

        ## so that this won't be optimized over
        rs_free = rs_param._free_attrs["value"].copy()
        rs_param._free_attrs["value"][:] = 0

        targets = plan.get_param("InContact", 2, {1: rs_param}, negated=False)

        if callback is not None:
            callback_a = lambda: callback(a)
        else:
            callback_a = None

        robot_poses = []

        if len(targets) == 0 or np.all(targets[0]._free_attrs["value"]):
            ## sample 4 possible poses
            coords = list(itertools.product(list(range(WIDTH)), list(range(HEIGHT))))
            random.shuffle(coords)
            robot_poses = [np.array(x)[:, None] for x in coords[:4]]
        elif np.any(targets[0]._free_attrs["value"]):
            ## there shouldn't be only some free_attrs set
            raise NotImplementedError
        else:
            grasp_dirs = [
                np.array([0, -1]),
                np.array([1, 0]),
                np.array([0, 1]),
                np.array([-1, 0]),
            ]
            grasp_len = plan.params["pr2"].geom.radius + targets[0].geom.radius - dsafe
            for g_dir in grasp_dirs:
                grasp = (g_dir * grasp_len).reshape((2, 1))
                robot_poses.append(targets[0].value + grasp)

        for rp in robot_poses:
            rs_param.value = rp
            success = False
            self.child_solver = NAMOSolver()
            success = self.child_solver.solve(
                plan,
                callback=callback_a,
                n_resamples=0,
                active_ts=active_ts,
                verbose=verbose,
                force_init=True,
            )
            if success:
                if recursive_solve():
                    break
                else:
                    success = False
        rs_param._free_attrs["value"] = rs_free
        return success

    def solve(
        self,
        plan,
        callback=None,
        n_resamples=5,
        active_ts=None,
        verbose=False,
        force_init=False,
    ):
        success = False

        if force_init or not plan.initialized:
            ## solve at priority -1 to get an initial value for the parameters
            self._solve_opt_prob(
                plan,
                priority=-1,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
            )
            plan.initialized = True
        success = self._solve_opt_prob(
            plan, priority=1, callback=callback, active_ts=active_ts, verbose=verbose
        )
        success = plan.satisfied(active_ts)
        if success:
            return success

        for _ in range(n_resamples):
            ## refinement loop
            ## priority 0 resamples the first failed predicate in the plan
            ## and then solves a transfer optimization that only includes linear constraints
            self._solve_opt_prob(
                plan,
                priority=0,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
            )
            success = self._solve_opt_prob(
                plan,
                priority=1,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
            )
            success = plan.satisfied(active_ts)
            if success:
                return success
        return success

    # #@profile
    def _solve_opt_prob(
        self,
        plan,
        priority,
        callback=None,
        active_ts=None,
        verbose=False,
        resample=True,
    ):
        ## active_ts is the inclusive timesteps to include
        ## in the optimization
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)

        model = grb.Model()
        model.params.OutputFlag = 0
        self._prob = Prob(model, callback=callback)
        # param_to_ll_old = self._param_to_ll.copy()
        self._spawn_parameter_to_ll_mapping(model, plan, active_ts)
        model.update()

        self._bexpr_to_pred = {}

        if priority == -1:
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_first_and_last_timesteps_of_actions(
                plan, priority=-1, active_ts=active_ts, verbose=verbose
            )
            tol = 1e-2
        elif priority == 0:
            ## this should only get called with a full plan for now
            assert active_ts == (0, plan.horizon - 1)

            obj_bexprs = []

            if resample:
                failed_preds = plan.get_failed_preds()
                ## this is an objective that places
                ## a high value on matching the resampled values
                obj_bexprs.extend(self._resample(plan, failed_preds))

            ## solve an optimization movement primitive to
            ## transfer current trajectories
            obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))
            self._add_obj_bexprs(obj_bexprs)
            # self._add_first_and_last_timesteps_of_actions(
            #     plan, priority=0, add_nonlin=False)
            self._add_first_and_last_timesteps_of_actions(
                plan, priority=-1, add_nonlin=True, verbose=verbose
            )
            self._add_all_timesteps_of_actions(
                plan, priority=0, add_nonlin=False, verbose=verbose
            )
            tol = 1e-2
        elif priority == 1:
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_all_timesteps_of_actions(
                plan, priority=1, add_nonlin=True, active_ts=active_ts, verbose=verbose
            )
            tol = 1e-4

        solv = Solver()
        solv.initial_penalty_coeff = self.init_penalty_coeff
        success = solv.solve(self._prob, method="penalty_sqp", tol=tol, verbose=verbose)
        self._update_ll_params()
        self._failed_groups = self._prob.nonconverged_groups
        return success

    def get_value(self, plan, penalty_coeff=1e0):
        model = grb.Model()
        model.params.OutputFlag = 0
        self._prob = Prob(model)
        # param_to_ll_old = self._param_to_ll.copy()
        self._spawn_parameter_to_ll_mapping(model, plan)
        model.update()
        self._bexpr_to_pred = {}

        obj_bexprs = self._get_trajopt_obj(plan)
        self._add_obj_bexprs(obj_bexprs)
        self._add_all_timesteps_of_actions(
            plan, priority=1, add_nonlin=True, verbose=False
        )
        return self._prob.get_value(penalty_coeff)

    def _get_transfer_obj(self, plan, norm):
        transfer_objs = []
        if norm in ["min-vel", "l2"]:
            for param in list(plan.params.values()):
                # if param._type in ['Robot', 'Can']:
                K = 2
                if param.is_symbol():
                    T = 1
                    pose = param.value
                else:
                    T = plan.horizon
                    pose = param.pose
                assert (K, T) == pose.shape
                KT = K * T
                if norm == "min-vel" and not param.is_symbol():
                    v = -1 * np.ones((KT - K, 1))
                    d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                    # [:,0] allows numpy to see v and d as one-dimensional so
                    # that numpy will create a diagonal matrix with v and d as a diagonal
                    P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                    Q = np.dot(np.transpose(P), P)
                else:  ## l2
                    Q = np.eye(KT)
                cur_pose = pose.reshape((KT, 1), order="F")
                A = -2 * cur_pose.T.dot(Q)
                b = cur_pose.T.dot(Q.dot(cur_pose))
                # QuadExpr is 0.5*x^Tx + Ax + b
                quad_expr = QuadExpr(
                    2 * self.transfer_coeff * Q,
                    self.transfer_coeff * A,
                    self.transfer_coeff * b,
                )
                ll_param = self._param_to_ll[param]
                if param.is_symbol():
                    ll_grb_vars = ll_param.value.reshape((KT, 1), order="F")
                else:
                    ll_grb_vars = ll_param.pose.reshape((KT, 1), order="F")
                bexpr = BoundExpr(quad_expr, Variable(ll_grb_vars, cur_pose))
                transfer_objs.append(bexpr)
        elif norm == "straightline":
            return self._get_trajopt_obj(plan)
        else:
            raise NotImplementedError
        return transfer_objs

    def _resample(self, plan, preds):
        val, attr_inds = None, None
        for negated, pred, t in preds:
            ## returns a vector of new values and an
            ## attr_inds (OrderedDict) that gives the mapping
            ## to parameter attributes
            if (
                self.early_converge
                and self._failed_groups != ["all"]
                and self._failed_groups != []
                and not np.any(
                    [param.name in self._failed_groups for param in pred.params]
                )
            ):
                ## using early converge and one group didn't converge
                ## but no parameter from this pred in that group
                continue

            val, attr_inds = pred.resample(negated, t, plan)
            ## no resample defined for that pred
            if val is not None:
                break
        if val is None:
            return []

        t_local = t
        bexprs = []
        i = 0
        for p in attr_inds:
            ## get the ll_param for p and gurobi variables
            ll_p = self._param_to_ll[p]
            if p.is_symbol():
                t_local = 0
            n_vals = 0
            grb_vars = []
            for attr, ind_arr in attr_inds[p]:
                n_vals += len(ind_arr)
                grb_vars.extend(list(getattr(ll_p, attr)[ind_arr, t_local]))

            for j, grb_var in enumerate(grb_vars):
                ## create an objective saying stay close to this value
                ## e(x) = x^2 - 2*val[i+j]*x + val[i+j]^2
                Q = np.eye(1)
                A = -2 * val[i + j] * np.ones((1, 1))
                b = np.ones((1, 1)) * np.power(val[i + j], 2)
                # QuadExpr is 0.5*x^Tx + Ax + b
                quad_expr = QuadExpr(
                    2 * Q * self.rs_coeff, A * self.rs_coeff, b * self.rs_coeff
                )
                v_arr = np.array([grb_var]).reshape((1, 1), order="F")
                init_val = np.ones((1, 1)) * val[i + j]
                bexpr = BoundExpr(
                    quad_expr, Variable(v_arr, val[i + j].reshape((1, 1)))
                )

                bexprs.append(bexpr)

            i += n_vals
        return bexprs

    def _add_pred_dict(
        self,
        pred_dict,
        effective_timesteps,
        add_nonlin=True,
        priority=MAX_PRIORITY,
        verbose=False,
    ):
        # verbose=True
        ## for debugging
        ignore_preds = []
        priority = np.maximum(priority, 0)
        if not pred_dict["hl_info"] == "hl_state":
            start, end = pred_dict["active_timesteps"]
            active_range = list(range(start, end + 1))
            if verbose:
                print("pred being added: ", pred_dict)
                print(active_range, effective_timesteps)
            negated = pred_dict["negated"]
            pred = pred_dict["pred"]

            if pred.get_type() in ignore_preds:
                return

            if pred.priority > priority:
                return
            assert isinstance(pred, common_predicates.ExprPredicate)
            expr = pred.get_expr(negated)

            for t in effective_timesteps:
                if t in active_range:
                    if expr is not None:
                        if add_nonlin or isinstance(expr.expr, AffExpr):
                            if verbose:
                                print("expr being added at time ", t)
                            var = self._spawn_sco_var_for_pred(pred, t)
                            bexpr = BoundExpr(expr, var)
                            self._bexpr_to_pred[bexpr] = (negated, pred, t)
                            groups = ["all"]
                            if self.early_converge:
                                ## this will check for convergence per parameter
                                ## this is good if e.g., a single trajectory quickly
                                ## gets stuck
                                groups.extend([param.name for param in pred.params])
                            self._prob.add_cnt_expr(bexpr, groups)

    def _add_first_and_last_timesteps_of_actions(
        self,
        plan,
        priority=MAX_PRIORITY,
        add_nonlin=False,
        active_ts=None,
        verbose=False,
    ):
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)
        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            ## only add an action
            if action_start >= active_ts[1] and action_start > active_ts[0]:
                continue
            if action_end < active_ts[0]:
                continue
            for pred_dict in action.preds:
                if action_start >= active_ts[0]:
                    self._add_pred_dict(
                        pred_dict,
                        [action_start],
                        priority=priority,
                        add_nonlin=add_nonlin,
                        verbose=verbose,
                    )
                if action_end <= active_ts[1]:
                    self._add_pred_dict(
                        pred_dict,
                        [action_end],
                        priority=priority,
                        add_nonlin=add_nonlin,
                        verbose=verbose,
                    )
            ## add all of the linear ineqs
            timesteps = list(
                range(
                    max(action_start + 1, active_ts[0]), min(action_end, active_ts[1])
                )
            )
            for pred_dict in action.preds:
                self._add_pred_dict(
                    pred_dict,
                    timesteps,
                    add_nonlin=False,
                    priority=priority,
                    verbose=verbose,
                )

    def _add_all_timesteps_of_actions(
        self,
        plan,
        priority=MAX_PRIORITY,
        add_nonlin=True,
        active_ts=None,
        verbose=False,
    ):
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)
        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            if action_start >= active_ts[1] and action_start > active_ts[0]:
                continue
            if action_end < active_ts[0]:
                continue

            timesteps = list(
                range(
                    max(action_start, active_ts[0]), min(action_end, active_ts[1]) + 1
                )
            )
            for pred_dict in action.preds:
                self._add_pred_dict(
                    pred_dict,
                    timesteps,
                    priority=priority,
                    add_nonlin=add_nonlin,
                    verbose=verbose,
                )

    def _update_ll_params(self):
        for ll_param in list(self._param_to_ll.values()):
            ll_param.update_param()
        if self.child_solver:
            self.child_solver._update_ll_params()

    def _spawn_parameter_to_ll_mapping(self, model, plan, active_ts=None):
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)
        horizon = active_ts[1] - active_ts[0] + 1
        self._param_to_ll = {}
        self.ll_start = active_ts[0]
        for param in list(plan.params.values()):
            ll_param = LLParamOSQP(model, param, horizon, active_ts)
            ll_param.create_grb_vars()
            self._param_to_ll[param] = ll_param
        model.update()
        for ll_param in list(self._param_to_ll.values()):
            ll_param.batch_add_cnts()

    def _add_obj_bexprs(self, obj_bexprs):
        for bexpr in obj_bexprs:
            self._prob.add_obj_expr(bexpr)

    def _get_trajopt_obj(self, plan, active_ts=None):
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)
        start, end = active_ts
        traj_objs = []
        for param in list(plan.params.values()):
            if param not in self._param_to_ll:
                continue
            if param._type in ["Robot", "Can"]:
                T = end - start + 1
                K = 2
                pose = param.pose
                KT = K * T
                v = -1 * np.ones((KT - K, 1))
                d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                # [:,0] allows numpy to see v and d as one-dimensional so
                # that numpy will create a diagonal matrix with v and d as a diagonal
                P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                Q = np.dot(np.transpose(P), P)

                Q *= TRAJOPT_COEFF

                quad_expr = QuadExpr(Q, np.zeros((1, KT)), np.zeros((1, 1)))
                robot_ll = self._param_to_ll[param]
                robot_ll_grb_vars = robot_ll.pose.reshape((KT, 1), order="F")
                init_val = param.pose[:, start : end + 1].reshape((KT, 1), order="F")
                bexpr = BoundExpr(quad_expr, Variable(robot_ll_grb_vars, init_val))
                traj_objs.append(bexpr)
        return traj_objs


class DummyLLSolverOSQP(LLSolverOSQP):
    def solve(self, plan):
        return "solve"
