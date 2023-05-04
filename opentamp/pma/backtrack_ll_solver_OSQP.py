import itertools
import random

import numpy as np

from sco_py.expr import AffExpr, BoundExpr, QuadExpr
from sco_py.sco_osqp.prob import Prob
from sco_py.sco_osqp.solver import Solver
from sco_py.sco_osqp.variable import Variable

from opentamp.core.internal_repr.parameter import Object
from opentamp.core.util_classes.matrix import Vector
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
from opentamp.core.util_classes.viewer import OpenRAVEViewer

from .ll_solver_OSQP import LLParamOSQP, LLSolverOSQP


MAX_PRIORITY = 3
BASE_MOVE_COEFF = 1.0
TRAJOPT_COEFF = 5e1
TRANSFER_COEFF = 1e-1
FIXED_COEFF = 1e3
INIT_TRAJ_COEFF = 1e-1
RS_COEFF = 1e2  # 1e2
COL_COEFF = 0
SAMPLE_SIZE = 5
BASE_SAMPLE_SIZE = 5
OSQP_EPS_ABS = 1e-06
OSQP_EPS_REL = 1e-09
OSQP_MAX_ITER = int(1e05)
OSQP_SIGMA = 1e-5
INIT_TRUST_REGION_SIZE = 1e-2
INIT_PENALTY_COEFF = 1e0
ADAPTIVE_RHO = True
MAX_MERIT_INCR = 5
RESAMPLE_ALL = False
DEBUG = True

# INIT_TRAJ_COEFF = 3e-1
# TRAJOPT_COEFF = 1e2
# RS_COEFF = 1e3
# # RESAMPLE_ALL = True
# OSQP_MAX_ITER = int(1e04)
# INIT_TRUST_REGION_SIZE = 1e-1


class BacktrackLLSolverOSQP(LLSolverOSQP):
    def __init__(self, 
                 early_converge=False, 
                 transfer_norm="min-vel",
                 trajopt_coeff=TRAJOPT_COEFF,
                 transfer_coeff=TRANSFER_COEFF,
                 fixed_coeff=FIXED_COEFF,
                 init_traj_coeff=INIT_TRAJ_COEFF,
                 rs_coeff=RS_COEFF,
                 col_coeff=COL_COEFF,
                 initial_trust_region_size=INIT_TRUST_REGION_SIZE,
                 initial_penalty_coeff=INIT_PENALTY_COEFF,
                 max_merit_coeff_increases=MAX_MERIT_INCR,
                 osqp_eps_abs=OSQP_EPS_ABS,
                 osqp_eps_rel=OSQP_EPS_REL,
                 osqp_max_iter=OSQP_MAX_ITER,
                 osqp_sigma=OSQP_SIGMA,
                 adaptive_rho=ADAPTIVE_RHO,
                 resample_all=RESAMPLE_ALL):
        # To avoid numerical difficulties during optimization, try to keep
        # range of coefficient within 1e9
        # (largest_coefficient/smallest_coefficient < 1e9)
        self.transfer_coeff = transfer_coeff
        self.fixed_coeff = fixed_coeff
        self.rs_coeff = rs_coeff
        self.trajopt_coeff = trajopt_coeff  # 1e-3#1e0
        self.init_traj_coeff = init_traj_coeff
        self.col_coeff = col_coeff
        self.initial_trust_region_size = initial_trust_region_size
        self.init_penalty_coeff = initial_penalty_coeff  # 4e3
        self.smooth_penalty_coeff = 1e2 # 1e0#7e4
        self.max_merit_coeff_increases = max_merit_coeff_increases
        self._param_to_ll = {}
        self.early_converge = early_converge
        self.child_solver = None
        self.solve_priorities = [-2, -1, 0, 1, 2, 3]
        self.transfer_norm = transfer_norm
        self.var_init_mapping = {}
        self.var_list = []
        self._osqpvar_to_scovar_ind = {}
        self.tol = 1e-3
        self.saved_params_free = {}
        self.fixed_objs = []
        self.osqp_eps_abs = osqp_eps_abs
        self.osqp_eps_rel = osqp_eps_rel
        self.osqp_max_iter = osqp_max_iter
        self.osqp_sigma = osqp_sigma
        self.adaptive_rho = adaptive_rho
        self.resample_all = resample_all

    def _solve_helper(self, plan, callback, active_ts, verbose):
        # certain constraints should be solved first
        success = False
        for priority in self.solve_priorities:
            success = self._solve_opt_prob(plan, priority=priority,
                            callback=callback, active_ts=active_ts,
                            verbose=verbose)

        return success

    def save_free(self, plan):
        old_params_free = {}
        for p in plan.params.values():
            p_attrs = {}
            old_params_free[p.name] = p_attrs
            for attr in p._free_attrs:
                p_attrs[attr] = p._free_attrs[attr].copy()
        self.saved_params_free[plan] = old_params_free

    def restore_free(self, plan):
        for p in self.saved_params_free[plan]:
            for attr in self.saved_params_free[plan][p]:
                plan.params[p]._free_attrs[attr] = self.saved_params_free[plan][p][
                    attr
                ].copy()

    def freeze_rs_param(self, act):
        return True

    # def get_resample_param(self, a):
    #     return a.params[0]

    def backtrack_solve(self, plan, callback=None, verbose=False, n_resamples=5):
        # populate belief_space stuff with an initial body of samples

        success = self._backtrack_solve(
            plan, callback, anum=0, verbose=verbose, n_resamples=n_resamples, 
        )
        return success

    def _backtrack_solve(
        self,
        plan,
        callback=None,
        anum=0,
        verbose=False,
        amax=None,
        n_resamples=5,
        init_traj=[],
        st=0,
        debug=False
    ):
        if amax is None:
            amax = len(plan.actions) - 1

        if anum > amax:
            return True
        a = plan.actions[anum]
        if DEBUG:
            print(("backtracking Solve on {}".format(a.name)))
        active_ts = a.active_timesteps
        active_ts = (max(st, active_ts[0]), active_ts[1])

        inits = {}
        rs_param = self.get_resample_param(a)
        success = False
        init_free_attrs = plan.get_free_attrs()
        for param in list(plan.params.values()):
            if param.is_symbol():
                continue
            for attr in param._free_attrs:
                param._free_attrs[attr][:, active_ts[0]] = 0.0

        def recursive_solve():
            ## don't optimize over any params that are already set
            old_params_free = {}
            old_params_free_2 = {}
            for p in plan.params.values():
                if p.is_symbol():
                    continue
                    # if p not in a.params: continue
                    old_params_free[p] = p._free_attrs
                    p._free_attrs = {}
                    for attr in list(old_params_free[p].keys()):
                        p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
                else:
                    p_attrs = {}
                    p_attrs_2 = {}
                    old_params_free[p] = p_attrs
                    old_params_free_2[p] = p_attrs_2
                    for attr in p._free_attrs:
                        p_attrs[attr] = p._free_attrs[attr][:, active_ts[1]].copy()
                        p_attrs_2[attr] = p._free_attrs[attr][:, active_ts[0]].copy()
                        p._free_attrs[attr][:, active_ts[1]] = 0
                        p._free_attrs[attr][:, active_ts[0]] = 0
            self.child_solver = self.__class__()

            success = self.child_solver._backtrack_solve(
                plan,
                callback=callback,
                anum=anum + 1,
                verbose=verbose,
                amax=amax,
                n_resamples=n_resamples,
                init_traj=init_traj,
                st=st,
            )

            # reset free_attrs
            for p in plan.params.values():
                if p.is_symbol():
                    continue
                    if p not in a.params:
                        continue
                    p._free_attrs = old_params_free[p]
                else:
                    for attr in p._free_attrs:
                        p._free_attrs[attr][:, active_ts[1]] = old_params_free[p][attr]
                        p._free_attrs[attr][:, active_ts[0]] = old_params_free_2[p][
                            attr
                        ]
            return success

        ### if there is no parameter to resample or some part of rs_param is fixed, then go ahead optimize over this action
        if (
            rs_param is None
        ):
            ## this parameter is fixed
            if callback is not None:
                callback_a = lambda: callback(a)
            else:
                callback_a = None
            self.child_solver = self.__class__()
            success = self.child_solver.solve(
                plan,
                callback=callback_a,
                n_resamples=n_resamples,
                active_ts=active_ts,
                verbose=verbose,
                force_init=True,
                init_traj=init_traj,
            )
            if not success:
                ## if planning fails we're done
                return False

            ## no other options, so just return here
            return recursive_solve()

        ### so that this won't be optimized over
        rs_params = rs_param if type(rs_param) is list else [rs_param]
        free_attrs = {}
        for param in rs_params:
            free = param._free_attrs
            if self.freeze_rs_param(plan.actions[anum]):
                param._free_attrs = {}
                for attr in list(free.keys()):
                    if param.is_symbol():
                        param._free_attrs[attr] = np.zeros(free[attr].shape)
                    else:
                        param._free_attrs[attr] = free[attr].copy()
                        param._free_attrs[attr][:, active_ts[1]] = np.zeros(
                            free[attr].shape[0]
                        )
            free_attrs[param] = free

        # sampler_begin

        robot_poses = self.obj_pose_suggester(plan, anum, resample_size=1, st=st)

        # sampler end
        
        if callback is not None:
            callback_a = lambda: callback(a)
        else:
            callback_a = None

        success = False
        for rp in robot_poses:
            if type(rs_param) is not list:
                rp = {rs_param: rp}

            self.child_solver = self.__class__()
            for param in rp:
                assert param in rs_params
                for attr, val in rp[param].items():
                    if param.is_symbol():
                        getattr(param, attr)[:, 0] = val.flatten()
                    else:
                        getattr(param, attr)[:, active_ts[1]] = val.flatten()
                
                if not self.freeze_rs_param(plan.actions[anum]):
                    self.child_solver.fixed_objs.append((param, rp[param]))
                success = self.child_solver.solve(plan, callback=callback_a, n_resamples=n_resamples,
                                              active_ts = active_ts, verbose=verbose,
                                              force_init=True, init_traj=init_traj)
                self.child_solver.fixed_objs = []

            # filter beliefs here
            # rs_params_current_act = [rsp for rsp in rs_params]
            if len(plan.belief_params) > 0:
                plan.filter_beliefs(rs_params)

            if success:
                if recursive_solve():
                    break
                else:
                    success = False

        for param in free_attrs:
            param._free_attrs = free_attrs[param]
        plan.store_free_attrs(init_free_attrs)
        return success

    def validate_wp(self, plan, callback=None, amin=0, amax=None, nsamples=1):
        if amax is None:
            amax = len(plan.actions) - 1
        n = 0
        success = False
        plan.save_free_attrs()
        failed_preds = []
        while not success and n < nsamples:
            success = True
            plan.restore_free_attrs()
            for a in range(amin, amax + 1):
                active_ts = plan.actions[a].active_timesteps
                robot_poses = self.obj_pose_suggester(plan, a, resample_size=1)
                ind = np.random.randint(len(robot_poses))
                rp = robot_poses[ind]
                for attr, val in list(rp.items()):
                    if rs_param.is_symbol():
                        setattr(rs_param, attr, val)
                    else:
                        getattr(rs_param, attr)[:, active_ts[1]] = val.flatten()
                failed_preds = plan.get_failed_preds(
                    active_ts=(active_ts[1], active_ts[1]), tol=1e-3
                )
                if len(failed_preds):
                    success = False
                    break
            n += 1

        plan.restore_free_attrs()
        return success, failed_preds

    # @profile
    def solve(
        self,
        plan,
        callback=None,
        n_resamples=5,
        active_ts=None,
        verbose=False,
        force_init=False,
        init_traj=[],
        debug=False,
    ):

        success = False
        if callback is not None:
            viewer = callback()

        highest_priority = -2
        for priority in self.solve_priorities:
            highest_priority = priority
            if DEBUG: print('solving at priority', priority)

            for attempt in range(max(1, n_resamples)):
                ## refinement loop
                success = self._solve_opt_prob(plan, priority=priority,
                                callback=callback, active_ts=active_ts, verbose=verbose,
                                init_traj=init_traj, debug=debug)
                # success = len(plan.get_failed_preds(active_ts=active_ts, tol=1e-3)) == 0

                # No point in resampling if the endpoints or linear constraints can't be satisfied
                if success or priority < 0 or n_resamples == 0:
                    break

                if DEBUG:
                    print("pre-resample attempt {} failed:".format(attempt))
                    print(plan.get_failed_preds(active_ts, priority=priority, tol=1e-3))

                success = self._solve_opt_prob(plan, priority=priority, callback=callback, 
                                               active_ts=active_ts, verbose=verbose, resample = True,
                                               init_traj=init_traj)

                if DEBUG:
                    print("resample attempt: {} at priority {}".format(attempt, priority))
                    print("FAILED PREDICATES:")
                    print(plan.get_failed_preds(active_ts, priority=priority, tol=1e-3))
                
                if success:
                    break

            if not success:
                break

        if DEBUG:
            print("FAILED PREDICATES")
            print((plan.get_failed_preds(active_ts=active_ts, priority=highest_priority, tol=1e-3), active_ts))

        self._cleanup_plan(plan, active_ts)

        return success

    # @profile
    def _solve_opt_prob(
        self,
        plan,
        priority,
        callback=None,
        init=True,
        active_ts=None,
        verbose=False,
        resample=False,
        smoothing=False,
        init_traj=[],
        debug=False,
    ):
        if callback is not None:
            viewer = callback()
        self.plan = plan
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)

        plan.save_free_attrs()  # Copies the current free_attrs
        self._prob = Prob(callback=callback)
        self._spawn_parameter_to_ll_mapping(plan, active_ts)

        obj_bexprs = []
        for param, values in self.fixed_objs:
            obj_bexprs.extend(
                self._get_fixed_obj(
                    param,
                    values,
                    "min-vel",
                    active_ts=(
                        active_ts[1] - active_ts[0],
                        active_ts[1] - active_ts[0],
                    ),
                )
            )

        if len(init_traj):
            obj_bexprs.extend(
                self._get_fixed_transfer_obj(
                    plan, "min-vel", init_traj, active_ts=active_ts
                )
            )

        initial_trust_region_size = self.initial_trust_region_size
        end_t = active_ts[1] - active_ts[0]

        if resample:
            tol = 1e-3
            """
            When Optimization fails, resample new values for certain timesteps
            of the trajectory and solver as initialization
            """
            ## this is an objective that places
            ## a high value on matching the resampled values
            # failed_preds = plan.get_failed_preds(active_ts = (active_ts[0]+1, active_ts[1]-1), priority=priority, tol = tol)
            # failed_preds = plan.get_failed_preds(active_ts = (active_ts[0], active_ts[1]-1), priority=priority, tol = tol)
            failed_preds = plan.get_failed_preds(
                active_ts=(active_ts[0], active_ts[1]), priority=priority, tol=tol
            )
            rs_obj = self._resample(plan, failed_preds)
            # _get_transfer_obj returns the expression saying the current trajectory should be close to it's previous trajectory.
            obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
            # obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))

            self._add_all_timesteps_of_actions(
                plan,
                priority=priority,
                add_nonlin=True,
                active_ts=active_ts,
                verbose=verbose,
            )
            obj_bexprs.extend(rs_obj)
            self._add_obj_bexprs(obj_bexprs)
            initial_trust_region_size = 1e3

        else:
            self._bexpr_to_pred = {}
            if self.col_coeff > 0 and priority >= 0:
                self._add_col_obj(plan, active_ts=active_ts)

            if priority == -2:
                """
                Initialize an linear trajectory while enforceing the linear constraints in the intermediate step.
                """
                obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
                self._add_obj_bexprs(obj_bexprs)
                self._add_first_and_last_timesteps_of_actions(
                    plan,
                    priority=MAX_PRIORITY,
                    active_ts=active_ts,
                    verbose=verbose,
                    add_nonlin=False,
                )
                tol = 1e-3
                initial_trust_region_size = 1e3

            elif priority == -1:
                """
                Solve the optimization problem while enforcing every constraints.
                """
                obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
                self._add_obj_bexprs(obj_bexprs)
                self._add_first_and_last_timesteps_of_actions(
                    plan,
                    priority=MAX_PRIORITY,
                    active_ts=active_ts,
                    verbose=verbose,
                    add_nonlin=True,
                )
                tol = 1e-3
            elif priority >= 0:
                obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
                self._add_obj_bexprs(obj_bexprs)
                self._add_all_timesteps_of_actions(
                    plan,
                    priority=priority,
                    add_nonlin=True,
                    active_ts=active_ts,
                    verbose=verbose,
                )
                tol = 1e-3

        solv = Solver()
        solv.initial_trust_region_size = initial_trust_region_size
        if smoothing:
            solv.initial_penalty_coeff = self.smooth_penalty_coeff
        else:
            solv.initial_penalty_coeff = self.init_penalty_coeff
        solv.max_merit_coeff_increases = self.max_merit_coeff_increases

        # Call the solver on this problem now that it's been constructed
        success = solv.solve(self._prob, method="penalty_sqp", tol=tol, verbose=verbose,\
            osqp_eps_abs=self.osqp_eps_abs, osqp_eps_rel=self.osqp_eps_rel,\
                osqp_max_iter=self.osqp_max_iter, sigma=self.osqp_sigma,
                adaptive_rho=self.adaptive_rho,)

        # Update the values of the variables by leveraging the ll_param mapping
        self._update_ll_params()
        if priority >= 0:
            failed_preds = plan.get_failed_preds(tol=tol, active_ts=active_ts, priority=priority)
            success = success and len(failed_preds) == 0

        """
        if resample:
            # During resampling phases, there must be changes added to sampling_trace
            if len(plan.sampling_trace) > 0 and 'reward' not in plan.sampling_trace[-1]:
                reward = 0
                if len(plan.get_failed_preds(active_ts = active_ts, priority=priority)) == 0:
                    reward = len(plan.actions)
                else:
                    failed_t = plan.get_failed_pred(active_ts=(0,active_ts[1]), priority=priority)[2]
                    for i in range(len(plan.actions)):
                        if failed_t > plan.actions[i].active_timesteps[1]:
                            reward += 1
                plan.sampling_trace[-1]['reward'] = reward
        """

        ##Restore free_attrs values
        plan.restore_free_attrs()
        self.reset_variable()

        return success

    # @profile
    def traj_smoother(self, plan, callback=None, n_resamples=5, verbose=False):
        # plan.save_free_attrs()
        a_num = 0
        success = True
        while a_num < len(plan.actions) - 1:
            act_1 = plan.actions[a_num]
            act_2 = plan.actions[a_num + 1]
            active_ts = (act_1.active_timesteps[0], act_2.active_timesteps[1])
            old_params_free = {}
            for p in plan.params.values():
                if p.is_symbol():
                    if p in act_1.params or p in act_2.params:
                        continue
                    old_params_free[p] = p._free_attrs
                    p._free_attrs = {}
                    for attr in list(old_params_free[p].keys()):
                        p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
                else:
                    p_attrs = {}
                    old_params_free[p] = p_attrs
                    for attr in p._free_attrs:
                        p_attrs[attr] = [
                            p._free_attrs[attr][:, : active_ts[0]].copy(),
                            p._free_attrs[attr][:, active_ts[1] :].copy(),
                        ]
                        p._free_attrs[attr][:, active_ts[1] :] = 0
                        p._free_attrs[attr][:, : active_ts[0]] = 0
            success = self._traj_smoother(
                plan, callback, n_resamples, active_ts, verbose
            )
            # reset free_attrs
            for p in plan.params.values():
                if p.is_symbol():
                    if p in act_1.params or p in act_2.params:
                        continue
                    p._free_attrs = old_params_free[p]
                else:
                    for attr in p._free_attrs:
                        p._free_attrs[attr][:, : active_ts[0]] = old_params_free[p][
                            attr
                        ][0]
                        p._free_attrs[attr][:, active_ts[1] :] = old_params_free[p][
                            attr
                        ][1]

            if not success:
                return success
            a_num += 1
        # try:
        #     success = self._traj_smoother(plan, callback, n_resamples, active_ts, verbose)
        # except:
        #     print "Error occured during planning, but not catched"
        #     return False
        # plan.restore_free_attrs()
        return success

    # @profile
    def _traj_smoother(
        self, plan, callback=None, n_resamples=5, active_ts=None, verbose=False
    ):
        priority = MAX_PRIORITY
        for attempt in range(n_resamples):
            # refinement loop
            success = self._solve_opt_prob(
                plan,
                priority=priority,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
                resample=False,
                smoothing=True,
            )
            if success or n_resamples == 0:
                break
            self._solve_opt_prob(
                plan,
                priority=priority,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
                resample=True,
                smoothing=True,
            )
        return success

    # @profile
    def _get_transfer_obj(self, plan, norm, coeff=None, active_ts=None):
        """
        This function returns the expression e(x) = P|x - cur|^2
        Which says the optimized trajectory should be close to the
        previous trajectory.
        Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
        """
        if coeff is None:
            coeff = self.transfer_coeff

        transfer_objs = []
        ts = active_ts
        if norm == "min-vel":
            for param in list(plan.params.values()):
                # if param._type in ['Robot', 'Can', 'EEPose']:
                for attr_name in param.__dict__.keys():
                    attr_type = param.get_attr_type(attr_name)
                    if issubclass(attr_type, Vector):
                        param_ll = self._param_to_ll[param]
                        if ts is None:
                            active_ts = 0, param_ll.active_ts[1] - param_ll.active_ts[0]
                        if param.is_symbol():
                            T = 1
                            attr_val = getattr(param, attr_name)
                        else:
                            T = active_ts[1] - active_ts[0] + 1  # param_ll._horizon
                            attr_val = getattr(param, attr_name)[
                                :,
                                param_ll.active_ts[0]
                                + active_ts[0] : param_ll.active_ts[0]
                                + active_ts[1]
                                + 1,
                            ]
                        if np.any(np.isnan(attr_val)):
                            continue
                        K = attr_type.dim

                        # pose = param.pose
                        if DEBUG:
                            assert (K, T) == attr_val.shape
                        KT = K * T
                        v = -1 * np.ones((KT - K, 1))
                        d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                        # [:,0] allows numpy to see v and d as one-dimensional so
                        # that numpy will create a diagonal matrix with v and d as a diagonal
                        P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                        P = np.eye(KT)
                        Q = (
                            np.dot(np.transpose(P), P)
                            if not param.is_symbol()
                            else np.eye(KT)
                        )
                        cur_val = attr_val.reshape((KT, 1), order="F")
                        A = -2 * cur_val.T.dot(Q)
                        b = cur_val.T.dot(Q.dot(cur_val))
                        transfer_coeff = coeff / float(plan.horizon)

                        # QuadExpr is 0.5*x^Tx + Ax + b
                        quad_expr = QuadExpr(
                            2 * transfer_coeff * Q,
                            transfer_coeff * A,
                            transfer_coeff * b,
                        )
                        ll_attr_val = getattr(param_ll, attr_name)
                        if not param.is_symbol():
                            ll_attr_val = ll_attr_val[
                                :, active_ts[0] : active_ts[1] + 1
                            ]
                        param_ll_osqp_vars = ll_attr_val.reshape((KT, 1), order="F")
                        sco_var = self.create_variable(param_ll_osqp_vars, cur_val)
                        bexpr = BoundExpr(quad_expr, sco_var)
                        transfer_objs.append(bexpr)
        else:
            raise NotImplemented
        return transfer_objs

    def create_variable(self, osqp_vars, init_vals, save=False):
        """
        if save is True
        Update the var_init_mapping so that each var_name is mapped to
        the right initial values.
        Then find the sco variables that includes the var names we are updating and change the corresponding initial values inside of it.
        if save is False
        Iterate the var_list and use the last initial value used for each var name, and construct the sco variables
        """
        sco_var, var_val_map, ret_val = None, {}, []

        for osqp_var, v in zip(osqp_vars.flatten(), init_vals.flatten()):
            if save:
                self.var_init_mapping[osqp_var] = v
            var_val_map[osqp_var] = self.var_init_mapping.get(osqp_var, v)
            ret_val.append(var_val_map[osqp_var])
            if osqp_var in list(self._osqpvar_to_scovar_ind.keys()):
                for var, i in self._osqpvar_to_scovar_ind[osqp_var]:
                    var._value[i] = var_val_map[osqp_var]
                    if np.all(var._osqp_vars is osqp_vars):
                        sco_var = var

        if sco_var is None:
            sco_var = Variable(osqp_vars, np.array(ret_val).reshape((len(ret_val), 1)))
            self.var_list.append(sco_var)
            for i, var_name in enumerate(osqp_vars.flatten()):
                index_val_list = self._osqpvar_to_scovar_ind.get(var_name, [])
                index_val_list.append((sco_var, i))
                self._osqpvar_to_scovar_ind[var_name] = index_val_list

        return sco_var

    def reset_variable(self):
        self.var_init_mapping = {}
        self.var_list = []
        self._osqpvar_to_scovar_ind = {}

    def _update(self, plan, update_values):
        bexprs = []
        for val, attr_inds in update_values:
            if val is not None:
                for p in attr_inds:
                    ## get the ll_param for p and gurobi variables
                    ll_p = self._param_to_ll[p]
                    n_vals, i = 0, 0
                    grb_vars = []
                    for attr, ind_arr, t in attr_inds[p]:
                        for j, grb_var in enumerate(
                            getattr(ll_p, attr)[
                                ind_arr, t - ll_p.active_ts[0]
                            ].flatten()
                        ):
                            Q = np.eye(1)
                            A = -2 * val[p][i + j] * np.ones((1, 1))
                            b = np.ones((1, 1)) * np.power(val[p][i + j], 2)
                            resample_coeff = self.rs_coeff / float(plan.horizon)
                            # QuadExpr is 0.5*x^Tx + Ax + b
                            quad_expr = QuadExpr(
                                2 * Q * resample_coeff,
                                A * resample_coeff,
                                b * resample_coeff,
                            )
                            v_arr = np.array([grb_var]).reshape((1, 1), order="F")
                            init_val = np.ones((1, 1)) * val[p][i + j]
                            sco_var = self.create_variable(
                                v_arr,
                                np.array([val[p][i + j]]).reshape((1, 1)),
                                save=True,
                            )
                            bexpr = BoundExpr(quad_expr, sco_var)
                            bexprs.append(bexpr)
                        i += len(ind_arr)
        return bexprs

    # @profile
    def _resample(self, plan, preds):
        """
        This function first calls fail predicate's resample function,
        then, uses the resampled value to create a square difference cost
        function e(x) = |x - rs_val|^2 that will be minimized later.
        rs_val is the resampled value
        """
        bexprs = []
        val, attr_inds = None, None
        pred_type = {}
        random.shuffle(preds)
        resample_coeff = self.rs_coeff / float(plan.horizon)
        for negated, pred, t in preds:
            ## returns a vector of new values and an
            ## attr_inds (OrderedDict) that gives the mapping
            ## to parameter attributes
            if pred_type.get(pred.get_type, False):
                continue
            val, attr_inds = pred.resample(negated, t, plan)
            if val is not None:
                pred_type[pred.get_type] = True
            ## if no resample defined for that pred, continue
            if val is not None:
                for p in attr_inds:
                    ## get the ll_param for p and gurobi variables
                    ll_p = self._param_to_ll[p]
                    n_vals, i = 0, 0
                    grb_vars = []
                    for attr, ind_arr, t in attr_inds[p]:
                        # if not np.all(p._free_attrs[attr][:,t]): continue
                        # if t >= ll_p.active_ts[1]: continue

                        if t > ll_p.active_ts[0] and t < ll_p.active_ts[1]:
                            ind_t = t - ll_p.active_ts[0]
                            attr_val = getattr(ll_p, attr)[ind_arr, ind_t].flatten()
                            n_inds = len(attr_val)
                            p_vals = np.array(val[p][i:i+n_inds]).reshape((-1, 1), order="F")

                            Q = np.eye(n_inds)
                            A = -2 * p_vals.T.dot(Q)
                            b = p_vals.T.dot(Q.dot(p_vals)) # np.sum(np.power(p_vals, 2))
                            quad_expr = QuadExpr(
                                2 * Q * resample_coeff,
                                A * resample_coeff,
                                b * resample_coeff,
                            )

                            v_arr = getattr(ll_p, attr)[ind_arr, ind_t]
                            sco_var = self.create_variable(
                                v_arr.reshape((-1, 1), order="F"),
                                p_vals.reshape((-1, 1), order="F"),
                                save=True,
                            )
                            bexpr = BoundExpr(quad_expr, sco_var)
                            bexprs.append(bexpr)

                            # for j, grb_var in enumerate(attr_val):
                            #     Q = np.eye(1)
                            #     A = -2 * val[p][i + j] * np.ones((1, 1))
                            #     b = np.ones((1, 1)) * np.power(val[p][i + j], 2)
                            #     # QuadExpr is 0.5*x^Tx + Ax + b
                            #     quad_expr = QuadExpr(
                            #         2 * Q * resample_coeff,
                            #         A * resample_coeff,
                            #         b * resample_coeff,
                            #     )
                            #     v_arr = np.array([grb_var]).reshape((1, 1), order="F")
                            #     init_val = np.ones((1, 1)) * val[p][i + j]
                            #     sco_var = self.create_variable(
                            #         v_arr,
                            #         np.array([val[p][i + j]]).reshape((1, 1)),
                            #         save=True,
                            #     )
                            #     bexpr = BoundExpr(quad_expr, sco_var)
                            #     bexprs.append(bexpr)
                        i += len(ind_arr)
                if not self.resample_all:
                    break
        return bexprs

    # @profile
    def _add_pred_dict(
        self,
        pred_dict,
        effective_timesteps,
        add_nonlin=True,
        priority=MAX_PRIORITY,
        verbose=False,
    ):
        """
        This function creates constraints for the predicate and adds to
        Prob class in sco.
        """
        ## for debugging
        ignore_preds = []
        
        original_priority = priority

        priority = np.maximum(priority, 0)

        if not pred_dict["hl_info"] == "hl_state":
            start, end = pred_dict["active_timesteps"]
            active_range = list(range(start, end + 1))
            negated = pred_dict["negated"]
            pred = pred_dict["pred"]

            if pred.get_type() in ignore_preds:
                return

            if pred.priority > priority:
                return
            expr = pred.get_expr(negated)

            if expr is not None:
                if add_nonlin or isinstance(expr.expr, AffExpr):
                    for t in effective_timesteps:
                        if t in active_range:
                            if t + pred.active_range[1] > effective_timesteps[-1]:
                                continue

                            if t + pred.active_range[0] < effective_timesteps[0]:
                                continue

                            var = self._spawn_sco_var_for_pred(pred, t)
                            bexpr = BoundExpr(expr, var)

                            # TODO: REMOVE line below, for tracing back predicate for debugging.
                            if DEBUG:
                                bexpr.source = (negated, pred, t)

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
        """
        Adding only non-linear constraints on the first and last timesteps of each action.
        """
        if active_ts is None:
            active_ts = (0, plan.horizon - 1)

        for action in plan.actions:
            true_start, true_end = action.active_timesteps
            action_start, action_end = action.active_timesteps
            ## only add an action
            if action_start >= active_ts[1]:
                continue

            if action_end <= active_ts[0]:
                continue

            if action_start < active_ts[0]:
                action_start = active_ts[0]

            if action_end > active_ts[1]:
                action_end = active_ts[1]

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
            timesteps = list(range(max(action_start, active_ts[0]),
                              min(action_end, active_ts[1])+1))

            for pred_dict in action.preds:
                self._add_pred_dict(
                    pred_dict,
                    timesteps,
                    add_nonlin=False,
                    priority=priority,
                    verbose=verbose,
                )

    # @profile
    def _add_all_timesteps_of_actions(
        self,
        plan,
        priority=MAX_PRIORITY,
        add_nonlin=True,
        active_ts=None,
        verbose=False,
    ):
        """
        This function adds both linear and non-linear predicates from
        actions that are active within the range of active_ts.
        """
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)

        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            if action_start >= active_ts[1]:
                continue

            if action_end <= active_ts[0]:
                continue

            if action_start < active_ts[0]:
                action_start = active_ts[0]

            if action_end > active_ts[1]:
                action_end = active_ts[1]

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

    # @profile
    def _update_ll_params(self):
        """
        update plan's parameters from low level OSQPVars.
        expected to be called after each optimization.
        """
        for ll_param in list(self._param_to_ll.values()):
            ll_param.update_param()

        # NOTE: this is null for the verify_namo_OSQP example, but needs
        # to be handled well if that's not the case!
        if self.child_solver:
            self.child_solver._update_ll_params()

    # @profile
    def _spawn_parameter_to_ll_mapping(self, plan, active_ts=None):
        """
        This function creates low level parameters for each parameter in the plan,
        initializes the corresponding grb_vars for each attributes in each timestep,
        updates the grb models
        adds in equality constraints,
        construct a dictionary as param-to-ll_param mapping.
        """
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)
        horizon = active_ts[1] - active_ts[0] + 1
        self._param_to_ll = {}
        self.ll_start = active_ts[0]
        for param in list(plan.params.values()):
            ll_param = LLParamOSQP(param, horizon, active_ts)
            ll_param.create_osqp_vars()
            self._param_to_ll[param] = ll_param
            # We need to add each new OSQPVar created in each new ll_param,
            # to the prob
            for attr in ll_param._num_attrs:
                for osqp_var in getattr(ll_param, attr).flatten().tolist():
                    self._prob.add_osqp_var(osqp_var)

        for ll_param in list(self._param_to_ll.values()):
            lin_eq_cnts = ll_param.batch_add_cnts()
            self._prob.batch_add_lin_cnts(lin_eq_cnts)

    # @profile
    def _add_obj_bexprs(self, obj_bexprs):
        """
        This function adds objective bounded expressions to the Prob class
        in sco.
        """
        for bexpr in obj_bexprs:
            self._prob.add_obj_expr(bexpr)

    def _get_trajopt_obj(self, plan, active_ts=None):
        """
        This function selects parameter of type Robot and Can and returns
        the expression e(x) = |Px|^2
        Which optimize trajectory so that robot and can's attributes in
        current timestep is close to that of next timestep.
        forming a straight line between each end points.

        Where P is the KT x KT matrix, where Px is the difference of
        value in current timestep compare to next timestep.
        """
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)
        start, end = active_ts
        traj_objs = []
        for param in list(plan.params.values()):
            if param not in self._param_to_ll:
                continue
            if isinstance(param, Object):
                for attr_name in param.__dict__.keys():
                    attr_type = param.get_attr_type(attr_name)
                    if issubclass(attr_type, Vector):
                        T = end - start + 1
                        K = attr_type.dim
                        attr_val = getattr(param, attr_name)
                        KT = K * T
                        v = -1 * np.ones((KT - K, 1))
                        d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                        # [:,0] allows numpy to see v and d as one-dimensional so
                        # that numpy will create a diagonal matrix with v and d as a diagonal

                        # v and d is computing the change in value between two timesteps
                        P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                        Q = np.dot(np.transpose(P), P)
                        Q *= self.trajopt_coeff / float(plan.horizon)

                        quad_expr = None
                        if attr_name == "pose" and param._type == "Robot":
                            quad_expr = QuadExpr(
                                BASE_MOVE_COEFF * Q, np.zeros((1, KT)), np.zeros((1, 1))
                            )
                        else:
                            quad_expr = QuadExpr(Q, np.zeros((1, KT)), np.zeros((1, 1)))

                        param_ll = self._param_to_ll[param]
                        ll_attr_val = getattr(param_ll, attr_name)
                        param_ll_vars = ll_attr_val.reshape((KT, 1), order="F")

                        attr_val = getattr(param, attr_name)
                        init_val = attr_val[:, start : end + 1].reshape(
                            (KT, 1), order="F"
                        )

                        sco_var = self.create_variable(param_ll_vars, init_val)
                        bexpr = BoundExpr(quad_expr, sco_var)
                        traj_objs.append(bexpr)

        return traj_objs

    def _get_fixed_obj(self, param, value_map, norm, coeff=None, active_ts=None):
        """
        This function returns the expression e(x) = P|x - cur|^2
        Which says the optimized value should be close to the
        provided value.
        Where P is the KT x KT matrix, where Px is the difference of
        parameter's attributes' current value and parameter's next
        timestep value
        """
        if coeff is None:
            coeff = self.fixed_coeff

        transfer_objs = []
        ts = active_ts
        if norm == "min-vel":
            for attr_name in value_map:  # param.__dict__.iterkeys():
                attr_type = param.get_attr_type(attr_name)
                if issubclass(attr_type, Vector):
                    param_ll = self._param_to_ll[param]
                    if ts is None:
                        active_ts = 0, param_ll.active_ts[1] - param_ll.active_ts[0]
                    if param.is_symbol():
                        T = 1
                        attr_val = value_map[attr_name]
                    else:
                        T = active_ts[1] - active_ts[0] + 1  # param_ll._horizon
                        attr_val = value_map[attr_name]
                    if np.any(np.isnan(attr_val)):
                        continue
                    K = attr_type.dim

                    if DEBUG:
                        assert (K, T) == attr_val.shape
                    KT = K * T
                    v = -1 * np.ones((KT - K, 1))
                    d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                    # [:,0] allows numpy to see v and d as one-dimensional so
                    # that numpy will create a diagonal matrix with v and d as a diagonal
                    P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                    P = np.eye(KT)
                    Q = (
                        np.dot(np.transpose(P), P)
                        if not param.is_symbol()
                        else np.eye(KT)
                    )
                    cur_val = attr_val.reshape((KT, 1), order="F")
                    A = -2 * cur_val.T.dot(Q)
                    b = cur_val.T.dot(Q.dot(cur_val))
                    transfer_coeff = coeff

                    # QuadExpr is 0.5*x^Tx + Ax + b
                    #quad_expr = QuadExpr(
                    #    2 * transfer_coeff * Q, transfer_coeff * A, transfer_coeff * b
                    #

                    quad_expr = QuadExpr(2 * transfer_coeff * Q, 
                                         transfer_coeff * A, 
                                         transfer_coeff * b)
                    ll_attr_val = getattr(param_ll, attr_name)
                    if not param.is_symbol():
                        ll_attr_val = ll_attr_val[:, active_ts[0] : active_ts[1] + 1]
                    param_ll_var_names = ll_attr_val.reshape((KT, 1), order="F")
                    sco_var = self.create_variable(param_ll_var_names, cur_val)
                    bexpr = BoundExpr(quad_expr, sco_var)
                    transfer_objs.append(bexpr)
        else:
            raise NotImplemented
        return transfer_objs

    def _get_fixed_transfer_obj(self, plan, norm, mean, coeff=None, active_ts=None):
        if active_ts is None:
            active_ts = (0, plan.horizon - 1)

        if not len(mean) or not hasattr(plan, "state_inds"):
            print("Cannot add fixed transfer; no data")
            return []

        if len(mean) < active_ts[1] - active_ts[0]:
            print(
                "Cannot add fixed transfer; insufficient data",
                len(mean),
                active_ts,
                plan.actions,
            )
            return []

        if coeff is None:
            coeff = self.init_traj_coeff

        transfer_objs = []
        if norm == "min-vel":
            for p_name, attr_name in plan.state_inds:
                param = plan.params[p_name]
                if param.is_symbol():
                    continue

                attr_type = param.get_attr_type(attr_name)
                param_ll = self._param_to_ll[param]
                K = attr_type.dim
                T = min(len(mean), param_ll._horizon)

                if len(mean) >= param_ll.active_ts[1]:
                    attr_val = mean[param_ll.active_ts[0] : param_ll.active_ts[1] + 1][
                        :, plan.state_inds[p_name, attr_name]
                    ]
                else:
                    attr_val = mean[-T:][:, plan.state_inds[p_name, attr_name]]

                T = len(attr_val)

                if DEBUG:
                    assert (K, T) == attr_val.shape
                KT = K * T
                v = -1 * np.ones((KT - K, 1))
                d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                Q = np.dot(np.transpose(P), P) if not param.is_symbol() else np.eye(KT)
                cur_val = attr_val.reshape((KT, 1), order="F")
                A = -2 * cur_val.T.dot(Q)
                b = cur_val.T.dot(Q.dot(cur_val))
                transfer_coeff = coeff / float(plan.horizon)

                quad_expr = QuadExpr(
                    2 * transfer_coeff * Q, transfer_coeff * A, transfer_coeff * b
                )
                ll_attr_val = getattr(param_ll, attr_name)[:, :T]
                param_ll_var_names = ll_attr_val.reshape((KT, 1), order="F")
                sco_var = self.create_variable(param_ll_var_names, cur_val)
                bexpr = BoundExpr(quad_expr, sco_var)
                transfer_objs.append(bexpr)

        return transfer_objs

    def _add_col_obj(self, plan, norm="min-vel", coeff=None, active_ts=None):
        return []

    def _cleanup_plan(self, plan, active_ts):
        return None
