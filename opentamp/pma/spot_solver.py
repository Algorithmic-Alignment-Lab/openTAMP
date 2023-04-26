import numpy as np

from sco_py.expr import BoundExpr

from opentamp.pma import backtrack_ll_solver_OSQP as backtrack_ll_solver_OSQP
from opentamp.core.util_classes.robot_predicates import ColObjPred

try:
    from opentamp.pma import backtrack_ll_solver_gurobi as backtrack_ll_solver_gurobi
    HAS_GRB = True
except:
    HAS_GRB = False


def spot_pose_suggester(plan, anum, resample_size=1, st=0):
    robot_pose = []
    assert anum + 1 <= len(plan.actions)

    if anum + 1 < len(plan.actions):
        act, next_act = plan.actions[anum], plan.actions[anum + 1]
    else:
        act, next_act = plan.actions[anum], None

    robot = plan.params["spot"]
    robot_body = robot.openrave_body
    start_ts, end_ts = act.active_timesteps
    start_ts = max(st, start_ts)
    oldx = robot.x[:, start_ts].reshape((1, 1))
    oldy = robot.y[:, start_ts].reshape((1, 1))
    old_rot = robot.theta[0, start_ts]
    for i in range(resample_size):
        if act.name == "moveto":
            target = act.params[2]
            target_rot = -np.arctan2(target.value[0,0] - oldx, target.value[1,0] - oldy)

            while target_rot < old_rot:
                target_rot += 2 * np.pi
            while target_rot > old_rot:
                target_rot -= 2*np.pi
            if np.abs(target_rot-old_rot) > np.abs(target_rot-old_rot+2*np.pi): target_rot += 2*np.pi

            robot_pose.append({'x': target.x, 'y': target.y, 'theta': np.array([[target_rot]])})
       
        else:
            raise NotImplementedError

    return robot_pose



def add_col_obj(solver, plan, norm="min-vel", coeff=None, active_ts=None):
    """
    This function returns the expression e(x) = P|x - cur|^2
    Which says the optimized trajectory should be close to the
    previous trajectory.
    Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
    """
    if active_ts is None:
        active_ts = (0, plan.horizon - 1)

    start = max(active_ts[0], 1)
    end = active_ts[1]-1

    if coeff is None: coeff = solver.col_coeff
    if coeff == 0: return []

    objs = []
    robot = plan.params["spot"]
    act = [
        act
        for act in plan.actions
        if act.active_timesteps[0] <= start
        and act.active_timesteps[1] >= end
    ][0]
    for param in plan.params.values():
        if param._type not in ['Box', 'Can']: continue
        if param in act.params: continue
        if act.name.lower().find('move') >= 0 and param in act.params: continue
        expected_param_types = ['Robot', param._type]
        params = [robot, param]
        pred = ColObjPred('obstr', params, expected_param_types, plan.env, coeff=coeff)
        for t in range(start, end):
            var = solver._spawn_sco_var_for_pred(pred, t)
            bexpr = BoundExpr(pred.neg_expr.expr, var)
            objs.append(bexpr)
            solver._prob.add_obj_expr(bexpr)
    return objs


class SpotSolverOSQP(backtrack_ll_solver_OSQP.BacktrackLLSolverOSQP):
    def get_resample_param(self, a):
        return a.params[0]

    def freeze_rs_param(self, act):
        return False

    def obj_pose_suggester(self, plan, anum, resample_size=1, st=0):
        return spot_pose_suggester(plan, anum, resample_size, st)

    def _add_col_obj(self, plan, norm="min-vel", coeff=None, active_ts=None):
        """
        This function returns the expression e(x) = P|x - cur|^2
        Which says the optimized trajectory should be close to the
        previous trajectory.
        Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
        """
        objs = add_col_obj(self, plan, norm, coeff, active_ts)
        for bexpr in objs:
            self._prob.add_obj_expr(bexpr)
        return objs


if HAS_GRB:
    class SpotSolverGurobi(backtrack_ll_solver_gurobi.BacktrackLLSolverGurobi):
        def get_resample_param(self, a):
            return a.params[0]

        def freeze_rs_param(self, act):
            return False

        def obj_pose_suggester(self, plan, anum, resample_size=1, st=0):
            return spot_pose_suggester(plan, anum, resample_size, st)

        def _add_col_obj(self, plan, norm="min-vel", coeff=None, active_ts=None):
            """
            This function returns the expression e(x) = P|x - cur|^2
            Which says the optimized trajectory should be close to the
            previous trajectory.
            Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
            """
            objs = add_col_obj(self, plan, norm, coeff, active_ts)
            for bexpr in objs:
                self._prob.add_obj_expr(bexpr)
            return objs
