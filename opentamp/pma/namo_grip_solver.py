import numpy as np

from sco_py.expr import BoundExpr, QuadExpr, AffExpr

from opentamp.pma import backtrack_ll_solver_OSQP as backtrack_ll_solver_OSQP
from opentamp.core.util_classes.namo_grip_predicates import (
    RETREAT_DIST,
    dsafe,
    opposite_angle,
    gripdist,
    ColObjPred,
    BoxObjPred,
)

try:
    from opentamp.pma import backtrack_ll_solver_gurobi as backtrack_ll_solver_gurobi
    HAS_GRB = True
except:
    HAS_GRB = False


def namo_obj_pose_suggester(plan, anum, resample_size=1, st=0):
    robot_pose = []
    assert anum + 1 <= len(plan.actions)

    if anum + 1 < len(plan.actions):
        act, next_act = plan.actions[anum], plan.actions[anum + 1]
    else:
        act, next_act = plan.actions[anum], None

    robot = plan.params["pr2"]
    robot_body = robot.openrave_body
    start_ts, end_ts = act.active_timesteps
    start_ts = max(st, start_ts)
    old_pose = robot.pose[:, start_ts].reshape((2, 1))
    robot_body.set_pose(old_pose[:, 0])
    oldx, oldy = old_pose.flatten()
    old_rot = robot.theta[0, start_ts]
    for i in range(resample_size):
        if next_act != None and (
            next_act.name == "grasp" or next_act.name == "putdown"
        ):
            target = next_act.params[2]
            target_pos = target.value - [[0], [0.0]]
            robot_pose.append(
                {
                    "value": target_pos,
                    "gripper": np.array([[-1.0]])
                    if next_act.name == "putdown"
                    else np.array([[1.0]]),
                }
            )
        elif (
            act.name == "moveto"
            or act.name == "new_quick_movetograsp"
            or act.name == "quick_moveto"
        ):
            target = act.params[2]
            grasp = act.params[5]
            target_rot = -np.arctan2(target.value[0,0] - oldx, target.value[1,0] - oldy)
            if target.value[1] > 1.7:
                target_rot = max(min(target_rot, np.pi/4), -np.pi/4)
            elif target.value[1] < -7.7 and np.abs(target_rot) < 3*np.pi/4:
                target_rot = np.sign(target_rot) * 3*np.pi/4

            while target_rot < old_rot:
                target_rot += 2 * np.pi
            while target_rot > old_rot:
                target_rot -= 2*np.pi
            if np.abs(target_rot-old_rot) > np.abs(target_rot-old_rot+2*np.pi): target_rot += 2*np.pi

            dist = gripdist + dsafe
            target_pos = target.value - [[-dist*np.sin(target_rot)], [dist*np.cos(target_rot)]]
            robot_pose.append({'pose': target_pos, 'gripper': np.array([[0.1]]), 'theta': np.array([[target_rot]])})
            # robot_pose.append({'pose': target_pos + grasp.value, 'gripper': np.array([[-1.]])})
        elif act.name == "transfer" or act.name == "new_quick_place_at":
            target = act.params[4]
            grasp = act.params[5]
            target_rot = -np.arctan2(target.value[0,0] - oldx, target.value[1,0] - oldy)
            if target.value[1] > 1.7:
                target_rot = max(min(target_rot, np.pi/4), -np.pi/4)
            elif target.value[1] < -7.7 and np.abs(target_rot) < 3*np.pi/4:
                target_rot = np.sign(target_rot) * 3*np.pi/4

            while target_rot < old_rot:
                target_rot += 2 * np.pi
            while target_rot > old_rot:
                target_rot -= 2 * np.pi
            if np.abs(target_rot - old_rot) > np.abs(
                target_rot - old_rot + 2 * np.pi
            ):
                target_rot += 2 * np.pi
            dist = -gripdist - dsafe
            target_pos = target.value + [[-dist*np.sin(target_rot)], [dist*np.cos(target_rot)]]
            robot_pose.append({'pose': target_pos, 'gripper': np.array([[-0.1]]), 'theta': np.array([[target_rot]])})
        elif act.name == 'place':
            target = act.params[4]
            grasp = act.params[5]
            target_rot = old_rot
            dist = -gripdist - dsafe - 1.0
            dist = -gripdist - dsafe - 1.4
            dist = -gripdist - dsafe - 1.7
            target_pos = target.value + [
                [-dist * np.sin(target_rot)],
                [dist * np.cos(target_rot)],
            ]
            robot_pose.append(
                {
                    "pose": target_pos,
                    "gripper": np.array([[-0.1]]),
                    "theta": np.array([[target_rot]]),
                }
            )
        elif act.name == 'move':
            can = act.params[1]
            can_pose = np.array([[can.pose[0,0]], [can.pose[1,0]]])
            can_rot = -np.arctan2(can.pose[0,0] - oldx, can.pose[1,0] - oldy)
            robot_pose.append(
                {
                    "pose": can_pose,
                    "theta": np.array([[can_rot]])
                }
            )
        else:
            raise NotImplementedError

    return robot_pose


def get_namo_col_obj(solver, plan, norm, mean, coeff=None, active_ts=None):
    """
    This function returns the expression e(x) = P|x - cur|^2
    Which says the optimized trajectory should be close to the
    previous trajectory.
    Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
    """
    if active_ts is None:
        active_ts = (0, plan.horizon - 1)

    start, end = active_ts
    if coeff is None:
        coeff = solver.transfer_coeff

    objs = []
    robot = plan.params["pr2"]
    ll_robot = solver._param_to_ll[robot]
    ll_robot_attr_val = getattr(ll_robot, "pose")
    robot_ll_grb_vars = ll_robot_attr_val.reshape((KT, 1), order="F")
    attr_robot_val = getattr(robot, "pose")
    init_robot_val = attr_val[:, start : end + 1].reshape((KT, 1), order="F")
    for robot in solver._robot_to_ll:
        param_ll = solver._param_to_ll[param]
        if param._type != "Can":
            continue
        attr_type = param.get_attr_type("pose")
        attr_val = getattr(param, "pose")
        init_val = attr_val[:, start : end + 1].reshape((KT, 1), order="F")
        K = attr_type.dim
        T = param_ll._horizon
        KT = K * T
        P = np.c_[np.eye(KT), -np.eye(KT)]
        Q = P.T.dot(P)
        quad_expr = QuadExpr(
            -2 * transfer_coeff * Q, np.zeros((KT)), np.zeros((1, 1))
        )
        ll_attr_val = getattr(param_ll, "pose")
        param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order="F")
        all_vars = np.r_[param_ll_grb_vars, robot_ll_grb_vars]
        sco_var = solver.create_variable(all_vars, np.r_[init_val, init_robot_val])
        bexpr = BoundExpr(quad_expr, sco_var)

    for p_name, attr_name in solver.state_inds:
        param = plan.params[p_name]
        if param.is_symbol():
            continue
        attr_type = param.get_attr_type(attr_name)
        param_ll = solver._param_to_ll[param]
        attr_val = mean[param_ll.active_ts[0] : param_ll.active_ts[1] + 1][
            :, solver.state_inds[p_name, attr_name]
        ]
        K = attr_type.dim
        T = param_ll._horizon

        if DEBUG:
            assert (K, T) == attr_val.shape
        KT = K * T
        v = -1 * np.ones((KT - K, 1))
        d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
        # [:,0] allows numpy to see v and d as one-dimensional so
        # that numpy will create a diagonal matrix with v and d as a diagonal
        P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
        # P = np.eye(KT)
        Q = np.dot(np.transpose(P), P) if not param.is_symbol() else np.eye(KT)
        cur_val = attr_val.reshape((KT, 1), order="F")
        A = -2 * cur_val.T.dot(Q)
        b = cur_val.T.dot(Q.dot(cur_val))
        transfer_coeff = coeff / float(plan.horizon)

        # QuadExpr is 0.5*x^Tx + Ax + b
        quad_expr = QuadExpr(
            2 * transfer_coeff * Q, transfer_coeff * A, transfer_coeff * b
        )
        ll_attr_val = getattr(param_ll, attr_name)
        param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order="F")
        sco_var = solver.create_variable(param_ll_grb_vars, cur_val)
        bexpr = BoundExpr(quad_expr, sco_var)
        transfer_objs.append(bexpr)


def add_namo_col_obj(solver, plan, norm="min-vel", coeff=None, active_ts=None):
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
    robot = plan.params["pr2"]
    act = [
        act
        for act in plan.actions
        if act.active_timesteps[0] <= start
        and act.active_timesteps[1] >= end
    ][0]
    for param in plan.params.values():
        if param._type not in ['Box', 'Can']: continue
        if param in act.params: continue
        if act.name.lower().find('transfer') >= 0 and param in act.params: continue
        if act.name.lower().find('move') >= 0 and param in act.params: continue
        expected_param_types = ['Robot', param._type]
        params = [robot, param]
        pred = ColObjPred('obstr', params, expected_param_types, plan.env, coeff=coeff)
        for t in range(start, end):
            # if act.name.lower().find('move') >= 0 \
            #    and param in act.params \
            #    and t >= act.active_timesteps[1]-4: continue

            var = solver._spawn_sco_var_for_pred(pred, t)
            bexpr = BoundExpr(pred.neg_expr.expr, var)
            objs.append(bexpr)
            solver._prob.add_obj_expr(bexpr)
    return objs


class NAMOSolverOSQP(backtrack_ll_solver_OSQP.BacktrackLLSolverOSQP):
    def get_resample_param(self, a):
        return a.params[0]

    def freeze_rs_param(self, act):
        return False

    def obj_pose_suggester(self, plan, anum, resample_size=1, st=0):
        return namo_obj_pose_suggester(plan, anum, resample_size, st)

    def _get_col_obj(self, plan, norm, mean, coeff=None, active_ts=None):
        """
        This function returns the expression e(x) = P|x - cur|^2
        Which says the optimized trajectory should be close to the
        previous trajectory.
        Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
        """
        return get_namo_col_obj(self, plan, norm, mean, coeff, active_ts)


    def _add_col_obj(self, plan, norm="min-vel", coeff=None, active_ts=None):
        """
        This function returns the expression e(x) = P|x - cur|^2
        Which says the optimized trajectory should be close to the
        previous trajectory.
        Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
        """
        return add_namo_col_obj(self, plan, norm, coeff, active_ts)


if HAS_GRB:
    class NAMOSolverGurobi(backtrack_ll_solver_gurobi.BacktrackLLSolverGurobi):
        def get_resample_param(self, a):
            return a.params[0]

        def freeze_rs_param(self, act):
            return False

        def obj_pose_suggester(self, plan, anum, resample_size=1, st=0):
            return namo_obj_pose_suggester(plan, anum, resample_size, st)

        def _get_col_obj(self, plan, norm, mean, coeff=None, active_ts=None):
            """
            This function returns the expression e(x) = P|x - cur|^2
            Which says the optimized trajectory should be close to the
            previous trajectory.
            Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
            """
            return get_namo_col_obj(self, plan, norm, mean, coeff, active_ts)


        def _add_col_obj(self, plan, norm="min-vel", coeff=None, active_ts=None):
            """
            This function returns the expression e(x) = P|x - cur|^2
            Which says the optimized trajectory should be close to the
            previous trajectory.
            Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
            """
            return add_namo_col_obj(self, plan, norm, coeff, active_ts)
