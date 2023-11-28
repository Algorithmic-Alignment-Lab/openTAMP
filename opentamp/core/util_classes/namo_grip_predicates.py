from opentamp.core.internal_repr.predicate import Predicate
from opentamp.core.internal_repr.plan import Plan
from opentamp.core.util_classes.common_predicates import ExprPredicate
from opentamp.core.util_classes.namo_predicates import NEAR_TOL
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
from opentamp.core.util_classes.torch_funcs import GaussianBump, ThetaDir
from opentamp.errors_exceptions import PredicateException

from sco_py.expr import Expr, AffExpr, EqExpr, LEqExpr

from collections import OrderedDict
import numpy as np
import os
import pybullet as P
import sys
import time
import traceback


"""
This file implements the predicates for the 2D NAMO domain.
"""

dsafe = 1e-3
# dmove = 1.1e0 # 5e-1
dmove = 1.8e0  # 5e-1
contact_dist = 2e-1  # dsafe
gripdist = 0.61  # 75

RS_SCALE = 0.5
N_DIGS = 5
GRIP_VAL = 0.1
COL_TS = 5  # 3
N_COLS = 8
RETREAT_DIST = 1.2

ATTRMAP = {
    "Robot": (
        ("pose", np.array(list(range(2)), dtype=np.int)),
        ("gripper", np.array(list(range(1)), dtype=np.int)),
        ("theta", np.array(list(range(1)), dtype=np.int)),
        ("vel", np.array(list(range(1)), dtype=np.int)),
        ("acc", np.array(list(range(1)), dtype=np.int)),
    ),
    "Can": (("pose", np.array(list(range(2)), dtype=np.int)),),
    "Target": (("value", np.array(list(range(2)), dtype=np.int)),),
    "RobotPose": (
        ("value", np.array(list(range(2)), dtype=np.int)),
        ("theta", np.array(list(range(1)), dtype=np.int)),
        ("gripper", np.array(list(range(1)), dtype=np.int)),
        ("vel", np.array(list(range(1)), dtype=np.int)),
        ("acc", np.array(list(range(1)), dtype=np.int)),
    ),
    "Obstacle": (("pose", np.array(list(range(2)), dtype=np.int)),),
    "Grasp": (("value", np.array(list(range(2)), dtype=np.int)),),
    "Rotation": (("value", np.array(list(range(1)), dtype=np.int)),),
}

HANDLE_OFFSET = 0.8



def add_to_attr_inds_and_res(t, attr_inds, res, param, attr_name_val_tuples):
    if param.is_symbol():
        t = 0
    for attr_name, val in attr_name_val_tuples:
        inds = np.where(param._free_attrs[attr_name][:, t])[0]
        getattr(param, attr_name)[inds, t] = val[inds]
        if param in attr_inds:
            res[param].extend(val[inds].flatten().tolist())
            attr_inds[param].append((attr_name, inds, t))
        else:
            res[param] = val[inds].flatten().tolist()
            attr_inds[param] = [(attr_name, inds, t)]


def process_traj(raw_traj, timesteps):
    """
    Process raw_trajectory so that it's length is desired timesteps
    when len(raw_traj) > timesteps
        sample Trajectory by space to reduce trajectory size
    when len(raw_traj) < timesteps
        append last timestep pose util the size fits

    Note: result_traj includes init_dof and end_dof
    """
    result_traj = []
    if len(raw_traj) == timesteps:
        result_traj = raw_traj.copy()
    else:
        traj_arr = [0]
        result_traj.append(raw_traj[0])
        # calculate accumulative distance
        for i in range(len(raw_traj) - 1):
            traj_arr.append(
                traj_arr[-1] + np.linalg.norm(raw_traj[i + 1] - raw_traj[i])
            )
        step_dist = traj_arr[-1] / (timesteps - 1)
        process_dist, i = 0, 1
        while i < len(traj_arr) - 1:
            if traj_arr[i] == process_dist + step_dist:
                result_traj.append(raw_traj[i])
                process_dist += step_dist
            elif traj_arr[i] < process_dist + step_dist < traj_arr[i + 1]:
                dist = process_dist + step_dist - traj_arr[i]
                displacement = (
                    (raw_traj[i + 1] - raw_traj[i])
                    / (traj_arr[i + 1] - traj_arr[i])
                    * dist
                )
                result_traj.append(raw_traj[i] + displacement)
                process_dist += step_dist
            else:
                i += 1
    result_traj.append(raw_traj[-1])
    return np.array(result_traj).T


def get_rrt_traj(env, robot, active_dof, init_dof, end_dof):
    # assert body in env.GetRobot()
    active_dofs = robot.GetActiveDOFIndices()
    robot.SetActiveDOFs(active_dof)
    robot.SetActiveDOFValues(init_dof)

    params = Planner.PlannerParameters()
    params.SetRobotActiveJoints(robot)
    params.SetGoalConfig(end_dof)  # set goal to all ones
    # # forces parabolic planning with 40 iterations
    # import ipdb; ipdb.set_trace()
    params.SetExtraParameters(
        """<_postprocessing planner="parabolicsmoother">
        <_nmaxiterations>20</_nmaxiterations>
    </_postprocessing>"""
    )

    planner = RaveCreatePlanner(env, "birrt")
    planner.InitPlan(robot, params)

    traj = RaveCreateTrajectory(env, "")
    result = planner.PlanPath(traj)
    if result == False:
        robot.SetActiveDOFs(active_dofs)
        return None
    traj_list = []
    for i in range(traj.GetNumWaypoints()):
        # get the waypoint values, this holds velocites, time stamps, etc
        data = traj.GetWaypoint(i)
        # extract the robot joint values only
        dofvalues = traj.GetConfigurationSpecification().ExtractJointValues(
            data, robot, robot.GetActiveDOFIndices()
        )
        # raveLogInfo('waypint %d is %s'%(i,np.round(dofvalues, 3)))
        traj_list.append(np.round(dofvalues, 3))
    robot.SetActiveDOFs(active_dofs)
    return np.array(traj_list)


def get_ompl_rrtconnect_traj(env, robot, active_dof, init_dof, end_dof):
    # assert body in env.GetRobot()
    dof_inds = robot.GetActiveDOFIndices()
    robot.SetActiveDOFs(active_dof)
    robot.SetActiveDOFValues(init_dof)

    params = Planner.PlannerParameters()
    params.SetRobotActiveJoints(robot)
    params.SetGoalConfig(end_dof)  # set goal to all ones
    # forces parabolic planning with 40 iterations
    planner = RaveCreatePlanner(env, "OMPL_RRTConnect")
    planner.InitPlan(robot, params)
    traj = RaveCreateTrajectory(env, "")
    planner.PlanPath(traj)

    traj_list = []
    for i in range(traj.GetNumWaypoints()):
        # get the waypoint values, this holds velocites, time stamps, etc
        data = traj.GetWaypoint(i)
        # extract the robot joint values only
        dofvalues = traj.GetConfigurationSpecification().ExtractJointValues(
            data, robot, robot.GetActiveDOFIndices()
        )
        # raveLogInfo('waypint %d is %s'%(i,np.round(dofvalues, 3)))
        traj_list.append(np.round(dofvalues, 3))
    robot.SetActiveDOFs(dof_inds)
    return traj_list


def opposite_angle(theta):
    return ((theta + 2 * np.pi) % (2 * np.pi)) - np.pi


def angle_diff(theta1, theta2):
    diff1 = theta1 - theta2
    diff2 = opposite_angle(theta1) - opposite_angle(theta2)
    if np.abs(diff1) < np.abs(diff2):
        return diff1
    return diff2


def add_angle(theta, delta):
    return ((theta + np.pi + delta) % (2 * np.pi)) - np.pi


def twostep_f(xs, dist, dim, pts=COL_TS, grad=False, isrobot=False):
    if grad:
        res = []
        jac = np.zeros((0, 2 * dim))
        for t in range(pts):
            coeff = float((pts - 1) - t) / (pts - 1)
            if len(xs) == 2:
                next_pos = coeff * xs[0] + (1 - coeff) * xs[1]
                if isrobot:
                    next_pos[2] = -GRIP_VAL  # min(xs[0][2], xs[1][2])
                    # next_pos[3] = np.arctan2(next_pos[0], next_pos[1])
            else:
                next_pos = xs[0]
            cur_jac = dist(next_pos)[1]
            filldim = dim - cur_jac.shape[1]
            # cur_jac = np.c_[cur_jac[:,:2], np.zeros((N_COLS, filldim)), cur_jac[:,2:]]
            # res.append(dist(next_pos)[1])
            if filldim > 0:
                cur_jac = np.c_[cur_jac, np.zeros((len(cur_jac), filldim))]
            jac = np.r_[jac, np.c_[coeff * cur_jac, (1 - coeff) * cur_jac]]
            # jac = np.r_[jac, np.c_[cur_jac, cur_jac]]
        return jac

    else:
        res = []
        for t in range(pts):
            coeff = float((pts - 1) - t) / (pts - 1)
            if len(xs) == 2:
                next_pos = coeff * xs[0] + (1 - coeff) * xs[1]
                if isrobot:
                    next_pos[2] = -GRIP_VAL  # min(xs[0][2], xs[1][2])
                    # next_pos[3] = np.arctan2(next_pos[0], next_pos[1])
            else:
                next_pos = xs[0]
            res.append(dist(next_pos)[0])
        return np.concatenate(res, axis=0)


## ensures belief_pose is constant from first step to last
## is an *optimistic* constraint, e.g. expected to not hold under replanned observations
class ConstantObservation(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## ConstantObservation Target
        self.target = params[0]
        attr_inds = OrderedDict(
            [
                (self.target, [("pose", np.array([0, 1], dtype=np.int))])
            ]
        )

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(ConstantObservation, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0, 1), priority=0)


class CollisionPredicate(ExprPredicate):
    def __init__(
        self,
        name,
        e,
        attr_inds,
        params,
        expected_param_types,
        dsafe=dsafe,
        debug=False,
        ind0=0,
        ind1=1,
        active_range=(0, 1),
        priority=3,
    ):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1

        self._cache = {}
        self.n_cols = N_COLS
        self.check_aabb = False

        super(CollisionPredicate, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=active_range, priority=priority)
        self._init_include = False

    def test(self, time, negated=False, tol=1e-4):
        # This test is overwritten so that collisions can be calculated correctly
        if not self.is_concrete():
            return False
        if time < 0:
            traceback.print_exception(*sys.exc_info())
            raise PredicateException("Out of range time for predicate '%s'." % self)
        try:
            result = self.neg_expr.eval(
                self.get_param_vector(time), tol=tol, negated=(not negated)
            )
            return result
        except IndexError:
            traceback.print_exception(*sys.exc_info())
            ## this happens with an invalid time
            raise PredicateException("Out of range time for predicate '%s'." % self)

    def plot_cols(self, env, t):
        _debug = self._debug
        self._env = env
        self._debug = True
        self.distance_from_obj(self.get_param_vector(t))
        self._debug = _debug

    # @profile
    def _set_robot_pos(self, x):
        flattened = tuple(x.round(N_DIGS).flatten())
        p0 = self.params[self.ind0]
        p1 = self.params[self.ind1]
        b0 = self._param_to_body[p0]
        b1 = self._param_to_body[p1]
        if b0.isrobot():
            robot = b0
            obj = b1
        # elif b1.isrobot():
        #    robot = b1
        #    obj = b0
        else:
            raise Exception("Should not call this without the robot!")
        pose0 = x[0:2]
        pose1 = x[4:6]
        b0.set_dof(
            {
                "left_grip": x[2],
                "right_grip": x[2],
                "robot_theta": x[3],
                "ypos": 0.0,
                "xpos": 0.0,
            }
        )
        b0.set_pose(pose0)
        b1.set_pose(pose1)
        if "door" in b1._geom.get_types():
            b1.set_dof({"door_hinge": x[6]})
        return pose0, pose1

    def set_pos(self, x):
        return self._set_pos(x)

    def _set_pos(self, x):
        flattened = tuple(x.round(N_DIGS).flatten())
        # if flattened in self._cache and self._debug is False:
        #     return self._cache[flattened]
        p0 = self.params[self.ind0]
        p1 = self.params[self.ind1]
        b0 = self._param_to_body[p0]
        b1 = self._param_to_body[p1]
        if b0.isrobot() or b1.isrobot():
            return self._set_robot_pos(x)
        pose0 = x[0:2]
        pose1 = x[2:4]
        b0.set_pose(pose0)
        b1.set_pose(pose1)
        return pose0, pose1

    def _check_robot_aabb(self, b0, b1):
        vals = np.zeros((self.n_cols, 1))
        jacs = np.zeros((self.n_cols, 4))
        (x1, y1, z1), (x2, y2, z2) = P.getAABB(b0.body_id, 5)
        (x3, y3, z3), (x4, y4, z4) = P.getAABB(b0.body_id, 7)
        (x5, y5, z5), (x6, y6, z6) = P.getAABB(b0.body_id, 3)
        grip_aabb = [
            (min(x1, x3, x5), min(y1, y3, y5), min(z1, z3, z5)),
            (max(x4, x2, x6), max(y4, y2, y6), max(z4, z2, z6)),
        ]
        minpt, maxpt = grip_aabb
        overlaps = P.getOverlappingObjects(grip_aabb[0], grip_aabb[1])
        if overlaps is not None and len(overlaps):
            ind = 0
            for body_id, link in overlaps:
                if body_id != b1.body_id:
                    continue
                cur_minpt, cur_maxpt = P.getAABB(body_id, link)
                d1, d2 = cur_minpt[0] - maxpt[0], minpt[0] - cur_maxpt[0]
                d3, d4 = cur_minpt[1] - maxpt[1], minpt[1] - cur_maxpt[1]
                if (
                    d1 <= self.dsafe
                    and d2 <= self.dsafe
                    and d3 <= self.dsafe
                    and d4 <= self.dsafe
                ):
                    xd = max(d1, d2)
                    yd = max(d3, d4)
                    if xd > yd:
                        vals[ind] = self.dsafe - xd
                        if d1 < d2:
                            jacs[ind, 0] = -1
                            jacs[ind, 2] = 1
                        else:
                            jacs[ind, 0] = 1
                            jacs[ind, 2] = -1
                    else:
                        vals[ind] = self.dsafe - yd
                        if d3 < d4:
                            jacs[ind, 1] = -1
                            jacs[ind, 3] = 1
                        else:
                            jacs[ind, 1] = 1
                            jacs[ind, 3] = -1
                ind += 1
        return vals, jacs

    def distance_from_obj(self, x, n_steps=0):
        pose0, pose1 = self.set_pos(x)
        p0 = self.params[self.ind0]
        p1 = self.params[self.ind1]
        b0 = self._param_to_body[p0]
        b1 = self._param_to_body[p1]
        vals = np.zeros((self.n_cols, 1))
        jacs = np.zeros((self.n_cols, 4))
        # if self.check_aabb:
        #    vals, jacs = self._check_robot_aabb(b0, b1)

        collisions = P.getClosestPoints(b0.body_id, b1.body_id, contact_dist)

        col_val, jac01 = self._calc_grad_and_val(
            p0.name, p1.name, pose0, pose1, collisions
        )
        final_val = col_val
        final_jac = jac01
        for i in range(len(final_val)):
            if final_val[i] < vals[i]:
                final_val[i] = vals[i]
                final_jac[i] = jacs[i]
        # self._cache[flattened] = (val.copy(), jac.copy())
        if b0.isrobot():
            if len(collisions):
                pose0, pose1 = np.r_[pose0, [[0]]], np.r_[pose1, [[0]]]
                colvec = np.array([c[5] for c in collisions])
                axisvec = np.array([[0, 0, 1] for _ in collisions])
                pos0vec = np.array([pose0.flatten() for _ in collisions])
                crosstorque = np.cross(colvec - pos0vec, [0, 0, 1])
                rotjac = np.dot(crosstorque, pose1 - pose0)
                rotjac = (
                    0 * np.r_[rotjac, np.zeros((len(final_jac) - len(collisions), 1))]
                )
            else:
                rotjac = np.zeros((final_jac.shape[0], 1))
            final_jac = np.c_[
                final_jac[:, :2], np.zeros_like(rotjac), rotjac, final_jac[:, 2:]
            ]
        return final_val, final_jac

    def _calc_rot_grad(self, rpose, objpose, colpos):
        jntaxis = np.array([0, 0, 1])
        return np.dot(objpose - rpose, np.cross(colpos - rpos, jntaxis))

    # @profile
    def _calc_grad_and_val(self, name0, name1, pose0, pose1, collisions):
        vals = np.zeros((self.n_cols, 1))
        jacs = np.zeros((self.n_cols, 4))

        val = -1 * float("inf")
        results = []
        n_cols = len(collisions)
        assert n_cols <= self.n_cols
        jac = np.zeros((1, 4))

        p0 = next(filter(lambda p: p.name == name0, list(self._param_to_body.keys())))
        p1 = next(filter(lambda p: p.name == name1, list(self._param_to_body.keys())))

        b0 = self._param_to_body[p0]
        b1 = self._param_to_body[p1]
        for i, c in enumerate(collisions):
            linkA, linkB = c[3], c[4]
            linkAParent, linkBParent = c[1], c[2]
            sign = 0
            if linkAParent == b0.body_id and linkBParent == b1.body_id:
                pt0, pt1 = c[5], c[6]
                linkRobot, linkObj = linkA, linkB
                sign = -1
            elif linkBParent == b0.body_id and linkAParent == b1.body_id:
                pt1, pt0 = c[5], c[6]
                linkRobot, linkObj = linkB, linkA
                sign = 1
            else:
                continue

            distance = c[8]  # c.contactDistance
            normal = np.array(c[7])  # c.contactNormalOnB # Pointing towards A
            results.append((pt0, pt1, distance))

            if self._debug:
                # self._plot_collision(pt0, pt1, distance)
                # print("pt0 = ", pt0)
                # print("pt1 = ", pt1)
                # print("distance = ", distance)
                # print("normal = ", normal)
                self._plot_collision_normal(pt0, pt1, distance, normal)

            vals[i, 0] = self.dsafe - distance
            jacs[i, :2] = -1 * normal[:2]
            jacs[i, 2:] = normal[:2]

        return np.array(vals).reshape((self.n_cols, 1)), np.array(jacs).reshape((self.n_cols, 4))

    def _plot_collision(self, ptA, ptB, distance):
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                # Red because collision
                rgb = (1, 0, 0)
            else:
                # Green because no collision
                rgb = (0, 1, 0)
            P.addUserDebugLine(ptA, ptB, rgb, 0.05)

    def _plot_collision_normal(self, ptA, ptB, distance, normal):
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                # Plot red arrow because collision
                P.addUserDebugLine(ptA, ptA + normal, (1, 0, 0), 0.01, 0.5)


class HLPoseUsed(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Can Target
        self.pose = params[0]
        if self.pose.is_symbol():
            k = "value"
        else:
            k = "pose"
        attr_inds = OrderedDict([(self.pose, [(k, np.array([0, 1], dtype=np.int))])])

        A = np.zeros((2, 2))
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(HLPoseUsed, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.hl_info = True

    def test(self, time, negated=False, tol=1e-4):
        if negated:
            return True
        return super(HLPoseUsed, self).test(time, tol=tol)


class HLGraspFailed(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.pose = params[0]
        if self.pose.is_symbol():
            k = "value"
        else:
            k = "pose"
        attr_inds = OrderedDict([(self.pose, [(k, np.array([0, 1], dtype=np.int))])])

        A = np.zeros((2, 2))
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(HLGraspFailed, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.hl_info = True

    def test(self, time, negated=False, tol=1e-4):
        return True


class HLTransferFailed(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.pose = params[0]
        if self.pose.is_symbol():
            k = "value"
        else:
            k = "pose"
        attr_inds = OrderedDict([(self.pose, [(k, np.array([0, 1], dtype=np.int))])])

        A = np.zeros((2, 2))
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(HLTransferFailed, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.hl_info = True

    def test(self, time, negated=False, tol=1e-4):
        return True


class HLPlaceFailed(HLTransferFailed):
    pass


class HLPoseAtGrasp(HLPoseUsed):

    # RobotAt Robot Can Grasp

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        self.r, self.c, self.g = params
        k = "pose" if not self.r.is_symbol() else "value"
        attr_inds = OrderedDict(
            [
                (self.r, [(k, np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[
            np.r_[np.eye(2), -np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
        ]
        b = np.zeros((4, 1))
        val = NEAR_TOL * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(HLPoseUsed, self).__init__(
            name, e, attr_inds, params, expected_param_types
        )
        self.hl_info = True


class HLAtGrasp(HLPoseUsed):
    pass


class HLPoseAtGrasp(HLPoseUsed):
    pass


class At(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Can Target
        self.can, self.targ = params
        attr_inds = OrderedDict(
            [
                (self.can, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.targ, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        self.coeff = 1e-2
        A = self.coeff * np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self._init_include = False


class AtNEq(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Can Target
        self.can, self.eq, self.targ = params
        attr_inds = OrderedDict(
            [
                (self.can, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.targ, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        if self.can is not self.eq:
            A = np.c_[np.eye(2), -np.eye(2)]
            b = np.zeros((2, 1))
            val = np.zeros((2, 1))
        else:
            A = np.zeros((2, 4))
            b = np.ones((2, 1))
            val = np.zeros((2, 1))

        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(AtNEq, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )


class AtInit(At):
    def test(self, time, negated=False, tol=1e-4):
        return True

    def hl_test(self, time, negated=False, tol=1e-4):
        return True


class RobotAt(At):

    # RobotAt Robot RobotPose

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        self.r, self.rp = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.rp, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)

class RobotAtTarget(At):

    # RobotAt Robot Targ

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot Targ
        self.r, self.rp = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.rp, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)



class RobotAtRot(At):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        self.r, self.rot = params
        attr_inds = OrderedDict(
            [
                (self.r, [("theta", np.array([0], dtype=np.int))]),
                (self.rot, [("value", np.array([0], dtype=np.int))]),
            ]
        )

        A = np.c_[np.eye(1), -np.eye(1)]
        b = np.zeros((1, 1))
        val = np.zeros((1, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class BoxAt(At):
    pass


class Near(At):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        self.r, self.c = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.c, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)]]
        b = np.zeros((4, 1))
        val = NEAR_TOL * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)
        self._init_include = False


class RobotNearTarget(At):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        self.r, self.t = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.t, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)]]
        b = np.zeros((4, 1))
        val = 0.25 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotNear(At):

    # RobotAt Robot Can

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        self.r, self.c = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)]]
        b = np.zeros((4, 1))
        val = 2 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class NotRobotNear(At):

    # RobotAt Robot Can

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        self.r, self.c = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)]]
        b = np.zeros((4, 1))
        val = -2 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class AtCloset(At):

    # RobotAt Robot RobotPose

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        (self.r,) = params
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                )
            ]
        )

        A = np.eye(3)
        b = np.array([[0.0], [-2.2], [0.0]])
        val = np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class OpenDoorApproach(At):

    # RobotAt Robot RobotPose

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        (self.r,) = params
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                )
            ]
        )

        A = np.eye(3)
        dist = gripdist + dsafe
        b = np.array([[-0.5 - dist - 1.2], [-((3 - HANDLE_OFFSET))], [-np.pi / 2.0]])
        val = np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class OpenDoorReady(At):

    # RobotAt Robot RobotPose

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        (self.r,) = params
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                )
            ]
        )

        A = np.eye(3)
        dist = gripdist + dsafe
        b = np.array([[-0.5 - dist], [-((3 - HANDLE_OFFSET))], [-np.pi / 2.0]])
        val = np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class CloseDoorApproach(At):

    # RobotAt Robot RobotPose

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        (self.r,) = params
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                )
            ]
        )

        A = np.c_[np.eye(3)]
        dist = gripdist + dsafe
        b = np.array([[1 + HANDLE_OFFSET], [-1.5 + dist + 1.2], [0]])
        val = np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class CloseDoorReady(At):

    # RobotAt Robot RobotPose

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        (self.r,) = params
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                )
            ]
        )

        A = np.c_[np.eye(3)]
        dist = gripdist + dsafe
        b = np.array([[(1 + HANDLE_OFFSET)], [-1.5 + dist], [0]])
        val = np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotWithinBounds(At):

    # RobotAt Robot Can

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        self.r, self.c = params
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int))])])

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((4, 1))
        val = 1.5e1 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotNearGrasp(At):

    # RobotAt Robot Can Grasp

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        self.r, self.c, self.g = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[
            np.r_[np.eye(2), -np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
        ]
        b = np.zeros((4, 1))
        val = 1.5 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotAtGrasp(At):

    # RobotAt Robot Can Grasp

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        self.r, self.c, self.g = params
        k = "pose" if not self.r.is_symbol() else "value"
        attr_inds = OrderedDict(
            [
                (self.r, [(k, np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[
            np.r_[np.eye(2), -np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
        ]
        # A[:,4:6] *= 1.75
        # A[:,4:6] *= 2.5
        # A[:,4:6] *= 1.5
        b = np.zeros((4, 1))
        val = NEAR_TOL * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotPoseAtGrasp(At):
    pass


class RobotWithinReach(At):

    # RobotAt Robot Target

    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Robot RobotPose
        self.r, self.t = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.t, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)]]
        b = np.zeros((4, 1))
        val = 20 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class InCloset(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        ## At Can Target
        (self.can,) = params
        k = "value" if self.can.is_symbol() else "pose"
        attr_inds = OrderedDict([(self.can, [(k, np.array([0, 1], dtype=np.int))])])

        A = np.r_[np.eye(2), -np.eye(2)]
        b = np.array([[0.0], [-5.5], [0.0], [5.5]])
        val = np.array([[0.25], [1.0], [0.25], [1.0]])
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(InCloset, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )


class InClosetObj(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, coeff=1e-1, debug=False
    ):
        ## At Can Target
        self.can, self.targ = params
        self.coeff = coeff
        k = "value" if self.can.is_symbol() else "pose"
        attr_inds = OrderedDict([(self.can, [(k, np.array([0, 1], dtype=np.int))])])

        A = np.r_[np.eye(2), -np.eye(2)]
        b = np.array([[0.0], [-5.5], [0.0], [5.5]])
        val = np.array([[0.25], [1.0], [0.25], [1.0]])
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(InClosetObj, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )

    def f(self, x):
        if x[1] < 3.0:
            return np.zeros((2, 1))
        return self.coeff * np.array([[x[0] ** 2 + (x[1.0] - 7.0) ** 2]])

    def grad(self, x):
        if x[1] < 3.0:
            return np.zeros((1, 2))
        return self.coeff * np.array([2 * x[0], 2 * x[1.0] - 14.0]).reshape((1, 2))

    def hess(self, x):
        if x[1] < 3.0:
            return np.zeros((2, 2))
        return self.coeff * 2.0 * np.eye(2)


class Pointing(ExprPredicate):
     def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        robot, target = params
        attr_inds = OrderedDict(
            [(robot, [("pose", np.array([0], dtype=np.int))])]
        )
        A = np.ones((1, 1))
        b = np.zeros((1, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, np.array([np.arctan(target.pose[1, 0]/target.pose[0, 0])]))
        super(Pointing, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )

class BPointing(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        self.robot, self.target = params
        attr_inds = OrderedDict(
            [
                (self.robot, [("pose", np.array([0], dtype=np.int))]),
                (self.target, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        # define these
        col_expr = Expr(self.f, grad=self.grad_f)
        val = np.zeros((1, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(BPointing, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            tol=1e-2,
            debug=debug,
            priority=0
        )

    def f(self, x):
        diff = np.tan(x[0]) - x[2]/x[1]
        # return np.array([diff, -diff])
        return diff

    def grad_f(self, x):
        grad = np.array([1/np.cos(x[0])**2, -x[2]/(x[1]**2), 1/x[1]]).reshape(1, -1)
        # return np.array([grad[0], -grad[0]])
        # breakpoint()
        return grad
    
    # def hess(self, x):
    #     hessian = np.array([[-2*x[0]/((1+x[0]**2)**2), 0, 0],
    #                         [0, 2*x[2]/(x[1]**2), -1/(x[1]**2)],
    #                         [0, -1/(x[1]**2), 0]])
    #     return hessian

    # def test(self, time, negated=False, tol=1e-3):
    #     # This test is overwritten so that collisions can be calculated correctly
    #     if not self.is_concrete():
    #         return False
    #     if time < 0:
    #         traceback.print_exception(*sys.exc_info())
    #         raise PredicateException("Out of range time for predicate '%s'." % self)
    #     try:
    #         result = self.expr.eval(
    #             self.get_param_vector(time), tol=tol
    #         )
    #         return result
    #     except IndexError:
    #         traceback.print_exception(*sys.exc_info())
    #         ## this happens with an invalid time
    #         raise PredicateException("Out of range time for predicate '%s'." % self)


## a stub computing only the
class CertainPosition(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.target,) = params
        attr_inds = OrderedDict(
            [(self.target, [("pose", np.array([0, 1], dtype=np.int))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        # unused constraints, pass some BS in
        A = np.zeros((1, 2))
        b = np.zeros((1,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((1, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(CertainPosition, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=1
        )

    def test(self, time, negated=False, tol=None):
        diff_vec = self.target.belief.samples[:,:,time].detach().numpy() - self.target.pose[:,time]
        
        if negated:
            return not np.sqrt(np.power(diff_vec, 2)).mean() <= 0.2
        return np.sqrt(np.power(diff_vec, 2)).mean() <= 0.2

class ConfirmedPosition(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        # NOTE: Below line is for debugging purposes only, should be commented out
        # and line below should be commented in
        # self._debug = True
        # self._debug = debug

        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        # self._env = env
        (self.target,) = params
        attr_inds = OrderedDict(
            [(self.target, [("pose", np.array([0, 1], dtype=np.int))])]
        )
        # self._param_to_body = {
        #     self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
        #     self.targ: self.lazy_spawn_or_body(
        #         self.targ, self.targ.name, self.targ.geom
        #     ),
        # }

        # INCONTACT_COEFF = 1e1
        # unused constraints, pass some BS in
        A = np.zeros((1,2))
        b = np.zeros((1,1))
        dummy_expr = AffExpr(A, b)
        val = np.zeros((1, 1)) # output of fcn should be zero
        # val = np.zeros((1, 1))
        e = EqExpr(dummy_expr, val)
        super(ConfirmedPosition, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            priority=-1
        )

class GripperClosed(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        (self.robot,) = params
        attr_inds = OrderedDict(
            [(self.robot, [("gripper", np.array([0], dtype=np.int))])]
        )
        A = np.ones((1, 1))
        b = np.zeros((1, 1))
        val = GRIP_VAL * np.ones((1, 1))  # (GRIP_TOL + 1e-1) * -np.ones((1,1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        # e = LEqExpr(aff_e, val)

        neg_val = -GRIP_VAL * np.ones((1, 1))  # (GRIP_TOL - 1e-1) * np.ones((1,1))
        neg_aff_e = AffExpr(A, b)
        self.neg_expr = EqExpr(neg_aff_e, neg_val)
        # self.neg_expr = LEqExpr(neg_aff_e, neg_val)
        super(GripperClosed, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return self.expr


class DoorClosed(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        (self.door,) = params
        if not hasattr(self, "coeff"):
            self.coeff = 1e0
        attr_inds = OrderedDict([(self.door, [("theta", np.array([0], dtype=np.int))])])
        A = self.coeff * np.ones((1, 1))
        b = np.zeros((1, 1))
        val = np.zeros((1, 1))  # (GRIP_TOL + 1e-1) * -np.ones((1,1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        # e = LEqExpr(aff_e, val)

        neg_val = (
            -np.pi / 2 * self.coeff * np.ones((1, 1))
        )  # (GRIP_TOL - 1e-1) * np.ones((1,1))
        neg_aff_e = AffExpr(A, b)
        self.neg_expr = EqExpr(neg_aff_e, neg_val)
        # self.neg_expr = LEqExpr(neg_aff_e, neg_val)
        super(DoorClosed, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return self.expr


class DoorNearClosed(DoorClosed):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        self.coeff = 1e-2
        super(DoorNearClosed, self).__init__(
            name, params, expected_param_types, env, sess
        )
        self._rollout = True


class InContact(CollisionPredicate):

    # InContact, Robot, RobotPose, Target

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self._env = env
        self.robot, self.rp, self.targ = params
        attr_inds = OrderedDict(
            [
                (self.rp, [("value", np.array([0, 1], dtype=np.int))]),
                (self.targ, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
            self.targ: self.lazy_spawn_or_body(
                self.targ, self.targ.name, self.targ.geom
            ),
        }

        INCONTACT_COEFF = 1e1
        self.f = lambda x: INCONTACT_COEFF * self.distance_from_obj(x)[0]
        self.grad = lambda x: INCONTACT_COEFF * self.distance_from_obj(x)[1]

        col_expr = Expr(self.f, self.grad)
        val = np.ones((1, 1)) * dsafe * INCONTACT_COEFF
        # val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(InContact, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            ind0=1,
            ind1=2,
            active_range=(0, 0),
        )

    def test(self, time, negated=False, tol=1e-4):
        return super(CollisionPredicate, self).test(time, negated, tol)


class Collides(CollisionPredicate):

    # Collides Can Obstacle (wall)

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self._env = env
        self.c, self.w = params
        attr_inds = OrderedDict(
            [
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }
        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        # f_neg = lambda x: self.distance_from_obj(x)[0]
        # def grad_neg(x):
        #     # print self.distance_from_obj(x)
        #     return -self.distance_from_obj(x)[1]

        col_expr = Expr(self.f, self.grad)
        val = np.zeros((COL_TS * N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(self.f_neg, self.grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Collides, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1
        )
        self.n_cols = N_COLS
        # self.priority = 1

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

    def f(self, x):
        return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4)

    def grad(self, x):
        return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4, grad=True)

    def f_neg(self, x):
        return -self.f(x)

    def grad_neg(self, x):
        return -self.grad(x)


class CanCollides(Collides):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        super(CanCollides, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1
        )
        self.dsafe = 0.2


class TargetGraspCollides(Collides):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self._env = env
        self.c, self.w, self.g = params
        if self.c.is_symbol():
            k = "value"
        else:
            k = "pose"
        attr_inds = OrderedDict(
            [
                (self.c, [(k, np.array([0, 1], dtype=np.int))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        self.dist = RETREAT_DIST
        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        # f_neg = lambda x: self.distance_from_obj(x)[0]
        # def grad_neg(x):
        #     # print self.distance_from_obj(x)
        #     return -self.distance_from_obj(x)[1]

        col_expr = Expr(self.f, self.grad)
        val = np.zeros((N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(self.f_neg, self.grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Collides, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            active_range=(0, 0),
            priority=2,
        )
        self.n_cols = N_COLS
        # self.priority = 1

    def f(self, x):
        disp = x[:2] + self.dist * x[4:6]
        new_x = np.concatenate([disp, x[2:4]])
        return -self.distance_from_obj(new_x)[0]

    def grad(self, x):
        disp = x[:2] + self.dist * x[4:6]
        new_x = np.concatenate([disp, x[2:4]])
        jac = self.distance_from_obj(new_x)[1]
        return np.c_[np.zeros((N_COLS, 2)), jac]

    def f_neg(self, x):
        return -self.f(x)

    def grad_neg(self, x):
        return self.grad(x)

    def set_pos(self, x):
        return self._set_pos(x)
        if self.c._type.lower() == "robot":
            return self._set_robot_pos(x)
        else:
            return self._set_pos(x)


class RobotCanGraspCollides(Collides):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self._env = env
        self.c, self.w, self.g = params
        if self.c.is_symbol():
            k = "value"
        else:
            k = "pose"
        attr_inds = OrderedDict(
            [
                (self.c, [(k, np.array([0, 1], dtype=np.int))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        def f(x):
            return -self.distance_from_obj(x[:4])[0]

        def grad(x):
            jac = self.distance_from_obj(x[:4])[1]
            return np.c_[jac, np.zeros((N_COLS, 2))]

        def f_neg(x):
            return -f(x)

        def grad_neg(x):
            return grad(x)

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        # f_neg = lambda x: self.distance_from_obj(x)[0]
        # def grad_neg(x):
        #     # print self.distance_from_obj(x)
        #     return -self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Collides, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            active_range=(0, 0),
            priority=2,
        )
        self.n_cols = N_COLS
        # self.priority = 1

    def set_pos(self, x):
        return self._set_pos(x)
        if self.c._type.lower() == "robot":
            return self._set_robot_pos(x)
        else:
            return self._set_pos(x)


class CanGraspCollides(TargetGraspCollides):
    pass


class TargetCanGraspCollides(TargetGraspCollides):
    pass


class TargetCollides(Collides):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self._env = env
        self.c, self.w = params
        attr_inds = OrderedDict(
            [
                (self.c, [("value", np.array([0, 1], dtype=np.int))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        def f(x):
            return -self.distance_from_obj(x)[0]

        def grad(x):
            return self.distance_from_obj(x)[1]

        def f_neg(x):
            return -f(x)

        def grad_neg(x):
            return grad(x)

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        # f_neg = lambda x: self.distance_from_obj(x)[0]
        # def grad_neg(x):
        #     # print self.distance_from_obj(x)
        #     return -self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Collides, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            active_range=(0, 0),
        )
        self.n_cols = N_COLS
        # self.priority = 1


class PoseCollides(TargetCollides):
    pass


class RCollides(CollisionPredicate):

    # RCollides Robot Obstacle (Wall)

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self._env = env
        self.hl_ignore = True
        self.r, self.w = params
        # self.check_aabb = True

        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("gripper", np.array([0], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
                (self.w, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )

        self._param_to_body = {
            self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        self.neg_coeff = 1e3
        self.neg_grad_coeff = 1e-1

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        col_expr = Expr(self.f, self.grad)
        val = np.zeros((COL_TS * N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(self.f_neg, self.grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(RCollides, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1
        )
        self.n_cols = N_COLS
        self.hl_ignore = True

        # self.priority = 1

    def f(self, x):
        return -twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6)

    def grad(self, x):
        return -twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6, grad=True)

    def f_neg(self, x):
        return -self.neg_coeff * self.f(x)

    def grad_neg(self, x):
        return -self.neg_grad_coeff * self.grad(x)

    def resample(self, negated, time, plan):
        assert negated
        res = OrderedDict()
        attr_inds = OrderedDict()
        a = 0
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None

        act = plan.actions[a]
        x1 = self.get_param_vector(time)
        val, jac = self.distance_from_obj(x1[:6], 0)
        jac = -jac[0, :2]
        if np.all(jac == 0):
            return None, None

        jac = jac / (np.linalg.norm(jac) + 1e-3)

        new_robot_pose = self.r.pose[:, time] + np.random.uniform(0.1, 0.3) * jac
        st = max(max(time - 3, 0), act.active_timesteps[0])
        ref_st = max(max(time - 3, 1), act.active_timesteps[0] + 1)
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        ref_et = min(min(time + 3, plan.horizon - 2), act.active_timesteps[1] - 1)
        for i in range(ref_st, et):
            dist = float(np.abs(i - time))
            if i <= time:
                inter_rp = (dist / 3.0) * self.r.pose[:, st] + (
                    (3.0 - dist) / 3.0
                ) * new_robot_pose
            else:
                inter_rp = (dist / 3.0) * self.r.pose[:, et] + (
                    (3.0 - dist) / 3.0
                ) * new_robot_pose

            add_to_attr_inds_and_res(i, attr_inds, res, self.r, [("pose", inter_rp)])
        return res, attr_inds

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


class Obstructs(CollisionPredicate):

    # Obstructs, Robot, RobotPose, RobotPose, Can;

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self._env = env
        # self.hl_ignore = True
        self.r, self.startp, self.endp, self.c = params

        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("gripper", np.array([0], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
        }

        self.rs_scale = RS_SCALE

        self.neg_coeff = 1e2  # 1e3
        self.neg_grad_coeff = 1e-2  # 1e-3

        col_expr = Expr(self.f, self.grad)
        val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(self.f_neg, self.grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Obstructs, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=3
        )
        # self.priority=1

    def f(self, x):
        return -twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6)

    def grad(self, x):
        grad = -twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6, grad=True)
        rotjac = np.arctan2(grad[:, 0] - grad[:, 6], grad[:, 1] - grad[:, 7])
        grad[:, 3] = -rotjac
        return grad

    def f_neg(self, x):
        return -self.neg_coeff * self.f(x)

    def grad_neg(self, x):
        return -self.neg_grad_coeff * self.grad(x)

    def resample(self, negated, time, plan):
        assert negated
        res = OrderedDict()
        attr_inds = OrderedDict()
        # for param in [self.startp, self.endp]:
        #     val, inds = sample_pose(plan, param, self.r, self.rs_scale)
        #     if val is None:
        #         continue
        #     res.extend(val[inds].flatten().tolist())
        #     # inds[0] returns the x values of the indices which is what we care
        #     # about, because the y values correspond to time.
        #     attr_inds[param] = [('value', inds[0])]
        #     import ipdb; ipdb.set_trace()
        a = 0
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None

        act = plan.actions[a]

        disp = self.c.pose[:, time] - self.r.pose[:, time]
        use_t = time
        new_disp = disp
        if (
            time < plan.actions[a].active_timesteps[1]
            and np.linalg.norm(disp)
            > self.r.geom.radius + self.c.geom.radius + self.dsafe
        ):
            new_disp = self.c.pose[:, time + 1] - self.r.pose[:, time + 1]

        if (
            np.linalg.norm(new_disp)
            < self.r.geom.radius + self.c.geom.radius + self.dsafe
        ):
            disp = new_disp
        else:
            disp = (disp + new_disp) / 2.0

        disp /= (np.linalg.norm(disp) + 1e-5)
        if disp[0] == 0:
            orth = np.array([1.0, 0.0])
        elif disp[1] == 0:
            orth = np.array([0.0, 1.0])
        else:
            orth = np.array([1./disp[0], -1./disp[1]])
        disp += np.random.normal(0, 0.2, (2,))
        disp += 1e-7

        st = max(max(time - 3, 0), act.active_timesteps[0])
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        long_disp = self.r.pose[:, et] - self.r.pose[:, st]
        long_disp /= np.linalg.norm(long_disp)
        d1, d2 = long_disp.dot(orth), long_disp.dot(-orth)

        if d1 > d2:
            w1, w2 = 0.1, 0.9
        else:
            w1, w2 = 0.9, 0.1
        orth *= np.random.choice([-1.0, 1.0], p=[w1, w2])
        orth = orth / np.linalg.norm(orth)

        # rdisp = -(self.c.geom.radius + self.r.geom.radius + self.dsafe + 1e-1) * disp / np.linalg.norm(disp)
        rdist = np.random.uniform(0, 0.6)
        rdisp = -(self.c.geom.radius + self.r.geom.radius + self.dsafe + rdist) * disp / np.linalg.norm(disp)
        orth = rdisp # + np.random.uniform(0.05, 0.2) * orth
        # orth *= np.random.uniform(0.7, 1.5) * (self.c.geom.radius + self.r.geom.radius + self.dsafe)
        # orth += np.random.uniform([-0.15, 0.15], [-0.15, 0.15])

        # new_robot_pose = self.r.pose[:, time] + orth
        disp = orth
        st = max(max(time - 3, 1), act.active_timesteps[0] + 1)
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        ref_st = max(max(time - 3, 0), act.active_timesteps[0])
        ref_et = min(min(time + 3, plan.horizon - 2), act.active_timesteps[1] - 1)
        poses = []
        for i in range(st, et):
            dist = float(np.abs(i - time))
            if i <= time:
                inter_rp = (dist / 3.0) * self.r.pose[:, st] + ((3.0 - dist) / 3.0) * (
                    self.r.pose[:, st] + disp
                )
                # inter_rp = (dist / 3.) * self.r.pose[:, time] + ((3. - dist) / 3.) * (self.r.pose[:, time] + disp)
                inter_rp = (dist / 3.0) * self.r.pose[:, ref_st] + (
                    (3.0 - dist) / 3.0
                ) * (self.r.pose[:, i] + disp)
            else:
                inter_rp = (dist / 3.0) * self.r.pose[:, et] + ((3.0 - dist) / 3.0) * (
                    self.r.pose[:, et] + disp
                )
                # inter_rp = (dist / 3.) * self.r.pose[:, time] + ((3. - dist) / 3.) * (self.r.pose[:, time] + disp)
                inter_rp = (dist / 3.0) * self.r.pose[:, ref_et] + (
                    (3.0 - dist) / 3.0
                ) * (self.r.pose[:, i] + disp)

            poses.append(inter_rp)
            if len(poses) > 1:
                newtheta = np.arctan2(*(poses[-1] - poses[-2]))
                curtheta = self.r.theta[0, time]
                opp_theta = opposite_angle(newtheta)
                theta = newtheta
                if np.abs(angle_diff(curtheta, newtheta)) > np.abs(
                    angle_diff(opp_theta, curtheta)
                ):
                    theta = opp_theta
                # add_to_attr_inds_and_res(i-1, attr_inds, res, self.r, [('theta', np.array([theta]))])
            add_to_attr_inds_and_res(i, attr_inds, res, self.r, [("pose", inter_rp)])
        return res, attr_inds

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


class WideObstructs(Obstructs):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        super(WideObstructs, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = 0.35 #0.001 #0.05  # 0.35
        self.check_aabb = False  # True


class ObstructsNoSym(Obstructs):
    pass


class WideObstructsNoSym(WideObstructs):
    pass


class DoorObstructs(CollisionPredicate):

    # Obstructs, Robot, RobotPose, RobotPose, Can;

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self._env = env
        # self.hl_ignore = True
        self.r, self.startp, self.endp, self.door = params

        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("gripper", np.array([0], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
                (
                    self.door,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
            ]
        )
        self._param_to_body = {
            self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
            self.door: self.lazy_spawn_or_body(
                self.door, self.door.name, self.door.geom
            ),
        }

        self.rs_scale = RS_SCALE

        self.neg_coeff = 1e2  # 1e3
        self.neg_grad_coeff = 1e-2  # 1e-3

        col_expr = Expr(self.f, self.grad)
        val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(self.f_neg, self.grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(DoorObstructs, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=3
        )
        # self.priority=1
        self.dsafe = 0.1

    def f(self, x):
        return -twostep_f([x[:7], x[7:14]], self.distance_from_obj, 7)

    def grad(self, x):
        grad = -twostep_f([x[:7], x[7:14]], self.distance_from_obj, 7, grad=True)
        rotjac = np.arctan2(grad[:, 0] - grad[:, 7], grad[:, 1] - grad[:, 8])
        grad[:, 3] = -rotjac
        return grad

    def f_neg(self, x):
        return -self.neg_coeff * self.f(x)

    def grad_neg(self, x):
        return -self.neg_grad_coeff * self.grad(x)

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


def sample_pose(plan, pose, robot, rs_scale):
    from pma.ll_solver_gurobi import NAMOSolver
    targets = plan.get_param("InContact", 2, {0: robot, 1: pose})
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    inds = np.where(pose._free_attrs["value"])
    if np.sum(inds) == 0:
        return None, None  ## no resampling for this one
    if len(targets) == 1:
        # print "one target", pose
        random_dir = np.random.rand(2, 1) - 0.5
        random_dir = random_dir / np.linalg.norm(random_dir)
        # assumes targets are symbols
        val = targets[0].value + random_dir * 3 * robot.geom.radius
    elif len(targets) == 0:
        ## old generator -- just add a random perturbation
        # print "no targets", pose
        val = np.random.normal(pose.value[:, 0], scale=rs_scale)[:, None]
    else:
        raise NotImplementedError
    # print pose, val
    pose.value = val

    ## make the pose collision free
    _, collision_preds = plan.get_param("RCollides", 1, negated=True, return_preds=True)
    _, at_preds = plan.get_param(
        "RobotAt", 1, {0: robot, 1: pose}, negated=False, return_preds=True
    )
    preds = [(collision_preds[0], True), (at_preds[0], False)]
    old_pose = robot.pose.copy()
    old_free = robot._free_attrs["pose"].copy()
    robot.pose = pose.value.copy()
    robot._free_attrs["pose"][:] = 1

    wall = collision_preds[0].params[1]
    old_w_pose = wall.pose.copy()
    wall.pose = wall.pose[:, 0][:, None]

    old_priority = [p.priority for p, n in preds]
    for p, n in preds:
        P.priority = -1
    p = Plan.create_plan_for_preds(preds, collision_preds[0]._env)
    s = NAMOSolver(transfer_norm="l2")
    success = s._solve_opt_prob(p, 0, resample=False, verbose=False)

    # print success

    ## debugging
    # import viewer
    # v = viewer.OpenRAVEViewer.create_viewer()
    # v.draw_plan_ts(p, 0)
    # print pose.value, val

    ## restore the old values
    robot.pose = old_pose
    robot._free_attrs["pose"] = old_free
    for i, (p, n) in enumerate(preds):
        P.priority = old_priority[i]

    wall.pose = old_w_pose

    return pose.value, inds


class ObstructsHolding(CollisionPredicate):

    # ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can;
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self._env = env
        r, startp, endp, obstr, held = params
        self.r = r
        self.startp, self.endp = startp, endp
        self.obstr = obstr
        self.held = held

        self.rs_scale = RS_SCALE

        attr_inds = OrderedDict(
            [
                (
                    r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("gripper", np.array([0], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
                (obstr, [("pose", np.array([0, 1], dtype=np.int))]),
                (held, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )

        self._param_to_body = {
            r: self.lazy_spawn_or_body(r, r.name, r.geom),
            obstr: self.lazy_spawn_or_body(obstr, obstr.name, obstr.geom),
            held: self.lazy_spawn_or_body(held, held.name, held.geom),
        }

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        self.neg_coeff = 1e2  # 1e3
        self.neg_grad_coeff = 1e-2  # 1e-3
        ## so we have an expr for the negated predicate
        # f_neg = lambda x: neg_coeff*self.distance_from_obj(x)[0]
        # grad_neg = lambda x: neg_grad_coeff*self.distance_from_obj(x)[1]

        col_expr = Expr(self.f, self.grad)
        val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(self.f_neg, self.grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(ObstructsHolding, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=3
        )
        # self.priority=1

    def f(self, x):
        if self.obstr.name == self.held.name:
            return -0 * twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6)
        else:
            return -twostep_f([x[:8], x[8:16]], self.distance_from_obj, 8)

    def grad(self, x):
        if self.obstr.name == self.held.name:
            grad = -twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6, grad=True)
            return grad
        else:
            grad = -twostep_f([x[:8], x[8:16]], self.distance_from_obj, 8, grad=True)
            rotjac = np.arctan2(grad[:, 0] - grad[:, 8], grad[:, 1] - grad[:, 9])
            grad[:, 3] = -rotjac
            return grad

    def f_neg(self, x):
        return -self.neg_coeff * self.f(x)

    def grad_neg(self, x):
        gradneg = -self.neg_grad_coeff * self.grad(x)
        return gradneg

    def resample(self, negated, time, plan):
        assert negated
        a = 0
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None
        act = plan.actions[a]

        res = OrderedDict()
        attr_inds = OrderedDict()
        disp1 = self.obstr.pose[:, time] - self.held.pose[:, time]
        disp2 = self.obstr.pose[:, time] - self.r.pose[:, time]
        disp = disp1 if np.linalg.norm(disp1) < np.linalg.norm(disp2) else disp2
        disp /= np.linalg.norm(disp) + 1e-7
        if disp[0] == 0:
            orth = np.array([1.0, 0.0])
        elif disp[1] == 0:
            orth = np.array([0.0, 1.0])
        else:
            orth = np.array([1./disp[0], -1./disp[1]])
        #disp += np.random.normal(0, 0.5, (2,))
        disp += 1e-7

        st = max(max(time - 3, 0), act.active_timesteps[0])
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        long_disp = self.r.pose[:, et] - self.r.pose[:, st]
        long_disp /= np.linalg.norm(long_disp)
        d1, d2 = long_disp.dot(orth), long_disp.dot(-orth)
        x = self.get_param_vector(time).flatten()
        if d1 > d2:
            w1, w2 = 0.5, 0.5
        else:
            w1, w2 = 0.5, 0.5
        orth *= np.random.choice([-1., 1.], p=[w1, w2])
        orth = orth / np.linalg.norm(orth)

        rdist = np.random.uniform(0., 0.6)
        rdisp = -(self.obstr.geom.radius + self.held.geom.radius + self.dsafe + rdist) * disp / np.linalg.norm(disp)
        orth = rdisp + np.random.uniform(0., 0.6) * orth
        # orth *= np.random.uniform(1.2, 1.8) * (self.obstr.geom.radius + self.r.geom.radius)
        # orth += np.random.uniform([-0.15, 0.15], [-0.15, 0.15])

        # ## assumes that self.startp, self.endp and target are all symbols
        # t_local = 0
        # for param in [self.startp, self.endp]:
        #     ## there should only be 1 target that satisfies this
        #     ## otherwise, choose to fail here
        #     val, inds = sample_pose(plan, param, self.r, self.rs_scale)
        #     if val is None:
        #         continue
        #     res.extend(val[inds].flatten().tolist())
        #     ## inds[0] returns the x values of the indices which is what we care
        #     ## about, because the y values correspond to time.
        #     attr_inds[param] = [('value', inds[0])]

        new_robot_pose = self.r.pose[:, time] + orth
        new_held_pose = self.held.pose[:, time] + orth
        # add_to_attr_inds_and_res(time, attr_inds, res, self.r, [('pose', new_robot_pose)])
        # add_to_attr_inds_and_res(time, attr_inds, res, self.held, [('pose', new_held_pose)])
        st = max(max(time - 3, 0), act.active_timesteps[0])
        ref_st = max(max(time - 3, 1), act.active_timesteps[0] + 1)
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        ref_et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        poses = []
        for i in range(ref_st, ref_et):
            dist = float(np.abs(i - time))
            if i <= time:
                inter_rp = (dist / 3.0) * self.r.pose[:, st] + ((3.0 - dist) / 3.0) * (
                    self.r.pose[:, i] + orth
                )
                # inter_hp = (dist / 3.) * self.held.pose[:, st] + ((3. - dist) / 3.) * (self.held.pose[:, time] + orth)
            else:
                inter_rp = (dist / 3.0) * self.r.pose[:, et] + ((3.0 - dist) / 3.0) * (
                    self.r.pose[:, i] + orth
                )
                # inter_hp = (dist / 3.) * self.held.pose[:, et] + ((3. - dist) / 3.) * (self.held.pose[:, time] + orth)
            theta = self.r.theta[0, i]
            poses.append(inter_rp)
            add_to_attr_inds_and_res(i, attr_inds, res, self.r, [("pose", inter_rp)])

            inter_hp = poses[-1] + [(gripdist + dsafe) * -np.sin(theta),
                                    (gripdist + dsafe) * np.cos(theta),]
            add_to_attr_inds_and_res(i, attr_inds, res, self.held, [("pose", inter_hp)])

            # if len(poses) > 1:
            #     newtheta = np.arctan2(*(poses[-1] - poses[-2]))
            #     curtheta = self.r.theta[0, i - 1]
            #     opp_theta = opposite_angle(newtheta)
            #     theta = newtheta
            #     if np.abs(angle_diff(curtheta, newtheta)) > np.abs(
            #         angle_diff(opp_theta, curtheta)
            #     ):
            #         theta = opp_theta
            #     inter_hp = poses[-2] + [
            #         (gripdist + dsafe) * -np.sin(theta),
            #         (gripdist + dsafe) * np.cos(theta),
            #     ]
            #     add_to_attr_inds_and_res(
            #         i - 1, attr_inds, res, self.held, [("pose", inter_hp)]
            #     )
            #     # add_to_attr_inds_and_res(i-1, attr_inds, res, self.r, [('theta', np.array([theta]))])
        return res, attr_inds

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

    def distance_from_obj(self, x, n_steps=0):
        # x = [rpx, rpy, obstrx, obstry, heldx, heldy]
        pose_r = x[:2]
        pose_obstr = x[4:6]
        b0 = self._param_to_body[self.r]
        b1 = self._param_to_body[self.obstr]
        pose0, pose1 = self.set_pos(x[:6])
        aabb_vals = np.zeros((self.n_cols, 1))
        aabb_jacs = np.zeros((self.n_cols, 4))
        collisions1 = P.getClosestPoints(b0.body_id, b1.body_id, contact_dist)

        col_val1, jac01 = self._calc_grad_and_val(
            self.r.name, self.obstr.name, pose_r, pose_obstr, collisions1
        )

        if self.obstr.name == self.held.name:
            ## add dsafe to col_val1 b/c we're allowed to touch, but not intersect
            ## 1e-3 is there because the collision checker's value has some error.
            col_val1 -= self.dsafe + 1e-3
            val = np.array(col_val1)
            jac = jac01
            collisions = collisions1
        else:
            # if self.check_aabb:
            #    aabb_vals, aabb_jacs = self._check_robot_aabb(b0, b1)
            b2 = self._param_to_body[self.held]
            pose_held = x[6:8]
            b2.set_pose(pose_held)

            collisions2 = P.getClosestPoints(b2.body_id, b1.body_id, contact_dist)

            col_val2, jac21 = self._calc_grad_and_val(
                self.held.name, self.obstr.name, pose_held, pose_obstr, collisions2
            )

            if np.max(col_val1) > np.max(col_val2):
                val = np.array(col_val1)
                jac = np.c_[jac01, np.zeros((N_COLS, 2))].reshape((N_COLS, 6))
                collisions = collisions1
            else:
                val = np.array(col_val2)
                jac = np.c_[np.zeros((N_COLS, 2)), jac21[:, 2:], jac21[:, :2]].reshape(
                    (N_COLS, 6)
                )
                collisions = collisions2

            # if np.max(aabb_vals) > np.max(val):
            #    val = aabb_vals
            #    jac = aabb_jacs

        if b0.isrobot():
            if len(collisions):
                pose0, pose1 = np.r_[pose0, [[0]]], np.r_[pose1, [[0]]]
                colvec = np.array([c[5] for c in collisions])
                axisvec = np.array([[0, 0, 1] for _ in collisions])
                pos0vec = np.array([pose0.flatten() for _ in collisions])
                crosstorque = np.cross(colvec - pos0vec, [0, 0, 1])
                rotjac = np.dot(crosstorque, pose1 - pose0)
                rotjac = 0 * np.r_[rotjac, np.zeros((len(jac) - len(collisions), 1))]
            else:
                rotjac = np.zeros((jac.shape[0], 1))
            jac = np.c_[jac[:, :2], rotjac, np.zeros_like(rotjac), jac[:, 2:]]

        return val, jac


class WideObstructsHolding(ObstructsHolding):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        super(WideObstructsHolding, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = 0.05  # 0.35
        self.check_aabb = False  # True


class ObstructsHoldingNoSym(ObstructsHolding):
    pass


class WideObstructsHoldingNoSym(WideObstructsHolding):
    pass


class InGripper(ExprPredicate):

    # InGripper, Robot, Can, Grasp

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.r, self.can, self.grasp = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.can, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.grasp, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        # want x0 - x2 = x4, x1 - x3 = x5
        A = 1e1 * np.array([[1, 0, -1, 0, -1, 0], [0, 1, 0, -1, 0, -1]])
        b = np.zeros((2, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((2, 1)))

        super(InGripper, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )


class InGraspAngle(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.r, self.can = params
        self.dist = 0.0
        self.dsafe = dsafe
        self.gripdist = gripdist
        self.coeff = 0.08
        if self.r.is_symbol():
            k = "value"
        else:
            k = "pose"

        if self.can.is_symbol():
            obj_k = "value"
        else:
            obj_k = "pose"

        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        (k, np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
                (self.can, [(obj_k, np.array([0, 1], dtype=np.int))]),
            ]
        )
        angle_expr = Expr(self.f, self.grad)
        e = EqExpr(angle_expr, np.zeros((3,1)))
        e = EqExpr(angle_expr, np.zeros((1,1)))

        super(InGraspAngle, self).__init__(name, e, attr_inds, params, expected_param_types, priority=1)
        self._init_include = False
        self._rollout = False

        super(InGraspAngle, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=1
        )

    def f(self, x):
        x = x.flatten()
        dist = self.dist + self.gripdist + self.dsafe
        targ_loc = [-dist * np.sin(x[2]), dist * np.cos(x[2])]
        can_loc = x[3:5]
        return self.coeff * np.array(
            [
                [
                    ((x[0] + targ_loc[0]) - can_loc[0]) ** 2
                    + ((x[1] + targ_loc[1]) - can_loc[1]) ** 2
                ]
            ]
        )

    def grad(self, x):
        x = x.flatten()
        curdisp = x[3:5] - x[:2]
        dist = self.dist + self.gripdist + self.dsafe
        theta = x[2]
        targ_disp = [-dist * np.sin(theta), dist * np.cos(theta)]
        off = (curdisp[0] - targ_disp[0]) ** 2 + (curdisp[1] - targ_disp[1]) ** 2
        (x1, y1), (x2, y2) = x[:2], x[3:5]

        x1_grad = -2 * ((x2 - x1) + dist * np.sin(theta))
        y1_grad = -2 * ((y2 - y1) - dist * np.cos(theta))
        theta_grad = 2 * dist * ((x2 - x1) * np.cos(theta) + (y2 - y1) * np.sin(theta))
        x2_grad = 2 * ((x2 - x1) + dist * np.sin(theta))
        y2_grad = 2 * ((y2 - y1) - dist * np.cos(theta))
        return self.coeff * np.array(
            [x1_grad, y1_grad, theta_grad, x2_grad, y2_grad]
        ).reshape((1, 5))

    def resample(self, negated, time, plan):
        res = OrderedDict()
        attr_inds = OrderedDict()
        a = 0
        # while  a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] < time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None

        act = plan.actions[a]
        x = self.get_param_vector(time).flatten()
        theta = -np.arctan2(*(x[3:5]-x[:2]))
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        disp = rot.dot([0, -gripdist-dsafe])
        offset = (x[3:5] + disp) - x[:2]
        if time == plan.actions[a].active_timesteps[1]:
            new_robot_pose = x[:2] + offset
            new_can_pose = x[3:5]
        else:
            new_robot_pose = x[:2] + offset / 2.
            new_can_pose = x[3:5] - offset / 2.
        # new_robot_pose = x[:2] + offset
        # new_can_pose = x[3:5]
        nsteps = 0
        st = max(max(time-nsteps,1), act.active_timesteps[0]+1)
        et = min(min(time+nsteps, plan.horizon-1), act.active_timesteps[1]+1)
        ref_st = max(max(time-nsteps,0), act.active_timesteps[0])
        ref_et = min(min(time+nsteps, plan.horizon-1), act.active_timesteps[1])
        attr_key = 'value' if self.can.is_symbol() else 'pose'
        add_to_attr_inds_and_res(time, attr_inds, res, self.can, [(attr_key, new_can_pose)])
        add_to_attr_inds_and_res(time, attr_inds, res, self.r, [('pose', new_robot_pose), ('theta', np.array([theta]))])
        for i in range(st, et):
            dist = float(np.abs(i - time))
            if i <= time:
                inter_rp = (dist / nsteps) * self.r.pose[:, ref_st] + ((nsteps - dist) / nsteps) * new_robot_pose
                inter_theta = (dist / nsteps) * self.r.theta[:, ref_st] + ((nsteps - dist) / nsteps) * theta
                inter_cp = (dist / nsteps) * self.can.pose[:, ref_st] + ((nsteps - dist) / nsteps) * new_can_pose
            else:
                inter_rp = (dist / nsteps) * self.r.pose[:, ref_et] + ((nsteps - dist) / nsteps) * new_robot_pose
                inter_theta = (dist / nsteps) * self.r.theta[:, ref_et] + ((nsteps - dist) / nsteps) * theta
                inter_cp = (dist / nsteps) * self.can.pose[:, ref_et] + ((nsteps - dist) / nsteps) * new_can_pose

            add_to_attr_inds_and_res(i, attr_inds, res, self.r, [('pose', inter_rp), ('theta', inter_theta)])
            add_to_attr_inds_and_res(i, attr_inds, res, self.can, [(attr_key, inter_cp)])
        # print(new_robot_pose, new_can_pose, x)
        return res, attr_inds


class DoorInGrasp(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.coeff = 1e0
        self.r, self.door = params
        door_geom = self.door.geom
        self.handle_pos = np.array([door_geom.length, -HANDLE_OFFSET])
        self.base_theta = np.arctan2(self.handle_pos[1], self.handle_pos[0])
        self.handle_dist = np.linalg.norm(self.handle_pos)

        self.dist = 0.0
        self.dsafe = dsafe
        self.gripdist = gripdist
        if self.r.is_symbol():
            k = "value"
        else:
            k = "pose"

        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        (k, np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
                (
                    self.door,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
            ]
        )
        angle_expr = Expr(self.f, self.grad)
        e = EqExpr(angle_expr, np.zeros((1, 1)))

        super(DoorInGrasp, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=2
        )

    def f(self, x):
        x = x.flatten()
        dist1 = self.handle_dist
        dist2 = self.dist + self.gripdist + self.dsafe

        rloc, dloc = x[:2], x[3:5]
        rloc += [-dist2 * np.sin(x[2]), dist2 * np.cos(x[2])]
        dloc += [
            dist1 * np.cos(self.base_theta + x[5]),
            dist1 * np.sin(self.base_theta + x[5]),
        ]
        return np.array([[self.coeff * np.sum((rloc - dloc) ** 2)]])

    def grad(self, x):
        x = x.flatten()
        dist1 = self.handle_dist
        dist2 = self.dist + self.gripdist + self.dsafe
        (x1, y1), (x2, y2) = x[:2], x[3:5]
        t1, t2 = x[2], x[5]
        x1_grad = 2 * (
            x1 - x2 - dist2 * np.sin(t1) - dist1 * np.cos(self.base_theta + t2)
        )
        x2_grad = -x1_grad
        y1_grad = 2 * (
            y1 - y2 + dist2 * np.cos(t1) - dist1 * np.sin(self.base_theta + t2)
        )
        y2_grad = -y1_grad
        t1_grad = (
            2
            * dist2
            * (
                np.cos(t1) * (-x1 + x2 + dist1 * np.cos(self.base_theta + t2))
                + np.sin(t1) * (-y1 + y2 + dist1 * np.sin(self.base_theta + t1))
            )
        )
        t2_grad = (
            -2
            * dist1
            * (
                np.sin(self.base_theta + t2) * (-x1 + x2 + dist2 * np.sin(t1))
                + np.cos(self.base_theta + t2) * (y1 - y2 + dist2 * np.cos(t1))
            )
        )
        return np.array([x1_grad, y1_grad, t1_grad, x2_grad, y2_grad, t2_grad]).reshape(
            (1, -1)
        )

    def resample(self, negated, time, plan):
        return None, None
        res = OrderedDict()
        attr_inds = OrderedDict()
        a = 0
        # while  a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] < time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None

        act = plan.actions[a]
        x = self.get_param_vector(time).flatten()
        theta = np.arctan2(*(x[3:5] - x[:2]))
        rot = np.array(
            [[np.cos(-x[2]), -np.sin(-x[2])], [np.sin(-x[2]), np.cos(-x[2])]]
        )
        disp = rot.dot([0, -gripdist - dsafe])
        offset = (x[3:5] + disp) - x[:2]
        new_robot_pose = x[:2] + offset / 2.0
        new_can_pose = x[3:5] - offset  # / 2.
        # new_robot_pose = x[:2] + offset
        # new_can_pose = x[3:5]
        nsteps = 0
        st = max(max(time - nsteps, 1), act.active_timesteps[0] + 1)
        et = min(min(time + nsteps, plan.horizon - 1), act.active_timesteps[1])
        ref_st = max(max(time - nsteps, 0), act.active_timesteps[0])
        ref_et = min(min(time + nsteps, plan.horizon - 1), act.active_timesteps[1])
        add_to_attr_inds_and_res(
            time, attr_inds, res, self.can, [("pose", new_can_pose)]
        )
        # add_to_attr_inds_and_res(time, attr_inds, res, self.r, [('pose', new_robot_pose)])
        for i in range(st, et):
            dist = float(np.abs(i - time))
            if i <= time:
                inter_rp = (dist / nsteps) * self.r.pose[:, ref_st] + (
                    (nsteps - dist) / nsteps
                ) * new_robot_pose
                inter_cp = (dist / nsteps) * self.can.pose[:, ref_st] + (
                    (nsteps - dist) / nsteps
                ) * new_can_pose
            else:
                inter_rp = (dist / nsteps) * self.r.pose[:, ref_et] + (
                    (nsteps - dist) / nsteps
                ) * new_robot_pose
                inter_cp = (dist / nsteps) * self.can.pose[:, ref_et] + (
                    (nsteps - dist) / nsteps
                ) * new_can_pose
            inter_theta = np.array([np.arctan2(*(inter_cp - inter_rp))])

            # add_to_attr_inds_and_res(i, attr_inds, res, self.r, [('pose', inter_rp),( 'theta', inter_theta)])
            add_to_attr_inds_and_res(i, attr_inds, res, self.can, [("pose", inter_cp)])
        # print(new_robot_pose, new_can_pose, x)
        return res, attr_inds


class ApproachGraspAngle(InGraspAngle):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        super(ApproachGraspAngle, self).__init__(
            name, params, expected_param_types, env, sess, debug
        )
        self.dist = 1.2


class NearGraspAngle(InGraspAngle):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        super(NearGraspAngle, self).__init__(name, params, expected_param_types,env, sess, debug)
        self.coeff = 4e-2
        self._rollout = True


class TargNearGraspAngle(NearGraspAngle):
    pass


# class NearGraspAngle(InGraspAngle):
#    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
#        self.r, self.can = params
#        self.tol = 2e-3
#        if self.r.is_symbol():
#            k = 'value'
#        else:
#            k = 'pose'
#        attr_inds = OrderedDict([(self.r, [(k, np.array([0, 1], dtype=np.int)),
#                                           ("theta", np.array([0], dtype=np.int))]),
#                                 (self.can, [("pose", np.array([0, 1], dtype=np.int))]),
#                                ])
#        angle_expr = Expr(self.f, self.grad)
#        e = LEqExpr(angle_expr, self.tol*np.ones((1,1)))
#
#        super(InGraspAngle, self).__init__(name, e, attr_inds, params, expected_param_types, priority=1)
#        self._rollout = True
#
#    def f(self, x):
#        x = x.flatten()
#        dist = gripdist + dsafe
#        targ_loc = [-dist * np.sin(x[2]), dist * np.cos(x[2])]
#        can_loc = x[3:5]
#        return np.array([[((x[0]+targ_loc[0])-can_loc[0])**2 + ((x[1]+targ_loc[1])-can_loc[1])**2]])
#
#    def grad(self, x):
#        x = x.flatten()
#        curdisp = x[3:5] - x[:2]
#        dist = gripdist + dsafe
#        theta = x[2]
#        targ_disp = [-dist * np.sin(theta), dist * np.cos(theta)]
#        off = (curdisp[0]-targ_disp[0])**2 + (curdisp[1]-targ_disp[1])**2
#        (x1, y1), (x2, y2) = x[:2], x[3:5]
#
#        x1_grad = -2 * ((x2-x1)+dist*np.sin(theta))
#        y1_grad = -2 * ((y2-y1)-dist*np.cos(theta))
#        theta_grad = 2 * dist * ((x2-x1)*np.cos(theta) + (y2-y1)*np.sin(theta))
#        x2_grad = 2 * ((x2-x1)+dist*np.sin(theta))
#        y2_grad = 2 * ((y2-y1)-dist*np.cos(theta))
#        return np.array([x1_grad, y1_grad, theta_grad, x2_grad, y2_grad]).reshape((1,5))
#
#    def resample(self, negated, time, plan):
#        res = OrderedDict()
#        attr_inds = OrderedDict()
#        a = 0
#        while  a < len(plan.actions) and plan.actions[a].active_timesteps[1] < time:
#            a += 1
#
#        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
#            return None, None
#
#        act = plan.actions[a]
#        x = self.get_param_vector(time).flatten()
#        theta = np.arctan2(*(x[3:5]-x[:2]))
#        rot = np.array([[np.cos(-theta), -np.sin(-theta)],
#                        [np.sin(-theta),  np.cos(-theta)]])
#        disp = rot.dot([0, -gripdist-dsafe])
#        offset = (x[3:5] + disp) - x[:2]
#        new_robot_pose = x[:2] + offset
#        new_can_pose = x[3:5] - offset
#        nsteps = 0
#        st = max(max(time-nsteps,1), act.active_timesteps[0]+1)
#        et = min(min(time+nsteps, plan.horizon-1), act.active_timesteps[1])
#        ref_st = max(max(time-nsteps,0), act.active_timesteps[0])
#        ref_et = min(min(time+nsteps, plan.horizon-1), act.active_timesteps[1])
#        add_to_attr_inds_and_res(time, attr_inds, res, self.r, [('pose', new_robot_pose), ('theta', np.array([theta]))])
#        for i in range(st, et):
#            dist =float(np.abs(i - time))
#            if i <= time:
#                inter_rp = (dist / nsteps) * self.r.pose[:, ref_st] + ((nsteps - dist) / nsteps) * new_robot_pose
#                inter_cp = (dist / nsteps) * self.can.pose[:, ref_st] + ((nsteps - dist) / nsteps) * new_can_pose
#            else:
#                inter_rp = (dist / nsteps) * self.r.pose[:, ref_et] + ((nsteps - dist) / nsteps) * new_robot_pose
#                inter_cp = (dist / nsteps) * self.can.pose[:, ref_et] + ((nsteps - dist) / nsteps) * new_can_pose
#
#            add_to_attr_inds_and_res(i, attr_inds, res, self.r, [('pose', inter_rp)])
#        return res, attr_inds


class AroundGraspAngle(NearGraspAngle):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        super(InGraspAngle, self).__init__(
            name, params, expected_param_types, env, sess, debug
        )
        self.tol = 2e-2
        self._rollout = True


class DoorNearGrasp(DoorInGrasp):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        super(DoorInGrasp, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=2
        )
        self.coeff = 5e-2
        self._rollout = True


class HandleAngleValid(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.r,) = params
        self.coeff = 1e0
        self.dist = 0.0
        self.dsafe = dsafe
        self.gripdist = gripdist
        self.door_len = 1.5
        self.door_wid = 0.2
        self.handle_size = 0.3
        self.handle_margin = 0.1
        self.hinge_pos = np.array([-1.0, 3.0])
        handle_pos = self.hinge_pos + [
            self.door_len,
            -self.door_wid / 2.0 - self.handle_size - self.handle_margin,
        ]
        self.handle_dist = np.linalg.norm(handle_pos - self.hinge_pos)
        rel_pos = handle_pos - self.hinge_pos
        self.base_theta = np.arctan2(rel_pos[1], rel_pos[0])
        if self.r.is_symbol():
            k = "value"
        else:
            k = "pose"

        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        (k, np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
            ]
        )
        angle_expr = Expr(self.f, self.grad)
        e = EqExpr(angle_expr, np.zeros((3, 1)))
        e = EqExpr(angle_expr, np.zeros((1, 1)))

        super(HandleAngleValid, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=1
        )

    def f(self, x):
        x = x.flatten()
        dist = self.handle_dist
        theta = self.base_theta + x[2]
        targ_loc = self.hinge_pos + [dist * np.cos(theta), dist * np.sin(theta)]
        can_loc = x[:2]
        return self.coeff * np.array(
            [[(targ_loc[0] - can_loc[0]) ** 2 + (targ_loc[1] - can_loc[1]) ** 2]]
        )

    def grad(self, x):
        x = x.flatten()
        dist = self.handle_dist
        theta = self.base_theta + x[2]

        (x1, y1), (x2, y2) = x[:2], self.hinge_pos

        x1_grad = -2 * ((x2 - x1) + dist * np.cos(theta))
        y1_grad = -2 * ((y2 - y1) + dist * np.sin(theta))
        theta_grad = -2 * dist * ((x2 - x1) * np.sin(theta) - (y2 - y1) * np.cos(theta))
        return self.coeff * np.array([x1_grad, y1_grad, theta_grad]).reshape((1, 3))

    def resample(self, negated, time, plan):
        res = OrderedDict()
        attr_inds = OrderedDict()
        a = 0
        # while  a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] < time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None

        act = plan.actions[a]
        x = self.get_param_vector(time).flatten()
        theta = x[2] + self.base_theta
        pos = self.hinge_pos + [
            self.handle_dist * np.cos(theta),
            self.handle_dist * np.sin(theta),
        ]
        add_to_attr_inds_and_res(
            time, attr_inds, res, self.r, [("pose", pos), ("theta", x[2:])]
        )
        return res, attr_inds


class LinearRetreat(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.r,) = params
        self.vel = RETREAT_DIST
        self.sign = 1.0
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
            ]
        )
        angle_expr = Expr(self.f, self.grad)
        e = EqExpr(angle_expr, np.zeros((2, 1)))

        super(LinearRetreat, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            priority=1,
            active_range=(0, 1),
        )
        self.hl_include = True

    def f(self, x):
        x = x.flatten()
        rot = np.array([[np.cos(x[5]), -np.sin(x[5])], [np.sin(x[5]), np.cos(x[5])]])
        # vel = np.linalg.norm(x[:5]-x[3:5])
        disp = self.sign * rot.dot([[0], [-self.vel]])
        x = x.reshape((-1, 1))
        return x[3:5] - disp - x[:2]  # .flatten()

    def grad(self, x):
        return -np.c_[np.eye(2), np.zeros((2, 1)), -np.eye(2), np.zeros((2, 1))]

    def resample(self, negated, time, plan):
        res = OrderedDict()
        attr_inds = OrderedDict()
        a = 0
        # while  a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] < time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None

        act = plan.actions[a]
        x = self.get_param_vector(time).flatten()
        # vel = np.linalg.norm(x[:2]-x[3:5])
        rot = np.array([[np.cos(x[5]), -np.sin(x[5])], [np.sin(x[5]), np.cos(x[5])]])
        disp = self.sign * rot.dot([0, -self.vel])
        offset = x[3:5] + disp - x[:2]
        new_robot_pose = x[:2] + offset
        st = max(max(time - 3, 1), act.active_timesteps[0] + 1)
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        ref_st = max(max(time - 3, 0), act.active_timesteps[0])
        ref_et = min(min(time + 3, plan.horizon - 2), act.active_timesteps[1] - 1)
        for i in range(ref_st, et + 1):
            dist = float(np.abs(i - time))
            if i <= time:
                inter_rp = (dist / 3.0) * self.r.pose[:, st] + (
                    (3.0 - dist) / 3.0
                ) * new_robot_pose
            else:
                inter_rp = (dist / 3.0) * self.r.pose[:, et] + (
                    (3.0 - dist) / 3.0
                ) * new_robot_pose

            add_to_attr_inds_and_res(i, attr_inds, res, self.r, [("pose", inter_rp)])
        return res, attr_inds

    def test(self, time, negated=False, tol=1e-4):
        if time == 0:
            return True
        return super(LinearRetreat, self).test(time, negated, tol)


class LinearApproach(LinearRetreat):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        super(LinearApproach, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.sign = -1.0


class RobotRetreat(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.r, self.grasp = params
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
            ]
        )
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.array([[1, 0, 0, -1, 0, 0], [0, 1, 0, 0, -1, 0]])
        b = np.zeros((2, 1))
        b[1] = 2 * self.r.radius + dsafe
        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((2, 1)))

        super(RobotRetreat, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            priority=-2,
            active_range=(0, 1),
        )


class RobotApproach(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.r, self.grasp = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.grasp, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.array([[-1, 0, 1.5, 0, 1, 0, 0, 0], [0, -1, 0, 1.5, 0, 1, 0, 0]])
        b = np.zeros((2, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((2, 1)))

        super(RobotApproach, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            priority=-2,
            active_range=(0, 1),
        )


class GraspValid(ExprPredicate):

    # GraspValid RobotPose Target Grasp

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.rp, self.target, self.grasp = params
        attr_inds = OrderedDict(
            [
                (self.rp, [("value", np.array([0, 1], dtype=np.int))]),
                (self.target, [("value", np.array([0, 1], dtype=np.int))]),
                (self.grasp, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.array([[1, 0, -1, 0, -1, 0], [0, 1, 0, -1, 0, -1]])
        b = np.zeros((2, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((2, 1)))

        super(GraspValid, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=0
        )


class RobotStationary(ExprPredicate):

    # Stationary, Can

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.c,) = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
        super(RobotStationary, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )
        self.hl_include = True

    def hl_test(self, time, negated=False, tol=None):
        return True


class Stationary(ExprPredicate):

    # Stationary, Can

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.c,) = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
        super(Stationary, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class StationaryDoor(ExprPredicate):

    # Stationary, Can

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.c,) = params
        attr_inds = OrderedDict([(self.c, [("theta", np.array([0], dtype=np.int))])])
        A = np.array(
            [
                [
                    1,
                    -1,
                ]
            ]
        )
        b = np.zeros((1, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((1, 1)))
        super(StationaryDoor, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class StationaryDoorPos(ExprPredicate):

    # Stationary, Can

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.c,) = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
        super(StationaryDoorPos, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class StationaryRot(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.c,) = params
        attr_inds = OrderedDict([(self.c, [("theta", np.array([0], dtype=np.int))])])
        A = np.array([[1, -1]])
        b = np.zeros((1, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((1, 1)))
        super(StationaryRot, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


"""
class StationaryRot(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        self.c,  = params
        attr_inds = OrderedDict([(self.c, [("theta", np.array([0], dtype=np.int))])])
        def f(x):
            return angle_diff(x[1], x[0])

        def grad(x):
            return np.array([-1, 1]).reshape((1,2))

        angleExpr = Expr(f, grad)
        e = EqExpr(angleExpr, np.zeros((1,1)))
        super(StationaryRot, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=0)
"""


class AtRot(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.r, self.rot = params
        attr_inds = OrderedDict(
            [
                (self.r, [("theta", np.array([0], dtype=np.int))]),
                (self.rot, [("value", np.array([0], dtype=np.int))]),
            ]
        )
        A = np.array([[1, -1]])
        b = np.zeros((1, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((1, 1)))
        super(AtRot, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0,),
            priority=-2,
        )


class StationaryNEq(ExprPredicate):

    # StationaryNEq, Can, Can
    # Assuming robot only holding one object,
    # it checks whether the can in the first argument is stationary
    # if that first can is not the second can which robot is holding

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.c, self.c_held = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))])])
        if self.c.name == self.c_held.name:
            A = np.zeros((1, 4))
            b = np.zeros((1, 1))
        else:
            A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
            b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryNEq, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class StationaryW(ExprPredicate):

    # StationaryW, Wall(Obstacle)

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.w,) = params
        attr_inds = OrderedDict([(self.w, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryW, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


# class IsMP(ExprPredicate):
#
#    # IsMP Robot
#
#    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False, dmove=dmove):
#        self.r, = params
#        ## constraints  |x_t - x_{t+1}| < dmove
#        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
#        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int))])])
#        A = np.array([[1, 0, -1, 0],
#                      [0, 1, 0, -1],
#                      [-1, 0, 1, 0],
#                      [0, -1, 0, 1]])
#        b = np.zeros((4, 1))
#        e = LEqExpr(AffExpr(A, b), dmove*np.ones((4, 1)))
#        super(IsMP, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2)


class IsMP(ExprPredicate):

    # IsMP Robot

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        sess=None,
        debug=False,
        dmove=dmove,
    ):
        (self.r,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                )
            ]
        )
        A = np.array(
            [
                [1, 0, 0, -1, 0, 0],
                [0, 1, 0, 0, -1, 0],
                [0, 0, 1, 0, 0, -1],
                [-1, 0, 0, 1, 0, 0],
                [0, -1, 0, 0, 1, 0],
                [0, 0, -1, 0, 0, 1],
            ]
        )
        b = np.zeros((6, 1))
        drot = np.pi / 3.0
        e = LEqExpr(
            AffExpr(A, b),
            np.array([dmove, dmove, drot, dmove, dmove, drot]).reshape((6, 1)),
        )
        super(IsMP, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )
        self._nonrollout = True


class DoorIsMP(ExprPredicate):

    # IsMP Robot

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        sess=None,
        debug=False,
        dmove=dmove,
    ):
        (self.r,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict([(self.r, [("theta", np.array([0], dtype=np.int))])])
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        b = np.zeros((2, 1))
        drot = np.pi / 8.0
        e = LEqExpr(AffExpr(A, b), np.array([drot, drot]).reshape((2, 1)))
        super(DoorIsMP, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class VelWithinBounds(At):

    # RobotAt Robot Can

    def __init__(self, name, params, expected_param_types, env=None, sess=None):
        ## At Robot RobotPose
        self.r, self.c = params
        attr_inds = OrderedDict([(self.r, [("vel", np.array([0, 1], dtype=np.int))])])

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((4, 1))
        val = dmove * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class AccWithinBounds(At):

    # RobotAt Robot Can

    def __init__(self, name, params, expected_param_types, env=None, sess=None):
        ## At Robot RobotPose
        self.r, self.c = params
        attr_inds = OrderedDict([(self.r, [("acc", np.array([0, 1], dtype=np.int))])])

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((4, 1))
        val = 2.5e1 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class VelValid(ExprPredicate):

    # VelValid Robot

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        sess=None,
        debug=False,
        dmove=dmove,
    ):
        (self.r,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("vel", np.array([0, 1], dtype=np.int)),
                    ],
                ),
            ]
        )
        A = np.array(
            [
                [-1, 0, 1, 0, 1, 0, 0, 0],
                [0, -1, 0, 1, 0, 1, 0, 0],
            ]
        )
        b = np.zeros((4, 1))

        e = LEqExpr(AffExpr(A, b), dmove * np.ones((4, 1)))
        super(VelValid, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class Decelerating(ExprPredicate):
    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        sess=None,
        debug=False,
        dmove=dmove,
    ):
        (self.r,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (self.r, [("vel", np.array([0, 1], dtype=np.int))]),
            ]
        )
        A = np.array(
            [
                [-1, 0, 1, 0],
                [0, -1, 0, 1],
            ]
        )
        b = np.zeros((4, 1))

        e = LEqExpr(AffExpr(A, b), b.copy())
        super(VelValid, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class Accelerating(ExprPredicate):
    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        sess=None,
        debug=False,
        dmove=dmove,
    ):
        (self.r,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (self.r, [("vel", np.array([0, 1], dtype=np.int))]),
            ]
        )
        A = np.array(
            [
                [1, 0, -1, 0],
                [0, 1, 0, -1],
            ]
        )
        b = np.zeros((4, 1))

        e = LEqExpr(AffExpr(A, b), b.copy())
        super(VelValid, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class VelValid(ExprPredicate):

    # VelValid Robot

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        sess=None,
        debug=False,
        dmove=dmove,
    ):
        (self.r,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("vel", np.array([0, 1], dtype=np.int)),
                    ],
                ),
            ]
        )
        A = np.array(
            [
                [-1, 0, 1, 0, 1, 0, 0, 0],
                [0, -1, 0, 1, 0, 1, 0, 0],
            ]
        )
        b = np.zeros((4, 1))

        e = LEqExpr(AffExpr(A, b), dmove * np.ones((4, 1)))
        super(VelValid, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class AccValid(VelValid):

    # AccValid Robot

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        sess=None,
        debug=False,
        dmove=dmove,
    ):
        super(AccValid, self).__init__(
            name, params, expected_param_types, env, debug, dmove
        )
        self.attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("vel", np.array([0, 1], dtype=np.int)),
                        ("acc", np.array([0, 1], dtype=np.int)),
                    ],
                ),
            ]
        )


class ScalarVelValid(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.r,) = params
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                        ("vel", np.array([0], dtype=np.int)),
                    ],
                ),
            ]
        )

        def f(x):
            x = x.flatten()
            curvel = np.linalg.norm(x[4:6] - x[:2])
            targ = [-curvel * np.sin(x[2]), curvel * np.cos(x[2])]
            curdisp = x[4:6] - x[:2]
            dist1 = (targ[0] - curdisp[0]) ** 2 + (targ[1] - curdisp[1]) ** 2
            dist2 = (targ[0] + curdisp[0]) ** 2 + (targ[1] + curdisp[1]) ** 2
            if dist2 < dist1:
                curvel *= -1
            return np.array([x[7] - curvel]).reshape((1, 1))

        def grad(x):
            curvel = np.linalg.norm(x[4:6] - x[:2])
            targ = [-curvel * np.sin(x[2]), curvel * np.cos(x[2])]
            curdisp = x[4:6] - x[:2]
            dist1 = (targ[0] - curdisp[0]) ** 2 + (targ[1] - curdisp[1]) ** 2
            dist2 = (targ[0] + curdisp[0]) ** 2 + (targ[1] + curdisp[1]) ** 2
            if dist2 < dist1:
                curvel *= -1
            gradx1, grady1, gradx2, grady2 = 0, 0, 0, 0
            if np.abs(curvel) > 1e-3:
                gradx1 = (x[4] - x[0]) / curvel
                grady1 = (x[5] - x[1]) / curvel
                gradx2 = -(x[4] - x[0]) / curvel
                grady2 = -(x[5] - x[1]) / curvel
            return np.array([gradx1, grady1, 0, 0, gradx2, grady2, 0, 1]).reshape(
                (1, 8)
            )

        angle_expr = Expr(f, grad)
        e = EqExpr(angle_expr, np.zeros((1, 1)))

        super(ScalarVelValid, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            priority=3,
            active_range=(0, 1),
        )

    def resample(self, negated, time, plan):
        return None, None
        res = OrderedDict()
        attr_inds = OrderedDict()
        a = 0
        # while  a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] < time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None

        act = plan.actions[a]
        x = self.get_param_vector(time).flatten()
        disp = x[4:6] - x[:2]
        curvel = np.linalg.norm(x[4:6] - x[:2])
        curtheta = np.arctan2(*(x[4:6] - x[:2]))
        if np.abs(angle_diff(curtheta, x[2])) > 3 * np.pi / 4:
            curvel *= -1
        add_to_attr_inds_and_res(
            time + 1, attr_inds, res, self.r, [("vel", np.array([curvel]))]
        )
        return res, attr_inds


class InDoorAngle(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.r, self.c = params
        attr_inds = OrderedDict(
            [
                (self.r, [("theta", np.array([0], dtype=np.int))]),
                (self.c, [("theta", np.array([0], dtype=np.int))]),
            ]
        )
        A = np.array([[1.0, -1.0]])
        b = np.zeros((1, 1))
        e = EqExpr(AffExpr(A, b), np.pi / 2.0 * np.ones((1, 1)))
        super(InDoorAngle, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.hl_include = True
        self._rollout = True


class ThetaDirValid(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.r,) = params
        for attr in ['forward', 'reverse']:
            if not hasattr(self, attr):
                setattr(self, attr, True)

        self.coeff = 1e0
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int)),
                                           ("theta", np.array([0], dtype=np.int))]),
                                ])

        self.torch_func = ThetaDir(use_forward=self.forward,
                                   use_reverse=self.reverse)
        angle_expr = Expr(self.f, self.grad)
        e = EqExpr(angle_expr, np.zeros((1, 1)))

        super(ThetaDirValid, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            priority=2,
            active_range=(0, 1),
        )


    def f(self, x):
        return self.coeff * np.array([[self.torch_func.eval_f(x)]])


    def grad(self, x):
        return self.coeff * self.torch_func.eval_grad(x).reshape((1,-1))


    def resample(self, negated, time, plan):
        res = OrderedDict()
        attr_inds = OrderedDict()
        a = 0
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
            # while  a < len(plan.actions) and plan.actions[a].active_timesteps[1] < time:
            a += 1

        if a >= len(plan.actions):  # or time == plan.actions[a].active_timesteps[0]:
            return None, None

        act = plan.actions[a]
        x = self.get_param_vector(time).flatten()
        theta = x[2]
        disp = x[4:6] - x[:2]
        dist = np.linalg.norm(disp)
        targ_disp = [-dist * np.sin(theta), dist * np.cos(theta)]
        pose1 = x[4:6] - targ_disp
        pose2 = x[:2] + targ_disp
        cur_theta = np.arctan2(disp[0], disp[1])
        opp_theta = opposite_angle(cur_theta)
        if not self.forward and np.abs(angle_diff(cur_theta, theta)) > np.abs(
            angle_diff(opp_theta, theta)
        ):
            cur_theta = opp_theta
        elif self.reverse:
            cur_theta = opp_theta
        add_to_attr_inds_and_res(
            time,
            attr_inds,
            res,
            self.r,
            [("pose", pose1), ("theta", np.array([cur_theta]))],
        )
        add_to_attr_inds_and_res(time + 1, attr_inds, res, self.r, [("pose", pose2)])

        return res, attr_inds
        theta_off = -angle_diff(theta, cur_theta)
        rot = np.array(
            [
                [np.cos(theta_off), -np.sin(theta_off)],
                [np.sin(theta_off), np.cos(theta_off)],
            ]
        )
        new_disp = rot.dot(disp)
        new_robot_pose_1 = x[:2] + new_disp
        new_robot_pose_2 = x[4:6] - new_disp  # (new_disp + disp) / 2.
        new_theta = np.array([add_angle(cur_theta, theta_off / 2.0)])
        # add_to_attr_inds_and_res(time, attr_inds, res, self.r, [('pose', new_robot_pose)])
        # add_to_attr_inds_and_res(time+1, attr_inds, res, self.r, [('theta', new_theta)])
        nsteps = 1
        st = max(max(time - nsteps, 1), act.active_timesteps[0] + 1)
        et = min(min(time + nsteps, plan.horizon - 1), act.active_timesteps[1])
        ref_st = max(max(time - nsteps, 0), act.active_timesteps[0])
        ref_et = min(min(time + nsteps, plan.horizon - 1), act.active_timesteps[1])
        poses = []
        for i in range(st, et + 1):
            if i <= time:
                dist = float(np.abs(i - time))
                inter_rp = (dist / nsteps) * self.r.pose[:, ref_st] + (
                    (nsteps - dist) / nsteps
                ) * new_robot_pose_2
                inter_theta = (dist / nsteps) * self.r.pose[:, ref_st] + (
                    (nsteps - dist) / nsteps
                ) * new_theta
            else:
                dist = float(np.abs(i - time - 1))
                inter_rp = (dist / nsteps) * self.r.pose[:, ref_et] + (
                    (nsteps - dist) / nsteps
                ) * new_robot_pose_1
                inter_theta = (dist / nsteps) * self.r.pose[:, ref_et] + (
                    (nsteps - dist) / nsteps
                ) * new_theta
            add_to_attr_inds_and_res(i, attr_inds, res, self.r, [("pose", inter_rp)])
            poses.append(inter_rp)
            if len(poses) > 1:
                newtheta = np.arctan2(*(poses[-1] - poses[-2]))
                curtheta = self.r.theta[0, time]
                opp_theta = opposite_angle(newtheta)
                theta = newtheta
                if self.reverse or (
                    not self.forward
                    and np.abs(angle_diff(curtheta, newtheta))
                    > np.abs(angle_diff(opp_theta, curtheta))
                ):
                    theta = opp_theta
                if i - 1 != time:
                    add_to_attr_inds_and_res(
                        i - 1, attr_inds, res, self.r, [("theta", np.array([theta]))]
                    )
        return res, attr_inds


class ForThetaDirValid(ThetaDirValid):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.forward = True
        super(ForThetaDirValid, self).__init__(
            name, params, expected_param_types, env, debug
        )


class RevThetaDirValid(ThetaDirValid):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.reverse = True
        super(RevThetaDirValid, self).__init__(
            name, params, expected_param_types, env, debug
        )


class ColObjPred(CollisionPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, coeff=1e3, debug=False
    ):
        self._env = env
        self.hl_ignore = True
        self.r, self.c = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
        }

        self.rs_scale = RS_SCALE
        self.radius = self.c.geom.radius + 2.5
        self.torch_func = GaussianBump(radius=self.radius, dim=2)

        self.col_ts = 2
        self.coeff = coeff
        self.neg_coeff = self.coeff
        col_expr = Expr(self.f, self.grad, self.hess)
        val = np.ones((1, 1))
        e = EqExpr(col_expr, val)

        neg_val = np.zeros((1, 1))
        col_expr_neg = Expr(self.f_neg, self.grad_neg, self.hess_neg)
        self.neg_expr = LEqExpr(col_expr_neg, neg_val)

        super(ColObjPred, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            active_range=(0, 1),
        )
        self.dsafe = 2.0


    def f(self, x):
        xs = [
            float(self.col_ts - t) / self.col_ts * x[:4] + float(t) / self.col_ts * x[4:]
            for t in range(self.col_ts + 1)
        ]
        vals = []
        for i, pt in enumerate(xs):
            vals.append(self.torch_func.eval_f(pt))

        return self.coeff * np.array([[np.sum(vals)]])


    def grad(self, x):
        xs = [
            float(self.col_ts - t) / self.col_ts * x[:4] + float(t) / self.col_ts * x[4:]
            for t in range(self.col_ts + 1)
        ]
        vals = []
        for i, pt in enumerate(xs):
            curcoeff = float(self.col_ts - i) / self.col_ts
            v = self.torch_func.eval_grad(pt).reshape((1,-1))
            vals.append(np.c_[curcoeff * v, (1 - curcoeff) * v])
        return self.coeff * np.sum(vals, axis=0)


    def hess(self, x):
        xs = [
            float(self.col_ts - t) / self.col_ts * x[:4] + float(t) / self.col_ts * x[4:]
            for t in range(self.col_ts + 1)
        ]
        vals = []
        for i, pt in enumerate(xs):
            curcoeff = float(self.col_ts - i) / self.col_ts
            v = self.torch_func.eval_hess(pt)
            new_v = np.r_[
                np.c_[curcoeff * v, np.zeros((4, 4))],
                np.c_[np.zeros((4, 4)), (1 - curcoeff) * v],
            ]
            vals.append(new_v.reshape((8, 8)))
        return np.sum(vals, axis=0).reshape((8, 8))


    def f_neg(self, x):
        return self.neg_coeff / self.coeff * self.f(x)


    def grad_neg(self, x):
        return self.neg_coeff / self.coeff * self.grad(x)


    def hess_neg(self, x):
        return self.neg_coeff / self.coeff * self.hess(x)


class BoxObjPred(CollisionPredicate):
    pass


class DoorColObjPred(CollisionPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, coeff=1e3, debug=False
    ):
        self._env = env
        self.hl_ignore = True
        self.door, self.robot = params
        attr_inds = OrderedDict(
            [
                (
                    self.door,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("theta", np.array([0], dtype=np.int)),
                    ],
                ),
                (self.robot, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.door: self.lazy_spawn_or_body(
                self.door, self.door.name, self.door.geom
            ),
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            ),
        }

        self.rs_scale = RS_SCALE
        self.radius = 2.3
        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        self.coeff = coeff
        self.neg_coeff = coeff
        self.neg_grad_coeff = 1e-1  # 1e-3
        col_expr = Expr(self.f, self.grad)
        val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(
            lambda x: self.coeff * self.f(x),
            lambda x: self.coeff * self.grad(x),
            lambda x: self.coeff * self.hess_neg(x),
        )
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(DoorColObjPred, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            active_range=(0, 1),
        )
        self.dsafe = 2.0

    def f(self, x):
        xs = [
            float(self.col_ts - t) / self.col_ts * x[:5] + float(t) / self.col_ts * x[5:]
            for t in range(self.col_ts + 1)
        ]
        if USE_TF:
            cur_tensor = get_tf_graph("door_bump_out")
            in_tensor = get_tf_graph("door_bump_in")
            radius_tensor = get_tf_graph("door_bump_radius")
            len_tensor = get_tf_graph("door_len")
            vals = []
            for i in range(self.col_ts + 1):
                pt = xs[i]
                true_pt = pt[:2] + [1.5 * np.cos(x[2]), 1.5 * np.sin(x[2])]
                if np.sum((true_pt - pt[3:]) ** 2) > (self.radius - 1e-3) ** 2:
                    vals.append(0)
                else:
                    val = np.array(
                        [
                            TF_SESS[0].run(
                                cur_tensor,
                                feed_dict={
                                    in_tensor: pt,
                                    radius_tensor: self.radius ** 2,
                                    len_tensor: 1.5,
                                },
                            )
                        ]
                    )
                    vals.append(val)
            return np.sum(vals, axis=0)

        col_vals = self.distance_from_obj(x)[0]
        col_vals = np.clip(col_vals, 0.0, 4)
        return -col_vals
        # return -self.distance_from_obj(x)[0] # twostep_f([x[:4]], self.distance_from_obj, 2, pts=1)

    def grad(self, x):
        xs = [
            float(self.col_ts - t) / self.col_ts * x[:5] + float(t) / self.col_ts * x[5:]
            for t in range(self.col_ts + 1)
        ]
        if USE_TF:
            cur_grads = get_tf_graph("door_bump_grads")
            in_tensor = get_tf_graph("door_bump_in")
            radius_tensor = get_tf_graph("door_bump_radius")
            len_tensor = get_tf_graph("door_len")
            vals = []
            for i in range(self.col_ts + 1):
                pt = xs[i]
                true_pt = pt[:2] + [1.5 * np.cos(x[2]), 1.5 * np.sin(x[2])]
                if np.sum((true_pt - pt[3:]) ** 2) > (self.radius - 1e-3) ** 2:
                    vals.append(np.zeros((1, 10)))
                else:
                    v = (
                        TF_SESS[0]
                        .run(
                            cur_grads,
                            feed_dict={
                                in_tensor: pt,
                                len_tensor: 1.5,
                                radius_tensor: self.radius ** 2,
                            },
                        )
                        .T
                    )
                    v[np.isnan(v)] = 0.0
                    v[np.isinf(v)] = 0.0
                    curcoeff = float(self.col_ts - i) / self.col_ts
                    vals.append(np.c_[curcoeff * v, (1 - curcoeff) * v])
            return np.sum(vals, axis=0)
        return (
            -self.coeff * self.distance_from_obj(x)[1]
        )  # twostep_f([x[:4]], self.distance_from_obj, 2, pts=1, grad=True)

    def f_neg(self, x):
        return -self.neg_coeff * self.f(x)

    def grad_neg(self, x):
        return -self.neg_grad_coeff * self.grad(x)

    def hess_neg(self, x):
        xs = [
            float(self.col_ts - t) / self.col_ts * x[:5] + float(t) / self.col_ts * x[5:]
            for t in range(self.col_ts + 1)
        ]
        if USE_TF:
            cur_hess = get_tf_graph("door_bump_hess")
            in_tensor = get_tf_graph("door_bump_in")
            radius_tensor = get_tf_graph("door_bump_radius")
            len_tensor = get_tf_graph("door_len")
            vals = []
            for i in range(self.col_ts + 1):
                pt = xs[i]
                true_pt = pt[:2] + [1.5 * np.cos(x[2]), 1.5 * np.sin(x[2])]
                if np.sum((true_pt - pt[3:]) ** 2) > (self.radius - 1e-3) ** 2:
                    vals.append(np.zeros((10, 10)))
                else:
                    v = TF_SESS[0].run(
                        cur_hess,
                        feed_dict={
                            in_tensor: pt,
                            len_tensor: 1.5,
                            radius_tensor: self.radius ** 2,
                        },
                    )
                    v[np.isnan(v)] = 0.0
                    v[np.isinf(v)] = 0.0
                    v = v.reshape((5, 5))
                    curcoeff = float(self.col_ts - i) / self.col_ts
                    new_v = np.r_[
                        np.c_[curcoeff * v, np.zeros((5, 5))],
                        np.c_[np.zeros((5, 5)), (1 - curcoeff) * v],
                    ]
                    vals.append(new_v.reshape((10, 10)))
            return np.sum(vals, axis=0).reshape((10, 10))
        j = self.grad(x)
        return j.T.dot(j)
