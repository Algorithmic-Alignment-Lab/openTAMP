from opentamp.core.internal_repr.predicate import Predicate
from opentamp.core.internal_repr.plan import Plan
from opentamp.core.util_classes.common_predicates import ExprPredicate
from opentamp.core.util_classes.namo_predicates import NEAR_TOL
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
from errors_exceptions import PredicateException
from sco_py.expr import Expr, AffExpr, EqExpr, LEqExpr
import numpy as np
import tensorflow as tf
import sys
import traceback

import pybullet as p

from collections import OrderedDict

from pma.ll_solver_gurobi import NAMOSolver


"""
This file implements the predicates for the 2D NAMO domain.
"""

dsafe = 1e-3
# dmove = 1.1e0 # 5e-1
dmove = 1.5e0  # 5e-1
contact_dist = 2e-1  # dsafe
gripdist = 0.55  # 75

RS_SCALE = 0.5
N_DIGS = 5
GRIP_VAL = 0.4
COL_TS = 10  # 3
N_COLS = 8
RETREAT_DIST = 2.0


ATTRMAP = {
    "Robot": (
        ("pose", np.array(list(range(2)), dtype=np.int_)),
        ("joint1", np.array(list(range(1)), dtype=np.int_)),
        ("joint2", np.array(list(range(1)), dtype=np.int_)),
        ("wrist", np.array(list(range(1)), dtype=np.int_)),
        ("gripper", np.array(list(range(1)), dtype=np.int_)),
        ("ee_pose", np.array(list(range(2)), dtype=np.int_)),
    ),
    "Can": (("pose", np.array(list(range(2)), dtype=np.int_)),),
    "Target": (("value", np.array(list(range(2)), dtype=np.int_)),),
    "RobotPose": (
        ("value", np.array(list(range(2)), dtype=np.int_)),
        ("joint1", np.array(list(range(1)), dtype=np.int_)),
        ("joint2", np.array(list(range(1)), dtype=np.int_)),
        ("wrist", np.array(list(range(1)), dtype=np.int_)),
        ("gripper", np.array(list(range(1)), dtype=np.int_)),
    ),
    "Obstacle": (("pose", np.array(list(range(2)), dtype=np.int_)),),
    "Grasp": (("value", np.array(list(range(2)), dtype=np.int_)),),
    "Rotation": (("value", np.array(list(range(1)), dtype=np.int_)),),
}


USE_TF = True
if USE_TF:
    tf_cache = {}

    def get_tf_graph(tf_name):
        if tf_name not in tf_cache:
            init_tf_graph()
        return tf_cache[tf_name]

    def init_tf_graph():
        linklen = 3.0
        tf_jnts = tf.compat.v1.placeholder(float, (4,), name="jnts")
        tf_theta1 = tf_jnts[0]
        tf_theta2 = tf_jnts[1]
        tf_theta3 = tf_jnts[2]
        tf_grip = tf_jnts[3]
        tf_cache["grip"] = tf_grip
        tf_cache["jnts"] = tf_jnts
        tf_dist = tf.compat.v1.placeholder(float, (), name="dist")
        tf_cache["dist"] = tf_dist

        tf_ee_theta = tf_theta1 + tf_theta2 + tf_theta3
        tf_cache["ee_theta"] = tf_ee_theta
        tf_joint2_x = -linklen * tf.sin(tf_theta1)
        tf_joint2_y = linklen * tf.cos(tf_theta1)
        tf_ee_x = tf_joint2_x - linklen * tf.sin(tf_theta1 + tf_theta2)
        tf_ee_y = tf_joint2_y + linklen * tf.cos(tf_theta1 + tf_theta2)

        tf_xy_pos = tf.compat.v1.placeholder(float, (2,), name="xy_pos")
        tf_cache["xy_pos"] = tf_xy_pos
        tf_ee_xy = tf.stack([tf_ee_x, tf_ee_y], axis=0)
        tf_cache["ee_xy"] = tf_ee_xy
        tf_ee_disp = tf.concat(
            [[tf_ee_x, tf_ee_y] - tf_xy_pos, tf_xy_pos - [tf_ee_x, tf_ee_y]], axis=0
        )
        tf_cache["ee_disp"] = tf_ee_disp
        tf_rot = tf.compat.v1.placeholder(float, (1,), name="rot")
        tf_cache["rot"] = tf_rot
        tf_rot_disp = tf.concat([tf_rot - tf_ee_theta, tf_ee_theta - tf_rot], axis=0)
        tf_cache["rot_disp"] = tf_rot_disp

        tf_obj_pos = tf.compat.v1.placeholder(float, (2,), name="obj_pos")
        tf_ee_disp = tf_obj_pos - tf_xy_pos
        tf_cache["obj_pose"] = tf_obj_pos
        tf_ee_grasp = tf.stack(
            [
                tf_ee_x - tf_dist * tf.sin(tf_ee_theta),
                tf_ee_y + tf_dist * tf.cos(tf_ee_theta),
            ],
            axis=0,
        )
        tf_cache["ee_grasp"] = tf_ee_grasp
        tf_ingrasp = tf.reduce_sum((tf_obj_pos - tf_ee_grasp) ** 2)
        tf_cache["ingrasp"] = tf_ingrasp
        tf_cache["ingrasp_gradients"] = tf.gradients(
            tf_cache["ingrasp"], [tf_cache["jnts"], tf_cache["obj_pose"]]
        )

        tf_cache["bump_in"] = tf.compat.v1.placeholder(float, (4, 1), name="bump_in")
        tf_cache["bump_radius"] = tf.compat.v1.placeholder(float, (), name="bump_radius")
        pos1 = tf_cache["bump_in"][:2]
        pos2 = tf_cache["bump_in"][2:]
        tf_cache["bump_diff"] = tf.reduce_sum((pos1 - pos2) ** 2)
        tf_cache["bump_out"] = tf.exp(
            -1.0
            * tf_cache["bump_radius"]
            / (tf_cache["bump_radius"] - tf_cache["bump_diff"])
        )
        tf_cache["bump_grads"] = tf.gradients(
            tf_cache["bump_out"], tf_cache["bump_in"]
        )[0]
        tf_cache["bump_hess"] = tf.hessians(tf_cache["bump_out"], tf_cache["bump_in"])[
            0
        ]


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
                    next_pos[3] = -0.4 if min(xs[0][3], xs[1][3]) < 0 else 0
            else:
                next_pos = xs[0]
            cur_jac = dist(next_pos)[1]
            filldim = dim - cur_jac.shape[1]
            # cur_jac = np.c_[cur_jac[:,:2], np.zeros((N_COLS, filldim)), cur_jac[:,2:]]
            # res.append(dist(next_pos)[1])
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
                    next_pos[3] = -0.4 if min(xs[0][3], xs[1][3]) < 0 else 0
            else:
                next_pos = xs[0]
            res.append(dist(next_pos)[0])
        return np.concatenate(res, axis=0)


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
        self._debug = debug
        # if self._debug:
        #     self._env.SetViewer("qtcoin")

        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1

        self._cache = {}
        self.n_cols = N_COLS
        self.check_aabb = False

        super(CollisionPredicate, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=active_range,
            priority=priority,
        )

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
        else:
            raise Exception("Should not call this without the robot!")
        pose1 = x[4:6]
        b0.set_dof(
            {
                "joint1": x[0],
                "joint2": x[1],
                "wrist": x[2],
                "left_grip": x[3],
                "right_grip": x[3],
            }
        )
        b1.set_pose(pose1)
        return x[:4], pose1

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

    def distance_from_obj(self, x, n_steps=0):
        pose0, pose1 = self.set_pos(x)
        p0 = self.params[self.ind0]
        p1 = self.params[self.ind1]
        b0 = self._param_to_body[p0]
        b1 = self._param_to_body[p1]
        vals = np.zeros((self.n_cols, 1))
        jacs = np.zeros((self.n_cols, 6))
        # if self.check_aabb:
        #    vals, jacs = self._check_robot_aabb(b0, b1)

        collisions = p.getClosestPoints(b0.body_id, b1.body_id, contact_dist)[
            : self.n_cols
        ]
        col_val, jac01 = self._calc_grad_and_val(
            p0.name, p1.name, pose0, pose1, collisions
        )
        final_val = col_val
        final_jac = jac01
        for i in range(len(final_val)):
            if final_val[i] < vals[i]:
                final_val[i] = vals[i]
                final_jac[i] = jacs[i]
        return final_val, final_jac

    # @profile
    def _calc_grad_and_val(self, name0, name1, pose0, pose1, collisions):
        vals = np.zeros((self.n_cols, 1))
        jacs = np.zeros((self.n_cols, 6))
        pose1 = pose1.flatten()

        val = -1 * float("inf")
        results = []
        n_cols = len(collisions)
        assert n_cols <= self.n_cols
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
            if linkRobot not in b0._geom.col_links:
                continue
            distance = c[8]  # c.contactDistance
            normal = np.array(c[7])  # c.contactNormalOnB # Pointing towards A
            results.append((pt0, pt1, distance))
            vals[i, 0] = self.dsafe - distance
            axis = [0, 0, 1]

            jac = []
            for jnt in [0, 2, 4]:
                jntpos = p.getLinkState(b0.body_id, jnt)[0]
                jac.append(
                    -np.dot(normal, np.cross(axis, np.r_[pose1, [0.5]] - jntpos))
                )
            jacs[i] = np.r_[jac, [0, 0, 0]]
            jacs[i, -2:] = normal[:2]
        return np.array(vals).reshape((self.n_cols, 1)), np.array(jacs).reshape(
            (self.n_cols, 6)
        )

    # @profile
    def _calc_obj_grad_and_val(self, name0, name1, pose0, pose1, collisions):
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
            vals[i, 0] = self.dsafe - distance
            jacs[i, :2] = -1 * normal[:2]
            jacs[i, 2:] = normal[:2]
        return np.array(vals).reshape((self.n_cols, 1)), np.array(jacs).reshape(
            (self.n_cols, 4)
        )

    def _plot_collision(self, ptA, ptB, distance):
        self.handles = []
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                self.handles.append(
                    self._env.drawarrow(p1=ptA, p2=ptB, linewidth=0.01, color=(1, 0, 0))
                )
            else:
                self.handles.append(
                    self._env.drawarrow(p1=ptA, p2=ptB, linewidth=0.01, color=(0, 0, 0))
                )


class HLPoseUsed(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None):
        ## At Can Target
        self.pose = params[0]
        if self.pose.is_symbol():
            k = "value"
        else:
            k = "pose"
        attr_inds = OrderedDict([(self.pose, [(k, np.array([0, 1], dtype=np.int_))])])

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
    def __init__(self, name, params, expected_param_types, env=None):
        self.pose = params[0]
        if self.pose.is_symbol():
            k = "value"
        else:
            k = "pose"
        attr_inds = OrderedDict([(self.pose, [(k, np.array([0, 1], dtype=np.int_))])])

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
    def __init__(self, name, params, expected_param_types, env=None):
        self.pose = params[0]
        if self.pose.is_symbol():
            k = "value"
        else:
            k = "pose"
        attr_inds = OrderedDict([(self.pose, [(k, np.array([0, 1], dtype=np.int_))])])

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

    def __init__(self, name, params, expected_param_types, env=None, sess=None):
        ## At Robot RobotPose
        self.r, self.c, self.g = params
        k = "pose" if not self.r.is_symbol() else "value"
        attr_inds = OrderedDict(
            [
                (self.r, [(k, np.array([0, 1], dtype=np.int_))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int_))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int_))]),
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
    def __init__(self, name, params, expected_param_types, env=None, sess=None):
        ## At Can Target
        self.can, self.targ = params
        attr_inds = OrderedDict(
            [
                (self.can, [("pose", np.array([0, 1], dtype=np.int_))]),
                (self.targ, [("value", np.array([0, 1], dtype=np.int_))]),
            ]
        )

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )


class AtNEq(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None):
        ## At Can Target
        self.can, self.eq, self.targ = params
        attr_inds = OrderedDict(
            [
                (self.can, [("pose", np.array([0, 1], dtype=np.int_))]),
                (self.targ, [("value", np.array([0, 1], dtype=np.int_))]),
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


class RobotInBounds(At):

    # RobotAt Robot RobotPose

    def __init__(self, name, params, expected_param_types, env=None, sess=None):
        ## At Robot RobotPose
        (self.r,) = params
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("joint1", np.array([0], dtype=np.int_)),
                        ("joint2", np.array([0], dtype=np.int_)),
                        ("wrist", np.array([0], dtype=np.int_)),
                    ],
                ),
            ]
        )

        A = np.r_[np.eye(3), -np.eye(3)]
        b = np.zeros((6, 1))
        val = np.r_[
            self.r.geom.upper_bounds[:3], -self.r.geom.lower_bounds[:3]
        ].reshape(
            (6, 1)
        )  # np.array([3.1, 3.1, 3.1, 3.1]).reshape((4,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotAt(At):

    # RobotAt Robot RobotPose

    def __init__(self, name, params, expected_param_types, env=None, sess=None):
        ## At Robot RobotPose
        self.r, self.rp = params
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("joint1", np.array([0], dtype=np.int_)),
                        ("joint2", np.array([0], dtype=np.int_)),
                        ("wrist", np.array([0], dtype=np.int_)),
                    ],
                ),
                (
                    self.rp,
                    [
                        ("joint1", np.array([0], dtype=np.int_)),
                        ("joint2", np.array([0], dtype=np.int_)),
                        ("wrist", np.array([0], dtype=np.int_)),
                    ],
                ),
            ]
        )

        A = np.c_[np.eye(3), -np.eye(3)]
        b = np.zeros((3, 1))
        val = np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class Near(At):
    def __init__(self, name, params, expected_param_types, env=None, sess=None):
        self.r, self.c = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int_))]),
                (self.c, [("value", np.array([0, 1], dtype=np.int_))]),
            ]
        )

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)]]
        b = np.zeros((4, 1))
        val = NEAR_TOL * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class GripperClosed(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None):
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


class Collides(CollisionPredicate):

    # Collides Can Obstacle (wall)

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self._env = env
        self.c, self.w = params
        attr_inds = OrderedDict(
            [
                (self.c, [("pose", np.array([0, 1], dtype=np.int_))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int_))]),
            ]
        )
        self._param_to_body = {
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        def f(x):
            return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4)

        def grad(x):
            return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4, grad=True)

        def f_neg(x):
            return -f(x)

        def grad_neg(x):
            return -grad(x)

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        # f_neg = lambda x: self.distance_from_obj(x)[0]
        # def grad_neg(x):
        #     # print self.distance_from_obj(x)
        #     return -self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((COL_TS * N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
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
                (self.c, [(k, np.array([0, 1], dtype=np.int_))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int_))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int_))]),
            ]
        )
        self._param_to_body = {
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        dist = RETREAT_DIST

        def f(x):
            disp = x[:2] + dist * x[4:6]
            new_x = np.concatenate([disp, x[2:4]])
            return -self.distance_from_obj(new_x)[0]

        def grad(x):
            disp = x[:2] + dist * x[4:6]
            new_x = np.concatenate([disp, x[2:4]])
            jac = self.distance_from_obj(new_x)[1]
            return np.c_[np.zeros((N_COLS, 2)), jac]

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
                (self.c, [(k, np.array([0, 1], dtype=np.int_))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int_))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int_))]),
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
                (self.c, [("value", np.array([0, 1], dtype=np.int_))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int_))]),
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
                        ("pose", np.array([0, 1], dtype=np.int_)),
                        ("gripper", np.array([0], dtype=np.int_)),
                        ("theta", np.array([0], dtype=np.int_)),
                    ],
                ),
                (self.w, [("pose", np.array([0, 1], dtype=np.int_))]),
            ]
        )
        self._param_to_body = {
            self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        neg_coeff = 1e3
        neg_grad_coeff = 1e-1

        def f(x):
            return -twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6)

        def grad(x):
            return -twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6, grad=True)

        def f_neg(x):
            return -neg_coeff * f(x)

        def grad_neg(x):
            return -neg_grad_coeff * grad(x)

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((COL_TS * N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(RCollides, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1
        )
        self.n_cols = N_COLS
        self.hl_ignore = True

        # self.priority = 1

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
                        ("joint1", np.array([0], dtype=np.int_)),
                        ("joint2", np.array([0], dtype=np.int_)),
                        ("wrist", np.array([0], dtype=np.int_)),
                        ("gripper", np.array([0], dtype=np.int_)),
                    ],
                ),
                (self.c, [("pose", np.array([0, 1], dtype=np.int_))]),
            ]
        )
        self._param_to_body = {
            self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
        }

        self.rs_scale = RS_SCALE

        neg_coeff = 1e0  # 1e3
        neg_grad_coeff = 1e-2  # 1e-3

        def f(x):
            val = -twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6)
            return val

        def grad(x):
            grad = -twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6, grad=True)
            return grad

        def f_neg(x):
            return -neg_coeff * f(x)

        def grad_neg(x):
            return -neg_grad_coeff * grad(x)

        col_expr = Expr(f, grad)
        val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Obstructs, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=3
        )
        # self.priority=1

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

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
        val = np.ones((1,))
        i = 0
        while i < 20 and np.any(val > 0):
            jnt1 = np.random.uniform(-np.pi, np.pi)
            jnt2 = np.random.uniform(-3, 3)
            wrist = np.random.uniform(-3, 3)
            newx = np.array([jnt1, jnt2, wrist] + x1[3:6].flatten().tolist())
            val, _ = self.distance_from_obj(newx)
            i += 1
        if np.any(val > 0):
            return None, None
        st = max(max(time - 3, 0), act.active_timesteps[0])
        ref_st = max(max(time - 3, 1), act.active_timesteps[0] + 1)
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        ref_et = min(min(time + 3, plan.horizon - 2), act.active_timesteps[1] - 1)
        st, et = act.active_timesteps
        nt = et - st + 1
        nlow = time - st
        nhigh = et - time
        for i in range(st + 1, et):
            dist = float(np.abs(i - time))
            if i <= time:
                inter_jnt1 = (dist / nlow) * self.r.joint1[:, st] + (
                    (nlow - dist) / nlow
                ) * jnt1
                inter_jnt2 = (dist / nlow) * self.r.joint2[:, st] + (
                    (nlow - dist) / nlow
                ) * jnt2
                inter_wrist = (dist / nlow) * self.r.wrist[:, st] + (
                    (nlow - dist) / nlow
                ) * wrist
            else:
                inter_jnt1 = (dist / nhigh) * self.r.joint1[:, et] + (
                    (nhigh - dist) / nhigh
                ) * jnt1
                inter_jnt2 = (dist / nhigh) * self.r.joint2[:, et] + (
                    (nhigh - dist) / nhigh
                ) * jnt2
                inter_wrist = (dist / nhigh) * self.r.wrist[:, et] + (
                    (nhigh - dist) / nhigh
                ) * wrist

            add_to_attr_inds_and_res(
                i, attr_inds, res, self.r, [("joint1", inter_jnt1)]
            )
            add_to_attr_inds_and_res(
                i, attr_inds, res, self.r, [("joint2", inter_jnt2)]
            )
            add_to_attr_inds_and_res(
                i, attr_inds, res, self.r, [("wrist", inter_wrist)]
            )
        return res, attr_inds


class WideObstructs(Obstructs):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        super(WideObstructs, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = 0.3
        self.check_aabb = False  # True


class ObstructsNoSym(Obstructs):
    pass


class WideObstructsNoSym(WideObstructs):
    pass


class ObstructsNoSym(Obstructs):
    pass


class WideObstructsNoSym(WideObstructs):
    pass


def sample_pose(plan, pose, robot, rs_scale):
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
        p.priority = -1
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
        p.priority = old_priority[i]

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
                        ("joint1", np.array([0], dtype=np.int_)),
                        ("joint2", np.array([0], dtype=np.int_)),
                        ("wrist", np.array([0], dtype=np.int_)),
                        ("gripper", np.array([0], dtype=np.int_)),
                    ],
                ),
                (obstr, [("pose", np.array([0, 1], dtype=np.int_))]),
                (held, [("pose", np.array([0, 1], dtype=np.int_))]),
            ]
        )

        self._param_to_body = {
            r: self.lazy_spawn_or_body(r, r.name, r.geom),
            obstr: self.lazy_spawn_or_body(obstr, obstr.name, obstr.geom),
            held: self.lazy_spawn_or_body(held, held.name, held.geom),
        }

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        neg_coeff = 1e0  # 1e3
        neg_grad_coeff = 1e-2  # 1e-3
        ## so we have an expr for the negated predicate
        # f_neg = lambda x: neg_coeff*self.distance_from_obj(x)[0]
        # grad_neg = lambda x: neg_grad_coeff*self.distance_from_obj(x)[1]

        def f(x):
            if self.obstr.name == self.held.name:
                return -twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6)
            else:
                return -twostep_f([x[:8], x[8:16]], self.distance_from_obj, 8)

        def grad(x):
            if self.obstr.name == self.held.name:
                grad = -twostep_f(
                    [x[:6], x[6:12]], self.distance_from_obj, 6, grad=True
                )
                return grad
            else:
                grad = -twostep_f(
                    [x[:8], x[8:16]], self.distance_from_obj, 8, grad=True
                )
                return grad

        def f_neg(x):
            return -neg_coeff * f(x)

        def grad_neg(x):
            gradneg = -neg_grad_coeff * grad(x)
            return gradneg

        col_expr = Expr(f, grad)
        val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)

        if self.held != self.obstr:
            col_expr_neg = Expr(f_neg, grad_neg)
        else:
            new_f_neg = lambda x: 0.0 * f(x)  # self.distance_from_obj(x)[0]
            new_grad_neg = lambda x: -grad(x)  # self.distance_from_obj(x)[1]
            col_expr_neg = Expr(new_f_neg, new_grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(ObstructsHolding, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=3
        )
        # self.priority=1

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

    def distance_from_obj(self, x, n_steps=0):
        # x = [rpx, rpy, obstrx, obstry, heldx, heldy]
        pose_r = x[:4]
        pose_obstr = x[4:6]
        b0 = self._param_to_body[self.r]
        b1 = self._param_to_body[self.obstr]
        pose0, pose1 = self.set_pos(x[:6])
        collisions1 = p.getClosestPoints(b0.body_id, b1.body_id, contact_dist)
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
            # b2 = self._param_to_body[self.held]
            # pose_held = x[6:8]
            # b2.set_pose(pose_held)
            # collisions2 = p.getClosestPoints(b2.body_id, b1.body_id, contact_dist)
            # col_val2, jac21 = self._calc_obj_grad_and_val(self.held.name, self.obstr.name, pose_held, pose_obstr, collisions2)

            val = np.array(col_val1)
            jac = np.c_[jac01, np.zeros((N_COLS, 2))].reshape((N_COLS, 8))
            collisions = collisions1
            # if np.max(col_val1) > np.max(col_val2):
            #    val = np.array(col_val1)
            #    jac = np.c_[jac01, np.zeros((N_COLS, 2))].reshape((N_COLS, 6))
            #    collisions = collisions1
            # else:
            #    val = np.array(col_val2)
            #    jac = np.c_[np.zeros((N_COLS, 2)), jac21[:, 2:], jac21[:, :2]].reshape((N_COLS, 6))
            #    collisions = collisions2

        return val, jac

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
        val = np.ones((1,))
        i = 0
        while i < 20 and np.any(val > 0):
            jnt1 = np.random.uniform(-np.pi, np.pi)
            jnt2 = np.random.uniform(-3, 3)
            wrist = np.random.uniform(-3, 3)
            newx = np.array([jnt1, jnt2, wrist] + x1[3:8].flatten().tolist())
            val, _ = self.distance_from_obj(newx)
            i += 1
        if np.any(val > 0):
            return None, None

        st, et = act.active_timesteps
        nt = et - st + 1
        nlow = time - st
        nhigh = et - time
        for i in range(st + 1, et):
            dist = float(np.abs(i - time))
            if i <= time:
                inter_jnt1 = (dist / nlow) * self.r.joint1[:, st] + (
                    (nlow - dist) / nlow
                ) * jnt1
                inter_jnt2 = (dist / nlow) * self.r.joint2[:, st] + (
                    (nlow - dist) / nlow
                ) * jnt2
                inter_wrist = (dist / nlow) * self.r.wrist[:, st] + (
                    (nlow - dist) / nlow
                ) * wrist
            else:
                inter_jnt1 = (dist / nhigh) * self.r.joint1[:, et] + (
                    (nhigh - dist) / nhigh
                ) * jnt1
                inter_jnt2 = (dist / nhigh) * self.r.joint2[:, et] + (
                    (nhigh - dist) / nhigh
                ) * jnt2
                inter_wrist = (dist / nhigh) * self.r.wrist[:, et] + (
                    (nhigh - dist) / nhigh
                ) * wrist

            add_to_attr_inds_and_res(
                i, attr_inds, res, self.r, [("joint1", inter_jnt1)]
            )
            add_to_attr_inds_and_res(
                i, attr_inds, res, self.r, [("joint2", inter_jnt2)]
            )
            add_to_attr_inds_and_res(
                i, attr_inds, res, self.r, [("wrist", inter_wrist)]
            )
        return res, attr_inds


class WideObstructsHolding(ObstructsHolding):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        super(WideObstructsHolding, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = 0.3
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
                (self.r, [("pose", np.array([0, 1], dtype=np.int_))]),
                (self.can, [("pose", np.array([0, 1], dtype=np.int_))]),
                (self.grasp, [("value", np.array([0, 1], dtype=np.int_))]),
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
        self.dist = gripdist
        self.ee_link = self.r.geom.ee_link
        self.coeff = 1e-1

        if self.r.is_symbol():
            k = "value"
        else:
            k = "pose"

        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("joint1", np.array([0], dtype=np.int_)),
                        ("joint2", np.array([0], dtype=np.int_)),
                        ("wrist", np.array([0], dtype=np.int_)),
                        ("gripper", np.array([0], dtype=np.int_)),
                    ],
                ),
                (self.can, [("pose", np.array([0, 1], dtype=np.int_))]),
            ]
        )

        def f(x):
            x = x.flatten()
            dist = self.dist
            jntstf = get_tf_graph("jnts")
            objtf = get_tf_graph("obj_pose")
            disttf = get_tf_graph("dist")
            valtf = get_tf_graph("ingrasp")
            val = self.sess.run(
                valtf, feed_dict={jntstf: x[:4], objtf: x[4:6], disttf: dist}
            )
            return self.coeff * np.array([[val]])

        def grad(x):
            x = x.flatten()
            dist = self.dist
            jntstf = get_tf_graph("jnts")
            objtf = get_tf_graph("obj_pose")
            disttf = get_tf_graph("dist")
            grads = get_tf_graph("ingrasp_gradients")
            robot_jacs, obj_jacs = np.array(
                self.sess.run(
                    grads, feed_dict={jntstf: x[:4], objtf: x[4:6], disttf: dist}
                )
            ).reshape((-1, 1))
            jac = np.r_[robot_jacs[0], obj_jacs[0]].reshape((1, 6))
            return self.coeff * jac

        self.f = f
        self.grad = grad
        angle_expr = Expr(f, grad)
        e = EqExpr(angle_expr, np.zeros((1, 1)))

        super(InGraspAngle, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=2
        )

    def resample(self, negated, time, plan):
        res = OrderedDict()
        attr_inds = OrderedDict()
        a = 0
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None

        act = plan.actions[a]
        x1 = self.get_param_vector(time).flatten()
        robot_body = self.r.openrave_body
        robot_body.set_dof({"joint1": x1[0], "joint2": x1[1], "wrist": x1[2]})
        ul = robot_body._geom.upper_bounds
        ll = robot_body._geom.lower_bounds
        jnt_rng = ul - ll
        target_pos = self.can.pose[:, time]
        jnts = p.calculateInverseKinematics(
            robot_body.body_id,
            self.ee_link,
            target_pos.tolist() + [0.5],
            lowerLimits=ll.tolist(),
            upperLimits=ul.tolist(),
            jointRanges=jnt_rng.tolist(),
            restPoses=x1[:4].tolist() + [x1[3]],
            maxNumIterations=500,
        )
        add_to_attr_inds_and_res(
            time, attr_inds, res, self.r, [("joint1", np.array([jnts[0]]))]
        )
        add_to_attr_inds_and_res(
            time, attr_inds, res, self.r, [("joint2", np.array([jnts[1]]))]
        )
        add_to_attr_inds_and_res(
            time, attr_inds, res, self.r, [("wrist", np.array([jnts[2]]))]
        )
        robot_body.set_dof({"joint1": jnts[0], "joint2": jnts[1], "wrist": jnts[2]})
        return res, attr_inds


class TargetInGraspAngle(InGraspAngle):
    pass


class NearGraspAngle(InGraspAngle):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.r, self.can = params
        self.tol = 1e-1
        self.dist = gripdist
        if self.r.is_symbol():
            k = "value"
        else:
            k = "pose"

        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("joint1", np.array([0], dtype=np.int_)),
                        ("joint2", np.array([0], dtype=np.int_)),
                        ("wrist", np.array([0], dtype=np.int_)),
                        ("gripper", np.array([0], dtype=np.int_)),
                    ],
                ),
                (self.can, [("pose", np.array([0, 1], dtype=np.int_))]),
            ]
        )

        def f(x):
            x = x.flatten()
            dist = self.dist
            jntstf = get_tf_graph("jnts")
            objtf = get_tf_graph("obj_pose")
            disttf = get_tf_graph("dist")
            valtf = get_tf_graph("ingrasp")
            val = self.sess.run(
                valtf, feed_dict={jntstf: x[:4], objtf: x[4:6], disttf: dist}
            )
            return np.array([[val]])

        def grad(x):
            x = x.flatten()
            dist = self.dist
            jntstf = get_tf_graph("jnts")
            objtf = get_tf_graph("obj_pose")
            disttf = get_tf_graph("dist")
            grads = get_tf_graph("ingrasp_gradients")
            robot_jacs, obj_jacs = np.array(
                self.sess.run(
                    grads, feed_dict={jntstf: x[:4], objtf: x[4:6], disttf: dist}
                )
            ).reshape((-1, 1))
            jac = np.r_[robot_jacs[0], obj_jacs[0]].reshape((1, 6))
            return jac

        angle_expr = Expr(f, grad)
        e = LEqExpr(angle_expr, self.tol * np.ones((1, 1)))

        super(InGraspAngle, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=2
        )


class ApproachGraspAngle(InGraspAngle):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        super(ApproachGraspAngle, self).__init__(
            name, params, expected_param_types, env, sess, debug
        )
        self.dist = RETREAT_DIST
        self.ee_link = self.r.geom.far_ee_link
        self.coeff = 5e-3


class TargetApproachGraspAngle(ApproachGraspAngle):
    pass


class RobotStationary(ExprPredicate):

    # Stationary, Can

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.c,) = params
        attr_inds = OrderedDict(
            [
                (
                    self.c,
                    [
                        ("joint1", np.array([0], dtype=np.int_)),
                        ("joint2", np.array([0], dtype=np.int_)),
                        ("wrist", np.array([0], dtype=np.int_)),
                    ],
                )
            ]
        )
        A = np.c_[np.eye(3), -np.eye(3)]
        b = np.zeros((3, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((3, 1)))
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


class StationaryRot(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.c,) = params
        attr_inds = OrderedDict([(self.c, [("wrist", np.array([0], dtype=np.int_))])])
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


class StationaryWrist(StationaryRot):
    pass


class Stationary(ExprPredicate):

    # Stationary, Can

    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.c,) = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int_))])])
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


class AtRot(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        self.r, self.rot = params
        attr_inds = OrderedDict(
            [
                (self.r, [("theta", np.array([0], dtype=np.int_))]),
                (self.rot, [("value", np.array([0], dtype=np.int_))]),
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
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int_))])])
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
        attr_inds = OrderedDict([(self.w, [("pose", np.array([0, 1], dtype=np.int_))])])
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
        (self.c,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (
                    self.c,
                    [
                        ("joint1", np.array([0], dtype=np.int_)),
                        ("joint2", np.array([0], dtype=np.int_)),
                        ("wrist", np.array([0], dtype=np.int_)),
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
        dmove = np.pi / 8.0
        drot = np.pi / 8.0
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


class VelWithinBounds(At):

    # RobotAt Robot Can

    def __init__(self, name, params, expected_param_types, env=None, sess=None):
        ## At Robot RobotPose
        self.r, self.c = params
        attr_inds = OrderedDict([(self.r, [("vel", np.array([0, 1], dtype=np.int_))])])

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
        attr_inds = OrderedDict([(self.r, [("acc", np.array([0, 1], dtype=np.int_))])])

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
                        ("pose", np.array([0, 1], dtype=np.int_)),
                        ("vel", np.array([0, 1], dtype=np.int_)),
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
                (self.r, [("vel", np.array([0, 1], dtype=np.int_))]),
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
                (self.r, [("vel", np.array([0, 1], dtype=np.int_))]),
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
                        ("pose", np.array([0, 1], dtype=np.int_)),
                        ("vel", np.array([0, 1], dtype=np.int_)),
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
                        ("vel", np.array([0, 1], dtype=np.int_)),
                        ("acc", np.array([0, 1], dtype=np.int_)),
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
                        ("pose", np.array([0, 1], dtype=np.int_)),
                        ("theta", np.array([0], dtype=np.int_)),
                        ("vel", np.array([0], dtype=np.int_)),
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


class ThetaDirValid(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        (self.r,) = params
        self.forward, self.reverse = False, False
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int_)),
                        ("theta", np.array([0], dtype=np.int_)),
                        ("vel", np.array([0], dtype=np.int_)),
                    ],
                ),
            ]
        )

        self.sess = sess
        if USE_TF:

            def f(x):
                cur_tensor = get_tf_graph("thetadir_tf_both")
                if self.forward:
                    cur_tensor = get_tf_graph("thetadir_tf_off")
                if self.reverse:
                    cur_tensor = get_tf_graph("thetadir_tf_opp")
                return np.array(
                    [
                        self.sess.run(
                            cur_tensor, feed_dict={get_tf_graph("thetadir_tf_in"): x}
                        )
                    ]
                )

            def grad(x):
                cur_grads = get_tf_graph("thetadir_tf_grads")
                if self.forward:
                    cur_grads = get_tf_graph("thetadir_tf_forgrads")
                if self.reverse:
                    cur_grads = get_tf_graph("thetadir_tf_revgrads")
                v = self.sess.run(
                    cur_grads, feed_dict={get_tf_graph("thetadir_tf_in"): x}
                ).T
                v[np.isnan(v)] = 0.0
                v[np.isinf(v)] = 0.0
                return v

        else:

            def f(x):
                x = x.flatten()
                curdisp = x[4:6] - x[:2]
                dist = np.linalg.norm(curdisp)
                theta = x[2]
                targ_disp = [-dist * np.sin(theta), dist * np.cos(theta)]
                off = (curdisp[0] - targ_disp[0]) ** 2 + (
                    curdisp[1] - targ_disp[1]
                ) ** 2
                opp_off = (curdisp[0] + targ_disp[0]) ** 2 + (
                    curdisp[1] + targ_disp[1]
                ) ** 2
                if self.forward:
                    return np.array([[off]])
                if self.reverse:
                    return np.array([[opp_off]])
                return np.array([[min(off, opp_off)]])

                if np.linalg.norm(curdisp) < 1e-3:
                    return np.zeros((1, 1))
                    return np.zeros((3, 1))
                curtheta = np.arctan2(curdisp[0], curdisp[1])
                theta = x[2]
                opp_theta = opposite_angle(curtheta)
                if not self.forward and np.abs(angle_diff(curtheta, theta)) > np.abs(
                    angle_diff(opp_theta, theta)
                ):
                    curtheta = opp_theta
                elif self.reverse:
                    curtheta = opp_theta

                theta_off = angle_diff(theta, curtheta)
                rot = np.array(
                    [
                        [np.cos(theta_off), -np.sin(theta_off)],
                        [np.sin(theta_off), np.cos(theta_off)],
                    ]
                )
                newdisp = rot.dot(curdisp)
                pos_off = newdisp - curdisp
                return np.array([[pos_off[0] ** 2 + pos_off[1] ** 2]])
                return np.r_[pos_off, [theta_off]].reshape((3, 1))

            def grad(x):
                x = x.flatten()
                curdisp = x[4:6] - x[:2]
                dist = np.linalg.norm(curdisp)
                theta = x[2]
                targ_disp = [-dist * np.sin(theta), dist * np.cos(theta)]
                off = (curdisp[0] - targ_disp[0]) ** 2 + (
                    curdisp[1] - targ_disp[1]
                ) ** 2
                opp_off = (curdisp[0] + targ_disp[0]) ** 2 + (
                    curdisp[1] + targ_disp[1]
                ) ** 2
                if not self.forward and (self.reverse or opp_off < off):
                    theta += np.pi
                (x1, y1), (x2, y2) = x[:2], x[4:6]
                xdiff = x2 - x1
                ydiff = y2 - y1
                x1_grad, x2_grad, y1_grad, y2_grad, theta_grad = 0, 0, 0, 0, 0
                if False and dist > 1e-4:
                    x2_grad = (
                        2
                        * (xdiff * np.sin(theta) / dist + 2 * xdiff)
                        * (np.sin(theta) * dist + xdiff ** 2)
                        - 2
                        * xdiff
                        * np.cos(theta)
                        * (ydiff ** 2 - np.cos(theta) * dist)
                        / dist
                    )
                    x1_grad = -x2_grad
                    y2_grad = 2 * ydiff * np.sin(theta) * (
                        np.sin(theta) * dist + xdiff ** 2
                    ) / dist + 2 * (ydiff ** 2 - np.cos(theta) * dist) * (
                        2 * ydiff - ydiff * np.cos(theta) / dist
                    )
                    y1_grad = -y2_grad
                    theta_grad = (
                        2
                        * dist
                        * (xdiff ** 2 * np.cos(theta) + ydiff ** 2 * np.sin(theta))
                    )
                    return np.array(
                        [x1_grad, y1_grad, theta_grad, 0, x2_grad, y2_grad, 0, 0]
                    ).reshape((1, 8))

                x1_grad = -2 * ((x2 - x1) + dist * np.sin(theta))
                y1_grad = -2 * ((y2 - y1) - dist * np.cos(theta))
                theta_grad = (
                    2 * dist * ((x2 - x1) * np.cos(theta) + (y2 - y1) * np.sin(theta))
                )
                x2_grad = 2 * ((x2 - x1) + dist * np.sin(theta))
                y2_grad = 2 * ((y2 - y1) - dist * np.cos(theta))
                return np.array(
                    [x1_grad, y1_grad, theta_grad, 0, x2_grad, y2_grad, 0, 0]
                ).reshape((1, 8))

                curtheta = np.arctan2(curdisp[0], curdisp[1])
                theta = x[2]
                opp_theta = opposite_angle(curtheta)
                if not self.forward and np.abs(angle_diff(curtheta, theta)) > np.abs(
                    angle_diff(opp_theta, theta)
                ):
                    curtheta = opp_theta
                elif self.reverse:
                    curtheta = opp_theta

                theta_off = angle_diff(theta, curtheta)
                rot = np.array(
                    [
                        [np.cos(theta_off), -np.sin(theta_off)],
                        [np.sin(theta_off), np.cos(theta_off)],
                    ]
                )
                newdisp = rot.dot(curdisp)
                pos_off = newdisp - curdisp

                x1_grad = -2 * ((x2 - x1) - xt)
                y1_grad = -2 * ((y2 - y1) - yt)
                theta_grad = 2 * dist * np.cos(theta) * (
                    (x2 - x1) + dist * np.sin(theta)
                ) + -2 * dist * np.sin(theta) * ((x2 - x1) - dist * np.cos(theta))
                x2_grad = 2 * ((x2 - x1) - xt)
                y2_grad = 2 * ((y2 - y1) - yt)
                posgrad = np.c_[
                    -np.eye(2), np.zeros((2, 2)), np.eye(2), np.zeros((2, 2))
                ]
                theta_grad = np.array([0, 0, 1, 0, 0, 0, 0, 0]).reshape((1, 8))
                return np.r_[posgrad, theta_grad].reshape((3, 8))

        angle_expr = Expr(f, grad)
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
        super(ForThetaDirValid, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.forward = True


class RevThetaDirValid(ThetaDirValid):
    def __init__(
        self, name, params, expected_param_types, env=None, sess=None, debug=False
    ):
        super(RevThetaDirValid, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.reverse = True


class ColObjPred(CollisionPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, coeff=1e3, debug=False
    ):
        self._env = env
        self.hl_ignore = True
        self.r, self.c = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int_))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int_))]),
            ]
        )
        self._param_to_body = {
            self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
        }

        self.rs_scale = RS_SCALE
        self.radius = self.c.geom.radius + 2.0
        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        neg_coeff = coeff
        neg_grad_coeff = 1e-1  # 1e-3

        def f(x):
            xs = [
                float(COL_TS - t) / COL_TS * x[:4] + float(t) / COL_TS * x[4:]
                for t in range(COL_TS + 1)
            ]
            if hasattr(self, "sess") and USE_TF:
                cur_tensor = get_tf_graph("bump_out")
                in_tensor = get_tf_graph("bump_in")
                radius_tensor = get_tf_graph("bump_radius")
                vals = []
                for i in range(COL_TS + 1):
                    pt = xs[i]
                    if np.sum((pt[:2] - pt[2:]) ** 2) > (self.radius - 1e-3) ** 2:
                        vals.append(0)
                    else:
                        val = np.array(
                            [
                                self.sess.run(
                                    cur_tensor,
                                    feed_dict={
                                        in_tensor: pt,
                                        radius_tensor: self.radius ** 2,
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

        def grad(x):
            xs = [
                float(COL_TS - t) / COL_TS * x[:4] + float(t) / COL_TS * x[4:]
                for t in range(COL_TS + 1)
            ]
            if hasattr(self, "sess") and USE_TF:
                cur_grads = get_tf_graph("bump_grads")
                in_tensor = get_tf_graph("bump_in")
                radius_tensor = get_tf_graph("bump_radius")
                vals = []
                for i in range(COL_TS + 1):
                    pt = xs[i]
                    if np.sum((pt[:2] - pt[2:]) ** 2) > (self.radius - 1e-3) ** 2:
                        vals.append(np.zeros((1, 8)))
                    else:
                        v = self.sess.run(
                            cur_grads,
                            feed_dict={in_tensor: pt, radius_tensor: self.radius ** 2},
                        ).T
                        v[np.isnan(v)] = 0.0
                        v[np.isinf(v)] = 0.0
                        curcoeff = float(COL_TS - i) / COL_TS
                        vals.append(np.c_[curcoeff * v, (1 - curcoeff) * v])
                return np.sum(vals, axis=0)
            return (
                -coeff * self.distance_from_obj(x)[1]
            )  # twostep_f([x[:4]], self.distance_from_obj, 2, pts=1, grad=True)

        def f_neg(x):
            return -neg_coeff * f(x)

        def grad_neg(x):
            return -neg_grad_coeff * grad(x)

        def hess_neg(x):
            xs = [
                float(COL_TS - t) / COL_TS * x[:4] + float(t) / COL_TS * x[4:]
                for t in range(COL_TS + 1)
            ]
            if hasattr(self, "sess") and USE_TF:
                cur_hess = get_tf_graph("bump_hess")
                in_tensor = get_tf_graph("bump_in")
                radius_tensor = get_tf_graph("bump_radius")
                vals = []
                for i in range(COL_TS + 1):
                    pt = xs[i]
                    if np.sum((pt[:2] - pt[2:]) ** 2) > (self.radius - 1e-3) ** 2:
                        vals.append(np.zeros((8, 8)))
                    else:
                        v = self.sess.run(
                            cur_hess,
                            feed_dict={in_tensor: pt, radius_tensor: self.radius ** 2},
                        )
                        v[np.isnan(v)] = 0.0
                        v[np.isinf(v)] = 0.0
                        v = v.reshape((4, 4))
                        curcoeff = float(COL_TS - i) / COL_TS
                        new_v = np.r_[
                            np.c_[curcoeff * v, np.zeros((4, 4))],
                            np.c_[np.zeros((4, 4)), (1 - curcoeff) * v],
                        ]
                        vals.append(new_v.reshape((8, 8)))
                return np.sum(vals, axis=0).reshape((8, 8))
            j = grad(x)
            return j.T.dot(j)

        col_expr = Expr(f, grad)
        val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(
            lambda x: coeff * f(x),
            lambda x: coeff * grad(x),
            lambda x: coeff * hess_neg(x),
        )
        self.neg_expr = LEqExpr(col_expr_neg, -val)

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
