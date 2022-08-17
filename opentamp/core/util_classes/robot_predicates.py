from opentamp.core.util_classes.common_predicates import ExprPredicate
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
import opentamp.core.util_classes.transform_utils as T
import opentamp.core.util_classes.common_constants as const
from opentamp.errors_exceptions import PredicateException
from opentamp.core.util_classes import robot_sampling

from sco_py.expr import Expr, AffExpr, EqExpr, LEqExpr

from collections import OrderedDict
import itertools
import numpy as np
import pybullet as p
import sys
import time
import traceback


DEFAULT_TOL = 1e-3
NEAR_TOL = 1e-1 # 2e-2
NEAR_ROT_TOL = 0.2


### HELPER FUNCTIONS

def init_robot_pred(pred, robot, params=[], robot_poses=[], attrs={}):
    """
    Initializes attr_inds and attr_dim from the robot's geometry
    """
    r_geom = robot.geom
    if robot not in attrs:
        attrs[robot] = ['pose', 'rotation'] + r_geom.arms + [r_geom.ee_link_names[arm] for arm in r_geom.arms] + r_geom.ee_attrs

    base_dim = 3 if not len(attrs[robot]) or 'pose' in attrs[robot] else 0
    base_dim += 3 if not len(attrs[robot]) or 'rotation' in attrs[robot] else 0
    arm_dims = sum([len(r_geom.jnt_names[arm]) for arm in r_geom.arms if not len(attrs[robot]) or arm in attrs[robot]])
    gripper_dims = sum([r_geom.gripper_dim(arm) for arm in r_geom.arms if not len(attrs[robot]) or r_geom.ee_link_names[arm] in attrs[robot]])
    ee_dims = sum([3 for attr in r_geom.ee_attrs if not len(attrs[robot]) or attr in attrs[robot]])
    cur_attr_dim = base_dim + arm_dims + gripper_dims + ee_dims
    cur_attr_dim *= 1 + len(robot_poses)
    robot_inds = []

    attr_inds = OrderedDict()
    robot_inds = []
    pose_inds = []
    attr_inds[robot] = robot_inds
    for attr in attrs[robot]:
        if attr in r_geom.jnt_names:
            njnts = len(r_geom.jnt_names[attr])
            robot_inds.append((attr, np.array(range(njnts), dtype='int32')))
            if len(robot_poses): pose_inds.append((attr, np.array(range(njnts), dtype='int32')))
        elif attr.find('ee_pos') >= 0:
            robot_inds.append((attr, np.array(range(3), dtype='int32')))
            if len(robot_poses): pose_inds.append((attr, np.array(range(3), dtype='int32')))
        elif attr.find('ee_rot') >= 0:
            robot_inds.append((attr, np.array(range(3), dtype='int32')))
            if len(robot_poses): pose_inds.append((attr, np.array(range(3), dtype='int32')))
        elif attr == 'pose':
            robot_inds.append((attr, np.array(range(3), dtype='int32')))
            if len(robot_poses): pose_inds.append(('value', np.array(range(3), dtype='int32')))
        elif attr == 'rotation':
            robot_inds.append((attr, np.array(range(3), dtype='int32')))
            if len(robot_poses): pose_inds.append(('rotation', np.array(range(3), dtype='int32')))
    #robot_inds = list(filter(lambda inds: inds[0] in attrs, r_geom.attr_map['robot']))
    if len(robot_poses):
        for pose in robot_poses:
            attr_inds[pose] = pose_inds

    for p in params:
        attr_inds[p] = [(attr, inds) for (attr, inds) in const.ATTRMAP[p._type] if p not in attrs or attr in attrs[p]]
        for (attr, inds) in attr_inds[p]:
            cur_attr_dim += len(inds)
    pred.attr_inds = attr_inds
    pred.attr_dim = cur_attr_dim
    return pred.attr_inds, pred.attr_dim

def parse_collision(c, obj_body, obstr_body, held_links=[], obs_links=[]):
    linkA, linkB = c[3], c[4]
    linkAParent, linkBParent = c[1], c[2]
    sign = 0
    if linkAParent == obj_body.body_id and linkBParent == obstr_body.body_id:
        ptObj, ptObstr = c[5], c[6]
        linkObj, linkObstr = linkA, linkB
        sign = -1
    elif linkBParent == obj_body.body_id and linkAParent == obstr_body.body_id:
        ptObj, ptObstr = c[6], c[5]
        linkObj, linkObstr = linkB, linkA
        sign = 1
    else:
        return None

    if (len(held_links) and linkObj not in held_links) or (len(obs_links) and linkObstr not in obs_links):
        return None

    # Obtain distance between two collision points, and their normal collision vector
    distance = np.array(c[8]) # c.contactDistance
    normal = np.array(c[7]) # c.contactNormalOnB # Pointing towards A
    ptObj = np.array(ptObj)
    ptObstr = np.array(ptObstr)
    return distance, normal, linkObj, linkObstr, ptObj, ptObstr

def parse_robot_collision(c, robot, robot_body, obj_body, col_links=[], obj_links=[]):
    linkA, linkB = c[3], c[4] # c.linkIndexA, c.linkIndexB
    linkAParent, linkBParent = c[1], c[2] # c.bodyUniqueIdA, c.bodyUniqueIdB
    sign = 0
    if linkAParent == robot_body.body_id and linkBParent == obj_body.body_id:
        ptRobot, ptObj = c[5], c[6] # c.positionOnA, c.positionOnB
        linkRobot, linkObj = linkA, linkB
        sign = -1
    elif linkBParent == robot_body.body_id and linkAParent == obj_body.body_id:
        ptRobot, ptObj = c[6], c[5] # c.positionOnB, c.positionOnA
        linkRobot, linkObj = linkB, linkA
        sign = 1
    else:
        return None

    if (len(col_links) and linkRobot not in col_links) or (len(obj_links) and linkObj not in obj_links):
        return None

    distance = c[8] # c.contactDistance
    normal = c[7] # c.contactNormalOnB # Pointing towards A
    jnts = robot_body._geom.get_free_jnts()
    n_jnts = len(jnts)
    robot_jac, robot_ang_jac = p.calculateJacobian(robot_body.body_id,
                                                   linkRobot,
                                                   ptRobot,
                                                   objPositions=jnts,
                                                   objVelocities=np.zeros(n_jnts).tolist(),
                                                   objAccelerations=np.zeros(n_jnts).tolist())
    normal = np.array(normal)
    ptRobot = np.array(ptRobot)
    ptObj = np.array(ptObj)
    robot_jac = -np.array(robot_jac)
    robot_ang_jac = np.array(robot_ang_jac)

    # PyBullet adds the first 6 indices if the base is floating
    if robot_jac.shape[-1] != n_jnts:
        robot_jac = robot_jac[:,6:]
        robot_ang_jac = robot_ang_jac[6:]
    return distance, normal, linkRobot, linkObj, ptRobot, ptObj, robot_jac, robot_ang_jac

def compute_arm_pos_jac(arm_joints, robot_body, pos):
    arm_jac = []
    for jnt_id in arm_joints:
        info = p.getJointInfo(robot_body.body_id, jnt_id)
        axis = info[13]
        jnt_state = p.getLinkState(robot_body.body_id, jnt_id)
        jnt_pos = np.array(jnt_state[0])
        quat = jnt_state[1]
        mat = T.quat2mat(quat)
        axis = mat.dot(axis)
        #parent_id = info[-1]
        #parent_frame_pos = info[14]
        #parent_info = p.getLinkState(robot_body.body_id, parent_id)
        #parent_pos = np.array(parent_info[0])
        arm_jac.append(np.cross(axis, pos - jnt_pos))
    arm_jac = np.array(arm_jac).T
    return arm_jac

def compute_arm_rot_jac(arm_joints, robot_body, obj_dir, world_dir, sign=1.):
    arm_jac = []
    for jnt_id in arm_joints:
        info = p.getJointInfo(robot_body.body_id, jnt_id)
        axis = info[13]
        quat = p.getLinkState(robot_body.body_id, jnt_id)[1]
        mat = T.quat2mat(quat)
        axis = mat.dot(axis)
        arm_jac.append(np.dot(obj_dir, np.cross(axis, sign * world_dir)))
    arm_jac = np.array(arm_jac).reshape((-1, len(arm_joints)))
    return arm_jac


### BASE CLASSES

class RobotPredicate(ExprPredicate):
    """
    Super-class for all robot predicates, defines several required functions
    """
    def __init__(self, name, expr, attr_inds, params, expected_param_types, env=None, active_range=(0,0), tol=DEFAULT_TOL, priority=0, debug=False):
        if not hasattr(self, 'arm') and hasattr(params[0].geom, 'arm'): self.arm = params[0].geom.arms[0]
        super(RobotPredicate, self).__init__(name, expr, attr_inds, params, expected_param_types, tol=tol, priority = priority, active_range=active_range)
        self._init_include = False

    def get_robot_info(self, robot_body, arm):
        arm_inds = robot_body._geom.get_arm_inds(arm)
        ee_link = robot_body._geom.get_ee_link(arm)
        info = p.getLinkState(robot_body.body_id, ee_link)
        pos, rot = info[0], info[1]
        robot_trans = OpenRAVEBody.transform_from_obj_pose(pos, rot)
        return robot_trans, arm_inds

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        geom = robot_body._geom
        dof_value_map = {}
        if hasattr(self, 'attr_inds'):
            for attr, inds in self.attr_inds[self.robot]:
                if attr not in self.robot.geom.dof_map: continue
                dof_value_map[attr] = x[self.attr_map[self.robot, attr]]
        else:
            for dof, dof_ind in geom.get_dof_inds():
                dof_value_map[dof] = x[dof_ind].flatten()
        robot_body.set_dof(dof_value_map)
        robot_body.set_pose(x[self.attr_map[self.robot, 'pose']].flatten(),\
                            x[self.attr_map[self.robot, 'rotation']].flatten(),)
        if hasattr(self, 'obj'):
            pos = x[self.attr_map[self.obj, 'pose']]
            rot = x[self.attr_map[self.obj, 'rotation']]
            self.obj.openrave_body.set_pose(pos, rot)
        elif hasattr(self, 'targ') and hasattr(self.targ, 'openrave_body'):
            if self.targ.openrave_body is not None:
                pos = x[self.attr_map[self.targ, 'value']]
                rot = x[self.attr_map[self.targ, 'rotation']]
                self.targ.openrave_body.set_pose(pos, rot)

        if hasattr(self, 'obstacle'):
            pos = x[self.attr_map[self.obstacle, 'pose']]
            rot = x[self.attr_map[self.obstacle, 'rotation']]
            self.obstacle.openrave_body.set_pose(pos, rot)

    def robot_obj_kinematics(self, x):
        # Getting the variables
        robot_body = self.robot.openrave_body

        # Setting the poses for forward kinematics to work
        self.set_robot_poses(x, robot_body)
        robot_trans, arm_inds = self.get_robot_info(robot_body, self.arm)

        # Assume target information is in the tail
        if hasattr(self, 'obj'):
            pos_inds, rot_inds = self.attr_map[self.obj, 'pose'], self.attr_map[self.obj, 'rotation']
            ee_pos, ee_rot = x[pos_inds], x[rot_inds]
        elif hasattr(self, 'targ'):
            pos_inds, rot_inds = self.attr_map[self.targ, 'value'], self.attr_map[self.targ, 'rotation']
            ee_pos, ee_rot = x[pos_inds], x[rot_inds]
        elif hasattr(self, 'ee_ref') and self.ee_ref:
            pos_inds, rot_inds = self.attr_map[self.robot, '{}_ee_pos'.format(self.arm)], self.attr_map[self.robot, '{}_ee_rot'.format(self.arm)]
            ee_pos, ee_rot = x[pos_inds], x[rot_inds]
        else:
            ee_pos, ee_rot = np.zeros((3,1)), np.zeros((3,1))

        ee_rot = ee_rot.flatten()
        obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos.flatten(), [ee_rot[2], ee_rot[1], ee_rot[0]])
        #obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos, [ee_rot[0], ee_rot[1], ee_rot[2]])
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, [ee_rot[2], ee_rot[1], ee_rot[0]])
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))] # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        return obj_trans, robot_trans, axises, arm_inds

 
    def setup_mov_limit_check(self, delta=False, ee_only=False):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        geom = robot_body._geom
        dof_map = geom.dof_map
        if ee_only:
            dof_inds = np.concatenate([dof_map[arm] for arm in geom.arms])
            lb = np.zeros(3) 
            ub = np.zeros(3) 
        else:
            dof_inds = np.concatenate([dof_map[arm] for arm in geom.arms])
            lb = np.zeros(6+sum([len(dof_map[arm]) for arm in geom.arms])+sum([geom.gripper_dim(arm) for arm in geom.arms]))
            ub = np.zeros(6+sum([len(dof_map[arm]) for arm in geom.arms])+sum([geom.gripper_dim(arm) for arm in geom.arms]))

        if delta:
            base_move = geom.get_base_move_limit()
            base_move = [-base_move, base_move]
        else:
            base_move = geom.get_base_limit()
        
        cur_ind = 0
        for attr, inds in self.attr_inds[self.robot]:
            ninds = len(inds)
            if attr == 'pose':
                lb[cur_ind:cur_ind+ninds] = base_move[0]
                ub[cur_ind:cur_ind+ninds] = base_move[1]
            elif attr == 'rotation':
                lb[cur_ind:cur_ind+ninds] = -4 * np.pi
                ub[cur_ind:cur_ind+ninds] = 4 * np.pi
            elif attr in geom.arms:
                arm_lb, arm_ub = geom.get_joint_limits(attr)
                lb[cur_ind:cur_ind+ninds] = arm_lb
                ub[cur_ind:cur_ind+ninds] = arm_ub
            elif attr in geom.ee_link_names.values():
                if delta:
                    gripper_lb, gripper_ub = -10, 10
                else:
                    gripper_lb = -1 # geom.get_gripper_closed_val()
                    gripper_ub = 1 # geom.get_gripper_open_val()
                lb[cur_ind:cur_ind+ninds] = gripper_lb
                ub[cur_ind:cur_ind+ninds] = gripper_ub
            elif ee_only:
                lb[cur_ind:cur_ind+ninds] = self.lb * geom.get_joint_move_factor()
                ub[cur_ind:cur_ind+ninds] = self.ub * geom.get_joint_move_factor()
            cur_ind += ninds
        '''
        inds = geom.dof_inds['pose']
        lb[inds] = base_move[0]
        ub[inds] = base_move[1]

        for arm in geom.arms:
            arm_lb, arm_ub = geom.get_joint_limits(arm)
            inds = geom.dof_inds[arm]
            lb[inds] = arm_lb
            ub[inds] = arm_ub
            gripper = geom.ee_link_names[arm]
            gripper_lb = geom.get_gripper_closed_val()
            gripper_ub = geom.get_gripper_open_val()
            inds = geom.dof_inds[gripper]
            lb[inds] = gripper_lb
            ub[inds] = gripper_ub
        '''

        if delta:
            joint_move = (ub-lb)/geom.get_joint_move_factor()
            # Setup the Equation so that: Ax+b < val represents
            # |base_pose_next - base_pose| <= const.BASE_MOVE
            # |joint_next - joint| <= joint_movement_range/const.JOINT_MOVE_FACTOR
            val = np.concatenate((joint_move, joint_move)).reshape((-1,1))
            A = np.eye(len(val)) - np.eye(len(val), k=len(val)//2) - np.eye(len(val), k=-len(val)//2)
            self.base_step = base_move 
            self.joint_step = joint_move
        else:
            val = np.concatenate((-lb, ub)).reshape((-1,1))
            A_lb_limit = -np.eye(len(lb))
            A_ub_limit = np.eye(len(ub))
            A = np.vstack((A_lb_limit, A_ub_limit))
        b = np.zeros((len(val),1))
        self.lower_limit = lb
        return A, b, val


    def get_arm_jac(self, arm_jacs, base_jac, obj_jac):
        dim = list(arm_jacs.values())[0].shape[0]
        jacobian = np.zeros((dim, self.attr_dim))
        for arm in arm_jacs:
            inds = self.attr_map[self.robot, arm]
            jacobian[:, inds] = arm_jacs[arm]
        inds = self.attr_map[self.robot, 'pose']
        jacobian[:, inds] = base_jac

        #ee_attr = '{}_ee_pos'.format(self.arm)
        #ee_rot_attr = '{}_ee_rot'.format(self.arm)
        #if (self.robot, ee_attr) in self.attr_map:
        #    inds = self.attr_map[self.robot, ee_attr]
        #    jacobian[:,inds] = -obj_jac[:,:3]
        #if (self.robot, ee_rot_attr) in self.attr_map:
        #    inds = self.attr_map[self.robot, ee_rot_attr]
        #    jacobian[:,inds] = -obj_jac[:,3:]

        if hasattr(self, 'obj'):
            inds = self.attr_map[self.obj, 'pose']
            jacobian[:, inds] = obj_jac[:,:3]
            inds = self.attr_map[self.obj, 'rotation']
            jacobian[:, inds] = obj_jac[:,3:]
        elif hasattr(self, 'targ'):
            inds = self.attr_map[self.targ, 'value']
            jacobian[:, inds] = obj_jac[:,:3]
            inds = self.attr_map[self.targ, 'rotation']
            jacobian[:, inds] = obj_jac[:,3:]
        elif hasattr(self, 'ee_ref') and self.ee_ref:
            inds = self.attr_map[self.robot, '{}_ee_pos'.format(self.arm)]
            jacobian[:, inds] = obj_jac[:,:3]
            inds = self.attr_map[self.robot, '{}_ee_rot'.format(self.arm)]
            jacobian[:, inds] = obj_jac[:,3:]
        return jacobian


class CollisionPredicate(RobotPredicate):

    #@profile
    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = const.DIST_SAFE, debug = False, ind0=0, ind1=1, tol=const.COLLISION_TOL, priority=0):
        self._debug = debug
        if self._debug:
            self._env.SetViewer("qtcoin")
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        self._plot_handles = []
        # self._cache = {}
        super(CollisionPredicate, self).__init__(name, e, attr_inds, params, expected_param_types, tol=tol, priority = priority)

    #@profile
    def robot_self_collision(self, x):
        # Parse the pose value
        self._plot_handles = []
        flattened = tuple(x.flatten())
        # cache prevents plotting
        # if flattened in self._cache and not self._debug:
        #     return self._cache[flattened]

        # Set pose of each rave body
        robot = self.params[self.ind0]
        robot_body = self._param_to_body[robot]
        self.set_robot_poses(x, robot_body)

        collisions = p.getClosestPoints(robot_body.body_id, robot_body.body_id, const.MAX_CONTACT_DISTANCE)

        # Calculate value and jacobian
        col_val, col_jac = self._calc_self_grad_and_val(robot_body, collisions)
        # set active dof value back to its original state (For successive function call)
        # self._cache[flattened] = (col_val.copy(), col_jac.copy())
        # print "col_val", np.max(col_val)
        return col_val, col_jac

    #@profile
    def robot_obj_collision(self, x):
        # Parse the pose value
        if np.any(np.isnan(x)):
            x[np.isnan(x)] = 0.
        self._plot_handles = []
        flattened = tuple(x.flatten())
        # cache prevents plotting
        # if flattened in self._cache and not self._debug:
        #     return self._cache[flattened]

        # Set pose of each rave body
        robot = self.params[self.ind0]
        robot_body = self._param_to_body[robot]
        self.set_robot_poses(x, robot_body)

        obj = self.params[self.ind1]
        obj_body = self._param_to_body[obj]
        pos_inds, rot_inds = self.attr_map[obj, 'pose'], self.attr_map[obj, 'rotation']
        obj_pos, obj_rot = x[pos_inds], x[rot_inds]
        obj_body.set_pose(obj_pos, obj_rot)
        obj_body._pos = obj_pos.flatten() 
        obj_body._orn = obj_rot.flatten()
        if hasattr(obj.geom, 'dof_map'):
            dof_map = {}
            for attr in obj.geom.dof_map:
                if (obj, attr) in self.attr_map:
                    dof_map[attr] = x[self.attr_map[obj, attr]]
            if len(dof_map.keys()):
                obj.openrave_body.set_dof(dof_map)

        collisions = p.getClosestPoints(robot_body.body_id, obj_body.body_id, const.MAX_CONTACT_DISTANCE)
        # Calculate value and jacobian
        col_val, col_jac = self._calc_grad_and_val(robot_body, obj_body, collisions)
        obj_body._pos = None
        obj_body._orn = None
        return col_val, col_jac

    #@profile
    def obj_obj_collision(self, x):
        self._plot_handles = []
        flattened = tuple(x.round(5).flatten())
        # cache prevents plotting
        # if flattened in self._cache and not self._debug:
        #     return self._cache[flattened]

        # Parse the pose value
        can_pos, can_rot = x[:3], x[3:6]
        obstr_pos, obstr_rot = x[6:9], x[9:]
        # Set pose of each rave body
        can = self.params[self.ind0]
        obstr = self.params[self.ind1]
        can_body = self._param_to_body[can]
        obstr_body = self._param_to_body[obstr]
        can_body.set_pose(can_pos, can_rot)
        obstr_body.set_pose(obstr_pos, obstr_rot)
        collisions = p.getClosestPoints(can_body.body_id, obstr_body.body_id, const.MAX_CONTACT_DISTANCE)

        # Calculate value and jacobian
        col_val, col_jac = self._calc_obj_grad_and_val(can_body, obstr_body, collisions)
        # self._cache[flattened] = (col_val.copy(), col_jac.copy())
        return col_val, col_jac

    #@profile
    def robot_obj_held_collision(self, x):
        self._plot_handles = []
        flattened = tuple(x.round(5).flatten())
        # cache prevents plotting
        # if flattened in self._cache and not self._debug:
        #     return self._cache[flattened]

        robot = self.params[self.ind0]
        robot_body = self._param_to_body[robot]
        self.set_robot_poses(x, robot_body)
        pos_inds, rot_inds = self.attr_map[self.obj, 'pose'], self.attr_map[self.obj, 'rotation']
        held_pos_inds, held_rot_inds = self.attr_map[self.obstacle, 'pose'], self.attr_map[self.obstacle, 'rotation']
        can_pos, can_rot = x[pos_inds], x[rot_inds]
        held_pose, held_rot = x[held_pos_inds], x[held_rot_inds]

        obj_body = self._param_to_body[self.obj]
        obj_body.set_pose(can_pos, can_rot)

        held_body = self._param_to_body[self.obstacle]
        held_body.set_pose(held_pose, held_rot)

        collisions2 = p.getClosestPoints(held_body.body_id, obj_body.body_id, const.MAX_CONTACT_DISTANCE)
        collisions1 = p.getClosestPoints(robot_body.body_id, obj_body.body_id, const.MAX_CONTACT_DISTANCE)
        col_val1, col_jac1 = self._calc_grad_and_val(robot_body, held_body, collisions1)
        #col_jac1 = np.c_[col_jac1, np.zeros((len(self.col_link_pairs), 6))]

        # find collision between object and object held
        collisions2 = p.getClosestPoints(held_body.body_id, obj_body.body_id, const.MAX_CONTACT_DISTANCE)

        col_val2, col_jac2 = self._calc_obj_held_grad_and_val(robot_body, held_body, obj_body, collisions2)

        # Stack these val and jac, and return
        val = np.vstack((col_val1, col_val2))
        jac = np.vstack((col_jac1, col_jac2))
        # self._cache[flattened] = (val.copy(), jac.copy())
        return val, jac

    #@profile
    def _calc_grad_and_val(self, robot_body, obj_body, collisions):
        # Initialization
        links = []
        robot = self.params[self.ind0]
        obj = self.params[self.ind1]
        col_links = robot.geom.col_links
        obj_links = obj.geom.col_links
        if hasattr(obj_body, '_pos') and obj_body._pos is not None:
            pos, orn = obj_body._pos, obj_body._orn
        else:
            pos, orn = obj_body.current_pose()
        pos, orn = np.array(pos), np.array(orn)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(pos, orn)
        rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
        link_pair_to_col = {}
        sign = 1
        base_jac = np.eye(3)
        base_jac[:,2] = 0 
        for c in collisions:
            # Identify the collision points
            col_info = parse_robot_collision(c, robot, robot_body, obj_body, col_links)
            if col_info is None: continue
            distance, normal, linkRobot, linkObj, ptRobot, ptObj, robot_jac, robot_ang_jac = col_info 
            grad = np.zeros((1, self.attr_dim))
            for arm in robot.geom.arms:
                inds = robot.geom.get_free_inds(arm)
                grad[:, self.attr_map[robot, arm]] = np.dot(sign * normal, robot_jac[:,inds])
            grad[:, self.attr_map[robot, 'pose']] = np.dot(sign*normal, base_jac)
            col_vec =  -sign*normal

            # Calculate object pose jacobian
            inds = self.attr_map[obj, 'pose']
            grad[:, inds] = col_vec

            # Calculate object rotation jacobian
            inds = self.attr_map[obj, 'rotation']
            torque = ptObj - pos[:3]
            rot_vec = np.array([[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]])
            # obj_jac = np.c_[obj_jac, rot_vec]
            grad[:, inds] = rot_vec

            # Constructing gradient matrix
            link_pair_to_col[(linkRobot, linkObj)] = [self.dsafe - distance, grad, linkRobot, linkObj]
            # if self._debug:
            #     self.plot_collision(ptRobot, ptObj, distance)

        vals, greds = [], []
        for robot_link, obj_link in self.col_link_pairs:
            col_infos = link_pair_to_col.get((robot_link, obj_link), [self.dsafe - const.MAX_CONTACT_DISTANCE, np.zeros((1, self.attr_dim)), None, None])
            vals.append(col_infos[0])
            greds.append(col_infos[1])

        return np.array(vals).reshape((len(vals), 1)), np.array(greds).reshape((len(greds), self.attr_dim))

    #@profile
    def _calc_self_grad_and_val(self, robot_body, collisions):
        # Initialization
        links = []
        robot = self.params[self.ind0]
        col_links = robot.geom.col_links
        link_pair_to_col = {}
        for c in collisions:
            linkA, linkB = c[3], c[4] # c.linkIndexA, c.linkIndexB
            linkAParent, linkBParent = c[1], c[2] # c.bodyUniqueIdA, c.bodyUniqueIdB
            sign = 0
            if linkAParent == robot_body.body_id and linkBParent == obj_body.body_id:
                ptRobot, ptObj = c[5], c[6] # c.positionOnA, c.positionOnB
                linkRobot, linkObj = linkA, linkB
                sign = -1
            elif linkBParent == robot_body.body_id and linkAParent == obj_body.body_id:
                ptRobot, ptObj = c[6], c[5] # c.positionOnB, c.positionOnA
                linkRobot, linkObj = linkB, linkA
                sign = 1
            else:
                continue

            geom = robot_body._goem
            if linkRobot1 not in col_links or linkRobot2 not in col_links:
                continue

            # Obtain distance between two collision points, and their normal collision vector
            distance = c[8]
            normal = c[7]
            n_jnts = p.getNumJoints(robot_body.body_id)
            jnts = p.getJointStates(list(range(n_jnts)))[0]
            robot_jac, robot_ang_jac = p.calculateJacobian(robot_body.body_id,
                                                           linkRobot1,
                                                           ptRobot1,
                                                           objPositions=jnts,
                                                           objVelocities=np.zeros(n_jnts),
                                                           objAccelerations=np.zeros(n_jnts))

            grad = np.zeros((1, self.attr_dim))
            grad[:, :self.attr_dim] = np.dot(sign * normal, robot_jac)

            # Constructing gradient matrix
            link_pair_to_col[(linkRobot1, linkRobot2)] = [self.dsafe - distance, grad, linkRobot1, linkRobot2]
            # if self._debug:
            #     self.plot_collision(ptRobot, ptObj, distance)

        vals, greds = [], []
        for robot_link, obj_link in self.col_link_pairs:
            col_infos = link_pair_to_col.get((robot_link, obj_link), [self.dsafe - const.MAX_CONTACT_DISTANCE, np.zeros((1, self.attr_dim)), None, None])
            vals.append(col_infos[0])
            greds.append(col_infos[1])

        return np.array(vals).reshape((len(vals), 1)), np.array(greds).reshape((len(greds), self.attr_dim))


    #@profile
    def _calc_obj_grad_and_val(self, obj_body, obstr_body, collisions):
        """
        Calculates collisions between two objects
        """
        held_links = self.obj.geom.col_links
        obs_links = self.obstacle.geom.col_links

        link_pair_to_col = {}
        sign = 1
        for c in collisions:
            col_info = parse_collision(c, obstr_body, obj_body)
            if col_info is None: continue
            distance, normal, linkObstr, linkObj, ptObj, ptObstr = col_info 
            col_vec = -sign*normal

            # Calculate object pose jacobian
            obj_jac = np.array([normal])
            obj_pos, obj_orn = obj_body.current_pose()
            torque = ptObj - obj_pos

            # Calculate object rotation jacobian
            Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obj_pos, obj_orn)
            rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
            rot_vec = np.array([[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]])
            obj_jac = np.c_[obj_jac, -rot_vec]

            # Calculate obstruct pose jacobian
            obstr_jac = np.array([-normal])
            obstr_pos, obstr_orn = obstr_body.current_pose()
            torque = ptObstr - obstr_pos
            Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obj_pos, obj_orn)
            rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
            rot_vec = np.array([[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]])
            obstr_jac = np.c_[obstr_jac, rot_vec]
            # Constructing gradient matrix
            robot_grad = np.c_[obj_jac, obstr_jac]

            link_pair_to_col[(linkObj, linkObstr)] = [self.dsafe - distance, robot_grad]
            # if self._debug:
            #     self.plot_collision(ptObj, ptObstr, distance)

        vals, grads = [], []
        for robot_link, obj_link in self.obj_obj_link_pairs:
            col_infos = link_pair_to_col.get((robot_link, obj_link), [self.dsafe - const.MAX_CONTACT_DISTANCE, np.zeros((1, 12)), None, None])
            vals.append(col_infos[0])
            grads.append(col_infos[1])

        vals = np.vstack(vals)
        grads = np.vstack(grads)
        return vals, grads


    def _calc_obj_held_grad_and_val(self, robot_body, obj_body, obstr_body, collisions):
        """
        Calculates collision between the robot's held object and an obstacle; does NOT calculate robot object collision
        Computes both jacobian to move objects out of collision and to move robot's arm out of collision
        Chooses arm based off the minimum distance (i.e. assumes the object is being held by one arm) unless overridden in subclass
        """
        robot_links = self.robot.geom.col_links
        held_links = self.obj.geom.col_links
        obs_links = self.obstacle.geom.col_links

        anchors = {}
        manips = {}
        dists = {}
        cur_ind = 0
        arms = robot_body._geom.arms
        for arm in arms:
            anchors[arm] = []
            for jnt_id in robot_body._geom.get_arm_inds(arm):
                info = p.getJointInfo(robot_body.body_id, jnt_id)
                parent_id = info[-1]
                parent_frame_pos = info[14]
                axis = info[13]
                if parent_id >= 0:
                    parent_info = p.getLinkState(robot_body.body_id, parent_id)
                    parent_pos = np.array(parent_info[0])
                else:
                    parent_pos = np.zeros(3)
                anchors[arm].append((parent_frame_pos + parent_pos, axis))
            ee_state = p.getLinkState(robot_body.body_id, robot_body._geom.get_ee_link(arm))
            manips[arm] = np.array(ee_state[0])
            n_jnts = len(robot_body._geom.get_arm_inds(arm))
            obj_pos, _ = obj_body.current_pose()
            diff = np.linalg.norm(np.array(obj_pos) - manips[arm])
            dists[arm] = diff
        arm = min(arms, key=lambda a: dists[a])
        link_pair_to_col = {}
        lb, ub = robot_body._geom.get_arm_bnds(arm)
        sign = 1
        for c in collisions:
            col_info = parse_collision(c, obstr_body, obj_body)
            if col_info is None: continue
            distance, normal, linkObstr, linkObj, ptObj, ptObstr = col_info 
            # 12 -> 3 objPos, 3 objRot, 3obstrPos, 3 obstrRot
            grad = np.zeros((1, self.attr_dim))
            arm_jac = np.array([np.cross(a[1], ptObj - a[0]) for a in anchors[arm]]).T.copy()
            grad[:, lb:ub] = np.dot(sign * normal, arm_jac)
            
            # Calculate obstruct pose jacobian
            obstr_jac = -sign*normal
            obstr_pos, obstr_orn = obstr_body.current_pose()
            torque = ptObstr - obstr_pos
            Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obstr_pos, obstr_orn)
            rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
            rot_vec = np.array([[np.dot(np.cross(axis, torque), obstr_jac) for axis in rot_axises]])
            pos_inds, rot_inds = self.attr_map[self.obstacle, 'pose'], self.attr_map[self.obstacle, 'rotation']
            grad[:, pos_inds] = obstr_jac
            grad[:, rot_inds] = rot_vec

            # Calculate object_held pose jacobian
            obj_jac = sign*normal
            obj_pos, obj_orn = obj_body.current_pose()
            torque = ptObj - obj_pos
            # Calculate object rotation jacobian
            Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obj_pos, obj_orn)
            rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
            rot_vec = np.array([[np.dot(np.cross(axis, torque), obj_jac) for axis in rot_axises]])
            pos_inds, rot_inds = self.attr_map[self.obj, 'pose'], self.attr_map[self.obj, 'rotation']
            grad[:, pos_inds] = obj_jac
            grad[:, rot_inds] = rot_vec

            link_pair_to_col[(linkObj, linkObstr)] = [self.dsafe - distance, grad]
            # if self._debug:
            #     self.plot_collision(ptObj, ptObstr, distance)

        vals, grads = [], []
        for robot_link, obj_link in self.obj_obj_link_pairs:
            col_infos = link_pair_to_col.get((robot_link, obj_link), [self.dsafe - const.MAX_CONTACT_DISTANCE, np.zeros((1, self.attr_dim)), None, None])
            vals.append(col_infos[0])
            grads.append(col_infos[1])

        vals = np.vstack(vals)
        grads = np.vstack(grads)
        return vals, grads

    #@profile
    def test(self, time, negated=False, tol=None):
        if tol is None:
            tol = self.tol
        # This test is overwritten so that collisions can be calculated correctly
        if not self.is_concrete():
            return False
        if time < 0:
            raise PredicateException("Out of range time for predicate '%s'."%self)
        try:
            return self.neg_expr.eval(self.get_param_vector(time), tol=tol, negated = (not negated))
        except IndexError as err:
            ## this happens with an invalid time
            traceback.print_exception(*sys.exc_info())
            raise PredicateException("Out of range time for predicate '%s'."%self)

    #@profile
    def plot_cols(self, env, t):
        _debug = self._debug
        self._env = env
        self._debug = True
        self.robot_obj_collision(self.get_param_vector(t))
        self._debug = _debug

    #@profile
    def plot_collision(self, ptA, ptB, distance):
        handles = []
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.001,color=(1,0,0)))
            else:
                handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.001,color=(0,0,0)))
        self._plot_handles.extend(handles)


class PosePredicate(RobotPredicate):

    #@profile
    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = const.DIST_SAFE, debug = False, ind0=0, ind1=1, tol=const.POSE_TOL, active_range=(0,0), priority = 0):
        self._debug = debug
        if self._debug:
            self._env.SetViewer("qtcoin")
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        self.handle = []

        self.arm = params[0].geom.arms[0] # Default to first arm in the list
        self.mask = np.ones((3,1))
        self.axes = np.eye(3)
        self.mats = {}
        self.quats = {}
        self.inv_mats = {}
        if not hasattr(self, 'axis'): self.axis = np.array([0, 0, -1])
        for arm in params[0].geom.arms:
            axis = params[0].geom.get_gripper_axis(arm)
            quat = OpenRAVEBody.quat_from_v1_to_v2(axis, self.axis)
            self.quats[arm] = quat
            self.mats[arm] = T.quat2mat(quat)
            self.inv_mats[arm] = np.linalg.inv(self.mats[arm])

        super(PosePredicate, self).__init__(name, e, attr_inds, params, expected_param_types, tol=tol, active_range=active_range, priority = priority)
        self._init_include = False

    #@profile
    def abs_ee_pos_check_f(self, x, abs_pt):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        return self.abs_pos_error_f(obj_trans, robot_trans, abs_pt)

    #@profile
    def abs_ee_pos_check_jac(self, x, abs_pt):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        return self.abs_pos_error_jac(obj_trans, robot_trans, axises, arm_joints, abs_pt)
     
    #@profile
    def rel_ee_pos_check_f(self, x, rel_pt):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        return self.rel_pos_error_f(obj_trans, robot_trans, rel_pt)

    #@profile
    def rel_ee_pos_check_jac(self, x, rel_pt):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        return self.rel_pos_error_jac(obj_trans, robot_trans, axises, arm_joints, rel_pt)

    #@profile
    def rel_pos_error_f(self, obj_trans, robot_trans, rel_pt):
        """
        This function calculates the value of the displacement between center of gripper and a point relative to the object

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        rel_pt: offset between your target point and object's pose
        """
        gp = rel_pt
        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[gp, 1])[:3]
        dist_val = (robot_pos - obj_pos).reshape((3,1))
        if np.any(self.mask < 1):
            base_dist = dist_val
            rel_offset = self.mask.reshape((3,1)) * np.linalg.inv(obj_trans[:3,:3]).dot(dist_val)
            dist_val = obj_trans[:3,:3].dot(rel_offset).round(4)
        return dist_val

    #@profile
    def rel_pos_error_jac(self, obj_trans, robot_trans, axises, arm_joints, rel_pt):
        """
            This function calculates the jacobian of the displacement between center of gripper and a point relative to the object

            obj_trans: object's rave_body transformation
            robot_trans: robot gripper's rave_body transformation
            axises: rotational axises of the object
            arm_joints: list of robot joints
            rel_pt: offset between your target point and object's pose
        """
        robot_body = self.params[0].openrave_body
        gp = rel_pt
        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[gp, 1])[:3]

        # Calculate the joint jacobian
        arm_jac = compute_arm_pos_jac(arm_joints, robot_body, robot_pos)

        # Calculate jacobian for the robot base
        base_pos_jac = np.eye(3)[:,:2]
        base_rot_jac = np.cross(np.array([0, 0, 1]), robot_pos).reshape((3,1))
        base_jac = np.c_[base_pos_jac, base_rot_jac]

        # Calculate object jacobian
        obj_jac = -1 * np.array([np.cross(axis, obj_pos - obj_trans[:3,3]) for axis in axises]).T
        obj_jac = np.c_[-np.eye(3), obj_jac]

        # Create final jacobian matrix -> (Gradient checked to be correct)
        dist_jac = self.get_arm_jac({self.arm: arm_jac}, base_jac, obj_jac)

        return dist_jac

    #@profile
    def abs_pos_error_f(self, obj_trans, robot_trans, rel_pt):
        gp = rel_pt
        robot_pos = robot_trans[:3, 3]
        obj_pos = obj_trans[:3,3] + rel_pt 
        dist_val = (robot_pos - obj_pos).reshape((3,1))
        return dist_val

    #@profile
    def abs_pos_error_jac(self, obj_trans, robot_trans, axises, arm_joints, rel_pt):
        robot_body = self.params[0].openrave_body
        gp = rel_pt
        robot_pos = robot_trans[:3, 3]
        obj_pos = obj_trans[:3,3] + rel_pt 

        # Calculate the joint jacobian
        arm_jac = compute_arm_pos_jac(arm_joints, robot_body, robot_pos)

        # Calculate jacobian for the robot base
        base_pos_jac = np.eye(3)[:,:2]
        base_rot_jac = np.cross(np.array([0, 0, 1]), robot_pos).reshape((3,1))
        base_jac = np.c_[base_pos_jac, base_rot_jac]

        # Calculate object jacobian
        obj_jac = -1 * np.array([np.cross(axis, rel_pt) for axis in axises]).T
        obj_jac = np.c_[-np.eye(3), obj_jac]

        # Create final jacobian matrix -> (Gradient checked to be correct)
        dist_jac = self.get_arm_jac({self.arm: arm_jac}, base_jac, obj_jac)

        return dist_jac

    #@profile
    def ee_rot_check_f(self, x, offset=np.eye(3), robot_off=np.eye(3)):
        """
            This function is used to check whether End Effective pose's rotational axis is parallel to that of robot gripper

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        return self.rot_lock_f(obj_trans, robot_trans, offset, robot_off)

    #@profile
    def ee_rot_check_jac(self, x, robot_off=np.eye(3)):
        """
            This function is used to check whether End Effective pose's rotational axis is parallel to that of robot gripper

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        return self.rot_lock_jac(obj_trans, robot_trans, axises, arm_joints, robot_off=robot_off)

    #@profile
    def rot_lock_f(self, obj_trans, robot_trans, offset=np.eye(3), robot_off=np.eye(3)):
        """
            This function calculates the value of the angle
            difference between robot gripper's rotational axis and
            object's rotational axis

            obj_trans: object's rave_body transformation
            robot_trans: robot gripper's rave_body transformation
        """
        rot_vals = []
        local_dir = np.eye(3)
        for i in range(3):
            obj_dir = np.dot(obj_trans[:3,:3], local_dir[i])
            world_dir = robot_trans[:3,:3].dot(robot_off.dot(local_dir[i]))
            #world_dir = robot_off.dot(robot_trans[:3,:3].dot(local_dir[i]))
            rot_vals.append([np.dot(obj_dir, world_dir) - offset[i].dot(local_dir[i])])
        rot_val = np.vstack(rot_vals)
        return rot_val

    #@profile
    def rot_lock_jac(self, obj_trans, robot_trans, axises, arm_joints, robot_off=np.eye(3)):
        """
            This function calculates the jacobian of the angle
            difference between robot gripper's rotational axis and
            object's rotational axis

            obj_trans: object's rave_body transformation
            robot_trans: robot gripper's rave_body transformation
            axises: rotational axises of the object
            arm_joints: list of robot joints
        """
        rot_jacs = []
        robot_body = self.params[0].openrave_body
        for local_dir in np.eye(3):
            obj_dir = np.dot(obj_trans[:3,:3], local_dir)
            world_dir = robot_trans[:3,:3].dot(robot_off.dot(local_dir))
            #world_dir = robot_off.dot(robot_trans[:3,:3].dot(local_dir))

            # computing robot's jacobian
            sign = np.sign(np.dot(obj_dir, world_dir))
            arm_jac = compute_arm_rot_jac(arm_joints, robot_body, obj_dir, world_dir, sign)
            '''
            arm_jac = []
            for jnt_id in arm_joints:
                info = p.getJointInfo(robot_body.body_id, jnt_id)
                parent_id = info[-1]
                parent_frame_pos = info[14]
                axis = info[13]
                parent_info = p.getLinkState(robot_body.body_id, parent_id)
                parent_pos = np.array(parent_info[0])
                arm_jac.append(np.dot(obj_dir, np.cross(axis, world_dir)))
            arm_jac = np.array(arm_jac).T
            '''
            arm_jac = arm_jac.reshape((1, len(arm_joints)))
            base_jac = np.array(np.dot(obj_dir, np.cross([0,0,1], world_dir)))
            base_jac = np.c_[np.zeros((1,2)), base_jac.reshape((1,1))]

            # computing object's jacobian
            obj_jac = np.array([np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises])
            obj_jac = np.r_[[0,0,0], obj_jac].reshape((1, 6))

            # Create final jacobian matrix
            rot_jacs.append(self.get_arm_jac({self.arm: arm_jac}, base_jac, obj_jac))
        rot_jac = np.vstack(rot_jacs)
        return rot_jac

    #@profile
    def pos_check_f(self, x, rel_pt=np.zeros((3, ))):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        return self.rel_pos_error_f(obj_trans, robot_trans, rel_pt)

    #@profile
    def pos_check_jac(self, x, rel_pt=np.zeros((3, ))):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        return self.rel_pos_error_jac(obj_trans, robot_trans, axises, arm_joints, rel_pt)

    #@profile
    def rot_check_f(self, x):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        local_dir = np.array([0.,0.,1.])
        return self.rot_error_f(obj_trans, robot_trans, local_dir)

    #@profile
    def rot_check_jac(self, x):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        local_dir = np.array([0.,0.,1.])
        return self.rot_error_jac(obj_trans, robot_trans, axises, arm_joints, local_dir)

    #@profile
    def rot_error_f(self, obj_trans, robot_trans, local_dir, robot_dir = None, robot_off=np.eye(3)):
        """
            This function calculates the value of the rotational error between
            robot gripper's rotational axis and object's rotational axis

            obj_trans: object's rave_body transformation
            robot_trans: robot gripper's rave_body transformation
            axises: rotational axises of the object
            arm_joints: list of robot joints
        """
        if robot_dir is None:
            robot_dir = local_dir
        obj_dir = np.dot(obj_trans[:3,:3], local_dir)
        #world_dir = robot_trans[:3,:3].dot(robot_dir)
        world_dir = robot_trans[:3,:3].dot(robot_off.dot(robot_dir))
        obj_dir = obj_dir/np.linalg.norm(obj_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        rot_val = np.array([[np.abs(np.dot(obj_dir, world_dir)) - 1]])
        return rot_val

    #@profile
    def rot_error_jac(self, obj_trans, robot_trans, axises, arm_joints, local_dir, robot_dir = None, robot_off=np.eye(3)):
        """
            This function calculates the jacobian of the rotational error between
            robot gripper's rotational axis and object's rotational axis

            obj_trans: object's rave_body transformation
            robot_trans: robot gripper's rave_body transformation
            axises: rotational axises of the object
            arm_joints: list of robot joints
        """
        if robot_dir is None:
            robot_dir = local_dir
        obj_dir = np.dot(obj_trans[:3,:3], local_dir)
        #world_dir = robot_off.dot(robot_dir)
        world_dir = robot_trans[:3,:3].dot(robot_off.dot(robot_dir))
        obj_dir = obj_dir/np.linalg.norm(obj_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        sign = np.sign(np.dot(obj_dir, world_dir))

        robot_body = self.params[0].openrave_body
        # computing robot's jacobian
        arm_jac = compute_arm_rot_jac(arm_joints, robot_body, obj_dir, world_dir, sign)
        '''
        arm_jac = []
        for jnt_id in arm_joints:
            info = p.getJointInfo(robot_body.body_id, jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = np.array(parent_info[0])
            arm_jac.append(np.dot(obj_dir, np.cross(axis, sign * world_dir)))
        arm_jac = np.array(arm_jac).T
        '''

        arm_jac = arm_jac.reshape((1, len(arm_joints)))
        base_jac = sign*np.array(np.dot(obj_dir, np.cross([0,0,1], world_dir))).reshape((1,1))
        base_jac = np.c_[np.zeros((1,2)), base_jac]

        # computing object's jacobian
        obj_jac = np.array([np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises])
        obj_jac = sign*np.r_[[0,0,0], obj_jac].reshape((1, 6))

        # Create final jacobian matrix
        rot_jac = self.get_arm_jac({self.arm: arm_jac}, base_jac, obj_jac)
        return rot_jac

    #@profile
    def both_arm_pos_check_f(self, x):
        """
            This function is used to check whether:
                basket is at both robot gripper's center

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        robot_body = self.robot.openrave_body
        self.arm = "left"
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)

        l_ee_trans, l_arm_inds = self.get_robot_info(robot_body, 'left')
        r_ee_trans, r_arm_inds = self.get_robot_info(robot_body, 'right')
        l_arm_joints = l_arm_inds
        r_arm_joints = r_arm_inds

        rel_pt = np.array([0,2*const.BASKET_OFFSET,0])
        # rel_pt = np.array([0, 2*const.BASKET_NARROW_OFFSET,0])
        l_pos_val = self.rel_pos_error_f(r_ee_trans, l_ee_trans, rel_pt)
        rel_pt = np.array([0,-2*const.BASKET_OFFSET,0])
        # rel_pt = np.array([0, -2*const.BASKET_NARROW_OFFSET,0])
        r_pos_val = self.rel_pos_error_f(l_ee_trans, r_ee_trans, rel_pt)
        rel_pt = np.array([const.BASKET_OFFSET,self.grip_offset,0])
        # rel_pt = np.array([0, 0, -const.BASKET_NARROW_OFFSET])
        obj_pos_val = self.rel_pos_error_f(obj_trans, l_ee_trans, rel_pt)
        return np.vstack([l_pos_val, r_pos_val, obj_pos_val])

    #@profile
    def both_arm_pos_check_jac(self, x):
        """
            This function is used to check whether:
                basket is at both robot gripper's center

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """

        robot_body = self.robot.openrave_body
        self.set_robot_poses(x, robot_body)

        l_ee_trans, l_arm_inds = self.get_robot_info(robot_body, 'left')
        r_ee_trans, r_arm_inds = self.get_robot_info(robot_body, 'right')
        # left_arm_focused
        rel_pt = np.array([0,2*const.BASKET_OFFSET,0])
        # rel_pt = np.array([0,2*const.BASKET_NARROW_OFFSET,0])
        robot_pos = l_ee_trans[:3, 3]
        obj_pos = np.dot(r_ee_trans, np.r_[rel_pt, 1])[:3]

        l_arm_jac = []
        for jnt_id in l_arm_inds:
            info = p.getJointInfo(robot_body.body_id, jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = np.array(parent_info[0])
            l_arm_jac.append(np.cross(axis, robot_pos - (parent_pos + parent_frame_pos)))
        l_arm_jac = np.array(l_arm_jac).T

        r_arm_jac = []
        for jnt_id in r_arm_inds:
            info = p.getJointInfo(robot_body.body_id, jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = np.array(parent_info[0])
            r_arm_jac.append(np.cross(axis, obj_pos - (parent_pos + parent_frame_pos)))
        r_arm_jac = -np.array(r_arm_jac).T

        l_pos_jac = np.hstack([l_arm_jac, np.zeros((3,1)), r_arm_jac, np.zeros((3, 8))])
        # right_arm_focused
        rel_pt = np.array([0,-2*const.BASKET_OFFSET,0])
        # rel_pt = np.array([0,-2*const.BASKET_NARROW_OFFSET,0])
        robot_pos = r_ee_trans[:3, 3]
        obj_pos = np.dot(l_ee_trans, np.r_[rel_pt, 1])[:3]

        l_arm_jac = []
        for jnt_id in l_arm_inds:
            info = p.getJointInfo(robot_body.body_id, jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = np.array(parent_info[0])
            l_arm_jac.append(np.cross(axis, obj_pos - (parent_pos + parent_frame_pos)))
        l_arm_jac = -np.array(l_arm_jac).T

        r_arm_jac = []
        for jnt_id in r_arm_inds:
            info = p.getJointInfo(robot_body.body_id, jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = np.array(parent_info[0])
            r_arm_jac.append(np.cross(axis, robot_pos - (parent_pos + parent_frame_pos)))
        r_arm_jac = np.array(r_arm_jac).T

        r_pos_jac = np.hstack([l_arm_jac, np.zeros((3,1)), r_arm_jac, np.zeros((3, 8))])

        self.arm = "left"
        obj_body = self.obj.openrave_body
        obj_body.set_pose(x[-6: -3], x[-3:])
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        rel_pt = np.array([const.BASKET_OFFSET,self.grip_offset,0])
        # rel_pt = np.array([0, 0, -const.BASKET_NARROW_OFFSET])
        obj_pos_jac = self.rel_pos_error_jac(obj_trans, l_ee_trans, axises, arm_joints, rel_pt)

        return np.vstack([l_pos_jac, r_pos_jac, obj_pos_jac])

    #@profile
    def both_arm_rot_check_f(self, x):
        """
            This function is used to check whether:
                object is at robot gripper's center

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        offset = np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1]])
        # offset = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        obj_body = self.obj.openrave_body
        obj_body.set_pose(x[-6: -3], x[-3:])
        self.arm = "left"
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        l_rot_val = self.rot_lock_f(obj_trans, robot_trans, offset)
        self.arm = "right"
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)

        r_rot_val = self.rot_lock_f(obj_trans, robot_trans, offset)

        return np.vstack([l_rot_val, r_rot_val])

    #@profile
    def both_arm_rot_check_jac(self, x):
        """
            This function is used to check whether:
                object is at robot gripper's center

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        obj_body = self.obj.openrave_body
        obj_body.set_pose(x[-6: -3], x[-3:])
        self.arm = "left"
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        l_rot_jac = self.rot_lock_jac(obj_trans, robot_trans, axises, arm_joints)
        self.arm = "right"
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        r_rot_jac = self.rot_lock_jac(obj_trans, robot_trans, axises, arm_joints)

        return np.vstack([l_rot_jac, r_rot_jac])


### LINEAR CONSTRAINTS

class At(ExprPredicate):
    """
        Format: # At, Can, Target

        Non-robot related
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.obj, self.target = params
        k = 'value' if self.target.is_symbol() else 'pose'
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([0,1,2], dtype='int32')),
                                             ("rotation", np.array([0,1,2], dtype='int32'))]),
                                 (self.target, [(k, np.array([0,1,2], dtype='int32')),
                                                ("rotation", np.array([0,1,2], dtype='int32'))])])

        A = np.c_[np.eye(6), -np.eye(6)]
        b, val = np.zeros((6, 1)), np.zeros((6, 1))
        if not self.target.is_symbol() and hasattr(self.target.geom, 'height'):
            h1 = self.obj.geom.height if hasattr(self.obj.geom, 'height') else self.obj.geom.radius
            h2 = self.target.geom.height
            b[2,0] = -(h1 + h2)

        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)

        super(At, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True

class AtRot(ExprPredicate):
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.obj, self.target = params
        attr_inds = OrderedDict([(self.obj, [("rotation", np.array([0,1,2], dtype='int32'))]),
                                 (self.target, [("rotation", np.array([0,1,2], dtype='int32'))])])

        A = np.c_[np.eye(3), -np.eye(3)]
        b, val = np.zeros((3, 1)), np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)

        super(AtRot, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True


class AtInit(At):
    def test(self, time, negated=False, tol=1e-4):
        return True

    def hl_test(self, time, negated=False, tol=1e-4):
        return True


class AtPose(ExprPredicate):
    """
        Format: # At, Can, Target

        Non-robot related
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.obj, self.target = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([0,1,2], dtype='int32'))]),
                                 (self.target, [("value", np.array([0,1,2], dtype='int32'))])])

        A = np.c_[np.r_[np.eye(3), -np.eye(3)], np.r_[-np.eye(3), np.eye(3)]]
        b, val = np.zeros((6, 1)), np.ones((6, 1))*1e-3
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(AtPose, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True

class Above(ExprPredicate):
    """
        Format: # At, Can, Target

        Non-robot related
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.obj, self.target = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([1,2], dtype='int32')),
                                             ("rotation", np.array([0,1,2], dtype='int32'))]),
                                 (self.target, [("value", np.array([0,1,2], dtype='int32')),
                                                ("rotation", np.array([1,2], dtype='int32'))])])

        A = np.c_[np.eye(5), -np.eye(5)]
        b, val = np.zeros((5, 1)), np.zeros((5, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)

        #A = np.c_[np.r_[np.eye(6), -np.eye(6)], np.r_[-np.eye(6), np.eye(6)]]
        #b, val = np.zeros((12, 1)), np.ones((12, 1))*1e-2
        #aff_e = AffExpr(A, b)
        #e = LEqExpr(aff_e, val)

        super(Above, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True


class Near(ExprPredicate):
    """
        Format: # At, Can, Target

        Non-robot related
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.obj, self.target = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([0,1,2], dtype='int32'))]),
                                 (self.target, [("value", np.array([0,1,2], dtype='int32'))])])
        #attr_inds = OrderedDict([(self.obj, [("pose", np.array([0,1,2], dtype='int32')),
        #                                     ("rotation", np.array([0,1,2], dtype='int32'))]),
        #                         (self.target, [("value", np.array([0,1,2], dtype='int32')),
        #                                        ("rotation", np.array([0,1,2], dtype='int32'))])])

        A = np.c_[np.r_[np.eye(3), -np.eye(3)], np.r_[-np.eye(3), np.eye(3)]]
        b, val = np.zeros((6, 1)), NEAR_TOL*np.ones((6, 1))
        #A = np.c_[np.r_[np.eye(6), -np.eye(6)], np.r_[-np.eye(6), np.eye(6)]]
        #b, val = np.zeros((12, 1)), np.ones((12, 1))
        #val[:3] *= NEAR_TOL
        #val[3:6] *= NEAR_ROT_TOL
        #val[6:9] *= NEAR_TOL
        #val[9:12] *= NEAR_ROT_TOL
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        #A = np.c_[np.r_[np.eye(3), -np.eye(3)], np.r_[-np.eye(3), np.eye(3)]]
        #b, val = np.zeros((6, 1)), NEAR_TOL*np.ones((6, 1))
        #aff_e = AffExpr(A, b)
        #e = LEqExpr(aff_e, val)

        super(Near, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True


class HLAnchor(ExprPredicate):
    """
        Format: # HLAnchor, RobotPose, RobotPose

        Non-robot related
        Should Always return True
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        attr_inds = self.attr_inds

        A = np.zeros((self.attr_dim, self.attr_dim))
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(HLAnchor, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True

class PoseAdjacent(ExprPredicate):
    """
        Format: PoseAdjacent, RobotPose, RobotPose
        Robot related
        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.robot_pose1, self.robot_pose2 = params
        self.attr_inds =  OrderedDict([(self.robot_pose1, [('value', np.array([0,1,2], dtype='int32'))]), (self.robot_pose2, [('value', np.array([0,1,2], dtype='int32'))])])
        self.attr_dim = 3

        # A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        A = np.zeros((3,6))
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(PoseAdjacent, self).__init__(name, e, self.attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = False
        
class RobotAt(ExprPredicate):
    """
        Format: RobotAt, Robot, RobotPose

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.robot, self.robot_pose = params
        attrs = self.robot.geom.arms + self.robot.geom.grippers + ['pose', 'rotation']
        attr_inds, attr_dim = init_robot_pred(self, self.robot, [], [self.robot_pose], attrs={self.robot: attrs})
        # A = np.c_[np.r_[np.eye(self.attr_dim), -np.eye(self.attr_dim)], np.r_[-np.eye(self.attr_dim), np.eye(self.attr_dim)]]
        # b, val = np.zeros((self.attr_dim*2, 1)), np.ones((self.attr_dim*2, 1))*1e-3
        # aff_e = AffExpr(A, b)
        # e = LEqExpr(aff_e, val)

        A = np.c_[np.eye(self.attr_dim//2), -np.eye(self.attr_dim//2)]
        b, val = np.zeros((self.attr_dim//2, 1)), np.zeros((self.attr_dim//2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(RobotAt, self).__init__(name, e, self.attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True

class IsMP(RobotPredicate):
    """
        Format: IsMP Robot (Just the Robot Base)

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            setup_mov_limit_check[Function]: function that sets constraint matrix
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, = params
        attrs = self.robot.geom.arms + self.robot.geom.grippers + ['pose', 'rotation']
        attr_inds, attr_dim = init_robot_pred(self, self.robot, [], attrs={self.robot: attrs})
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = self.attr_inds
        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        A, b, val = self.setup_mov_limit_check(delta=True)
        e = LEqExpr(AffExpr(A, b), val)
        super(IsMP, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False
        self._nonrollout = True

class EEIsMP(RobotPredicate):
    """
        Format: IsMP Robot (Just the Robot Base)

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            setup_mov_limit_check[Function]: function that sets constraint matrix
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, = params
        attrs = []
        self.lb = -const.EE_STEP
        self.ub = const.EE_STEP
        for arm in self.robot.geom.arms:
            attrs.append('{}_ee_pos'.format(arm))
        attr_inds, attr_dim = init_robot_pred(self, self.robot, [], attrs={self.robot: attrs})
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = self.attr_inds
        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        A, b, val = self.setup_mov_limit_check(delta=True, ee_only=True)
        e = LEqExpr(AffExpr(A, b), val)
        super(EEIsMP, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False
        self._nonrollout = True

class WithinJointLimit(RobotPredicate):
    """
        Format: WithinJointLimit Robot

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            setup_mov_limit_check[Function]: function that sets constraint matrix
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, = params
        attrs = self.robot.geom.arms + ["pose", "rotation"] + self.robot.geom.grippers
        attr_inds, attr_dim = init_robot_pred(self, self.robot, attrs={self.robot: attrs})

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        A, b, val = self.setup_mov_limit_check()
        e = LEqExpr(AffExpr(A, b), val)
        super(WithinJointLimit, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = False
        self._nonrollout = True

class WipedSurface(ExprPredicate):
    """
        Format: WipedSurface RobotPose

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self.region_pose, = params
        self.attr_inds =  OrderedDict([(self.region_pose, [('value', np.array([0,1,2], dtype='int32'))])])
        self.attr_dim = 3

        A = np.zeros((3,3))
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(WipedSurface, self).__init__(name, e, self.attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = False

class Stationary(ExprPredicate):
    """
        Format: Stationary, Can

        Non-robot related
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self.obj,  = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([0,1,2], dtype='int32')),
                                             ("rotation", np.array([0,1,2], dtype='int32'))])])

        A = np.c_[np.eye(6), -np.eye(6)]
        b, val = np.zeros((6, 1)), np.zeros((6, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False
        self._init_include = False

class Fixed(Stationary):
    pass

class StationaryRot(ExprPredicate):
    """
        Format: Stationary, Can

        Non-robot related
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self.obj,  = params
        attr_inds = OrderedDict([(self.obj, [("rotation", np.array([0,1,2], dtype='int32'))])])

        A = np.c_[np.eye(3), -np.eye(3)]
        b, val = np.zeros((3, 1)), np.zeros((3, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryRot, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False

class StationaryBase(ExprPredicate):
    """
        Format: StationaryBase, Robot (Only Robot Base)

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self.robot,  = params
        self.attr_inds =  OrderedDict([(self.robot, [('pose', np.array([0,1,2], dtype='int32'))])])
        self.attr_dim = 3

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryBase, self).__init__(name, e, self.attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False

class StationaryBasePos(ExprPredicate):
    """
        Format: StationaryBase, Robot (Only Robot Base)

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self.robot,  = params
        self.attr_dim = 2
        self.attr_inds =  OrderedDict([(self.robot, [('pose', np.array([0,1], dtype='int32'))])])

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryBase, self).__init__(name, e, self.attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False

class StationaryArms(ExprPredicate):
    """
        Format: StationaryArms, Robot (Only Robot Arms)

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self.robot,  = params
        attr_inds = self.attr_inds
        if not hasattr(self, 'arms'): self.arms = self.robot.geom.arms
        attr_inds, attr_dim = init_robot_pred(self, self.robot, [], attrs={params[0]:self.arms})

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryArms, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False

class StationaryArm(ExprPredicate):
    """
        Format: StationaryArms, Robot (Only Robot Arms)

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self.robot,  = params
        if not hasattr(self, 'arm'): self.arm = self.robot.geom.arms[0]
        attr_inds, attr_dim = init_robot_pred(self, self.robot, [], attrs={params[0]:[self.arm]})

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryArm, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False

class StationaryLeftArm(StationaryArm):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.arm = 'left'
        super(StationaryLeftArm, self).__init__(name, params, expected_param_types, env)

class StationaryRightArm(StationaryArm):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.arm = 'right'
        super(StationaryRightArm, self).__init__(name, params, expected_param_types, env)

class StationaryW(ExprPredicate):
    """
        Format: StationaryW, Obstacle

        Non-robot related
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.w, = params

        cur_attr_dim = 0
        attr_inds = const.ATTRMAP[self.w._type]
        for (attr, inds) in attr_inds:
            cur_attr_dim += len(inds)

        w_inds = [(self.w, [(attr, np.array(inds, dtype='int32')) for (attr, inds) in attr_inds])]
        attr_inds = OrderedDict(w_inds)

        A = np.c_[np.eye(cur_attr_dim), -np.eye(cur_attr_dim)]
        b = np.zeros((cur_attr_dim, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryW, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False

class StationaryWNEq(ExprPredicate):
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.w, self.obj = params

        cur_attr_dim = 0
        attr_inds = const.ATTRMAP[self.w._type]
        for (attr, inds) in attr_inds:
            cur_attr_dim += len(inds)

        w_inds = [(self.w, [(attr, np.array(inds, dtype='int32')) for (attr, inds) in attr_inds])]
        attr_inds = OrderedDict(w_inds)

        A = np.c_[np.eye(cur_attr_dim), -np.eye(cur_attr_dim)]
        if self.w == self.obj: A[:] = 0.
        b = np.zeros((cur_attr_dim, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryWNEq, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False

class StationaryWBase(ExprPredicate):
    """
        Format: StationaryW, Obstacle

        Non-robot related
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.w, = params
        attr_inds = OrderedDict([(self.w, [("pose", np.array([0, 1, 2], dtype='int32')),
                                           ("rotation", np.array([0, 1, 2], dtype='int32'))])])
        A = np.c_[np.eye(6), -np.eye(6)]
        b = np.zeros((6, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryWBase, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False

class StationaryNEq(ExprPredicate):
    """
        Format: StationaryNEq, Can, Can(Hold)

        Non-robot related
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        if len(params) > 1:
            self.obj, self.obj_held = params
        else:
            self.obj, = params
            self.obj_held = self.obj

        attr_inds = OrderedDict([(self.obj, [("pose", np.array([0, 1, 2], dtype='int32')),
                                             ("rotation", np.array([0, 1, 2], dtype='int32'))])])

        if self.obj.name == self.obj_held.name:
            A = np.zeros((1, 12))
            b = np.zeros((1, 1))
        else:
            A = np.c_[np.eye(6), -np.eye(6)]
            b = np.zeros((6, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryNEq, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)
        self.spacial_anchor = False

class StationaryXZ(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self.obj,  = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([0,2], dtype='int32'))])])

        A = np.c_[np.eye(2), -np.eye(2)]
        b, val = np.zeros((2, 1)), np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryXZ, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2)
        self.spacial_anchor = False
        self._init_include = False

class StationaryYZ(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self.obj,  = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([1,2], dtype='int32'))])])

        A = np.c_[np.eye(2), -np.eye(2)]
        b, val = np.zeros((2, 1)), np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryYZ, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2)
        self.spacial_anchor = False
        self._init_include = False

class StationaryXY(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self.obj,  = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([0,1], dtype='int32'))])])

        A = np.c_[np.eye(2), -np.eye(2)]
        b, val = np.zeros((2, 1)), np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryYZ, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2)
        self.spacial_anchor = False
        self._init_include = False

class GraspValid(ExprPredicate):
    """
        Format: GraspValid EEPose Target

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.ee_pose, self.target = params
        attr_inds = self.attr_inds

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim,1)), np.zeros((self.attr_dim,1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(GraspValid, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True


class SlideDoorAt(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        assert params[1].geom.hinge_type == 'prismatic'
        self.handle, self.door = params
        ind = 0
        for val in self.door.geom.open_dir:
            if val != 0: break
            ind += 1

        attr_inds = OrderedDict([(self.handle, [("pose", np.array([0, 1, 2], dtype='int32'))]),
                                 (self.door, [("pose", np.array([0, 1, 2], dtype='int32')),
                                                ("hinge", np.array([0], dtype='int32'))])])

        A = np.zeros((3,7))
        for i in range(3):
            A[i, i] = 1.
            A[i, 3+i] = -1.
            if i == ind:
                A[i, -1] = -1.

        b = -np.array(self.door.geom.handle_pos).reshape((-1,1))
        val = np.zeros((3,1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(SlideDoorAt, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True


class OpenSlideDoorAt(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        assert params[1].geom.hinge_type == 'prismatic'
        self.handle, self.door = params
        ind = 0
        for val in self.door.geom.open_dir:
            if val != 0: break
            ind += 1

        attr_inds = OrderedDict([(self.handle, [("pose", np.array([0, 1, 2], dtype='int32'))]),
                                 (self.door, [("pose", np.array([0, 1, 2], dtype='int32')),
                                                ("hinge", np.array([0], dtype='int32'))])])

        A = np.zeros((3,7))
        for i in range(3):
            A[i, i] = 1.
            A[i, 3+i] = -1.
            if i == ind:
                A[i, -1] = -1.

        A = np.r_[A, -A]
        b = -np.array(self.door.geom.open_handle_pos).reshape((-1,1))
        b = np.r_[b, -b]
        val = np.array([self.door.geom.push_open_region])
        val = np.r_[val, val].reshape((-1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(OpenSlideDoorAt, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True


class CloseSlideDoorAt(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        assert params[1].geom.hinge_type == 'prismatic'
        self.handle, self.door = params
        ind = 0
        for val in self.door.geom.close_dir:
            if val != 0: break
            ind += 1

        attr_inds = OrderedDict([(self.handle, [("pose", np.array([0, 1, 2], dtype='int32'))]),
                                 (self.door, [("pose", np.array([0, 1, 2], dtype='int32')),
                                                ("hinge", np.array([0], dtype='int32'))])])

        A = np.zeros((3,7))
        for i in range(3):
            A[i, i] = 1.
            A[i, 3+i] = -1.
            if i == ind:
                A[i, -1] = -1.

        A = np.r_[A, -A]
        b = -np.array(self.door.geom.close_handle_pos).reshape((-1,1))
        b = np.r_[b, -b]
        val = np.array([self.door.geom.push_close_region])
        val = np.r_[val, val].reshape((-1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(CloseSlideDoorAt, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True


class SlideDoorAtOpen(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        assert params[1].geom.hinge_type == 'prismatic'
        self.handle, self.door = params
        ind = 0
        for val in self.door.geom.open_dir:
            if val != 0: break
            ind += 1

        attr_inds = OrderedDict([(self.door, [("hinge", np.array([0], dtype='int32'))])])

        open_val = self.door.geom.open_val
        close_val = self.door.geom.close_val
        sgn = 1 if open_val < close_val else -1
        self.coeff = 1. # 1e-3 / (np.abs(open_val-close_val) / 2.)
        A = sgn*self.coeff*np.array([[1.]])
        b = sgn*self.coeff*np.array([[-self.door.geom.open_val]])
        val = np.zeros((1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        #neg_b = self.coeff*np.array([[-self.door.geom.handle_pos[ind]], [0.]])
        #neg_aff_e = AffExpr(A, neg_b)
        #self.neg_expr = EgExpr(neg_aff_e, val)
        super(SlideDoorAtOpen, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True
        self._init_include = False


class SlideDoorAtClose(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        assert params[1].geom.hinge_type == 'prismatic'
        self.handle, self.door = params
        ind = 0
        for val in self.door.geom.open_dir:
            if val != 0: break
            ind += 1

        attr_inds = OrderedDict([(self.door, [("hinge", np.array([0], dtype='int32'))])])

        open_val = self.door.geom.open_val
        close_val = self.door.geom.close_val
        sgn = 1 if open_val > close_val else -1
        self.coeff = 1. # 1e-3 / (np.abs(open_val-close_val) / 2.)
        A = sgn*self.coeff*np.array([[1.]])
        b = sgn*self.coeff*np.array([[-self.door.geom.close_val]])
        val = np.zeros((1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(SlideDoorAtClose, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True
        self._init_include = False

    #def test(self, time, negated=False, tol=None):
    #    if not self.is_concrete(): return False
    #    if tol is None: tol = self.tol
    #    expr = self.neg_expr if negated else expr
    #    return expr.eval(self.get_param_vector(time), tol=tol, negated=False)



class SlideDoorOpen(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        assert params[1].geom.hinge_type == 'prismatic'
        self.handle, self.door = params
        ind = 0
        for val in self.door.geom.open_dir:
            if val != 0: break
            ind += 1

        attr_inds = OrderedDict([(self.door, [("hinge", np.array([0], dtype='int32'))])])

        open_val = self.door.geom.open_val
        close_val = self.door.geom.close_val
        sgn = 1 if open_val < close_val else -1
        self.coeff = 1. # 1e-3 / (np.abs(open_val-close_val) / 2.)
        A = sgn*self.coeff*np.array([[1.]])
        b = sgn*self.coeff*np.array([[-self.door.geom.open_thresh]])
        val = np.zeros((1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        #neg_b = self.coeff*np.array([[-self.door.geom.handle_pos[ind]], [0.]])
        #neg_aff_e = AffExpr(A, neg_b)
        #self.neg_expr = EgExpr(neg_aff_e, val)
        super(SlideDoorOpen, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True
        self._init_include = False


class SlideDoorClose(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        assert params[1].geom.hinge_type == 'prismatic'
        self.handle, self.door = params
        ind = 0
        for val in self.door.geom.open_dir:
            if val != 0: break
            ind += 1

        attr_inds = OrderedDict([(self.door, [("hinge", np.array([0], dtype='int32'))])])

        open_val = self.door.geom.open_val
        close_val = self.door.geom.close_val
        sgn = 1 if open_val > close_val else -1
        self.coeff = 1. # 1e-3 / (np.abs(open_val-close_val) / 2.)
        A = sgn*self.coeff*np.array([[1.]])
        b = sgn*self.coeff*np.array([[-self.door.geom.close_thresh]])
        val = np.zeros((1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(SlideDoorClose, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True
        self._init_include = False

    #def test(self, time, negated=False, tol=None):
    #    if not self.is_concrete(): return False
    #    if tol is None: tol = self.tol
    #    expr = self.neg_expr if negated else expr
    #    return expr.eval(self.get_param_vector(time), tol=tol, negated=False)


class InSlideDoor(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        assert params[1].geom.hinge_type == 'prismatic'
        self.obj, self.door = params
        #ind = 0
        #for val in self.door.geom.open_dir:
        #    if val != 0: break
        #    ind += 1

        attr_inds = OrderedDict([(self.obj, [("pose", np.array([0, 1, 2], dtype='int32')),
                                             ("rotation", np.array([0, 1, 2], dtype='int32'))]),
                                 (self.door, [("pose", np.array([0, 1, 2], dtype='int32')),
                                              ("rotation", np.array([0, 1, 2], dtype='int32'))]),
                                 ])

        A = np.c_[np.eye(6), -np.eye(6)]
        A[-3:,-3:] = 0.
        A = np.r_[A, -A]
        b = -np.array(self.door.geom.in_pos).reshape((-1,1))
        b = np.r_[b, -np.array(self.door.geom.in_orn).reshape((-1,1))]
        b = np.r_[b, -b]
        val = np.zeros((6,1))
        val[0] = self.door.geom.width
        val[1] = 0.01
        val[2] = 0.03
        val[3:6] = 0.01
        val = np.r_[val, val]
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(InSlideDoor, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True
        self._init_include = False
        self._nonrollout = True

class Stacked(ExprPredicate):
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.obj, self.base_obj = params
        h1 = self.obj.geom.height if hasattr(self.obj.geom, 'height') else self.obj.geom.radius
        h2 = self.base_obj.geom.height if hasattr(self.base_obj.geom, 'height') else self.base_obj.geom.radius
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([0,1,2], dtype='int32')),
                                             ("rotation", np.array([0,1,2], dtype='int32'))]),
                                 (self.base_obj, [("pose", np.array([0,1,2], dtype='int32')),
                                                ("rotation", np.array([0,1,2], dtype='int32'))])])

        self.coeff = 2e-2
        self.rot_coeff = 5e-3
        if self.obj is self.base_obj:
            A = np.zeros((6,6))
        else:
            A = np.c_[np.eye(6), -np.eye(6)]
            A[:3] *= self.coeff
            A[3:] *= self.rot_coeff
        b, val = np.zeros((6, 1)), np.zeros((6, 1))
        b[2,0] = -self.coeff*(h1 + h2)
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)

        super(Stacked, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True
        self._init_include = False
        self._nonrollout = True


### EE CONSTRAINTS - Non-linear constraints on end effector pos/orn

class InContact(ExprPredicate):
    """
        Format: InContact Robot

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
            GRIPPER_CLOSE[Float]: Constants, specifying gripper value when gripper is closed
            GRIPPER_OPEN[Float]: Constants, specifying gripper value when gripper is open
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot = params
        if not hasattr(self, 'arm'): self.arm = params[0].geom.arms[0]
        self.gripper = params[0].geom.get_gripper(self.arm)
        dim = params[0].geom.gripper_dim(self.arm)
        attr_inds, attr_dim = init_robot_pred(self, params[0], [], attrs={params[0]:[self.gripper]})

        A = np.eye(dim)
        b = np.zeros(dim).reshape((dim,1))

        val = self.GRIPPER_CLOSE.reshape((dim,1))
        aff_expr = AffExpr(A, b)
        e = EqExpr(aff_expr, val)

        aff_expr = AffExpr(A, b)
        val = self.GRIPPER_OPEN.reshape((dim, 1))
        self.neg_expr = EqExpr(aff_expr, val)

        super(InContact, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True

class InContacts(ExprPredicate):
    """
        Format: InContact Robot
        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
            GRIPPER_CLOSE[Float]: Constants, specifying gripper value when gripper is closed
            GRIPPER_OPEN[Float]: Constants, specifying gripper value when gripper is open
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot = params
        if not hasattr(self, 'arm'): self.arm = params[0].geom.arms[0]
        attr_inds = self.attr_inds

        A = np.eye(2).reshape((2,2))
        b = np.zeros((2,1))

        avg = np.mean([self.GRIPPER_CLOSE, self.GRIPPER_OPEN])
        sign = -1 if self.GRIPPER_CLOSE > self.GRIPPER_OPEN else 1
        #val = np.array([[self.GRIPPER_CLOSE, self.GRIPPER_CLOSE]]).T
        val = np.array([[avg, avg]]).T - sign * 0.01
        aff_expr = AffExpr(sign*A, b)
        e = LEqExpr(aff_expr, sign*val)

        aff_expr = AffExpr(-sign*A, b)
        #val = np.array([[self.GRIPPER_OPEN, self.GRIPPER_OPEN]]).T
        val = np.array([[avg, avg]]).T + sign * 0.01
        self.neg_expr = LEqExpr(aff_expr, -sign*val)

        super(InContacts, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True

class CloseGripper(InContact):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.GRIPPER_CLOSE = np.array(params[0].geom.get_gripper_closed_val(scalar=False)) # const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = np.array(params[0].geom.get_gripper_open_val(scalar=False)) # const.GRIPPER_OPEN_VALUE
        super(CloseGripper, self).__init__(name, params, expected_param_types, env, debug)

class CloseGripperLeft(CloseGripper):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.arm = 'left'
        super(CloseGripperLeft, self).__init__(name, params, expected_param_types, env, debug)

class CloseGripperRight(CloseGripper):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.arm = 'right'
        super(CloseGripperRight, self).__init__(name, params, expected_param_types, env, debug)

class OpenGripper(CloseGripper):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(OpenGripper, self).__init__(name, params, expected_param_types, env, debug)
        self.expr = self.neg_expr

class OpenGripperLeft(OpenGripper):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.arm = 'left'
        super(OpenGripperLeft, self).__init__(name, params, expected_param_types, env, debug)

class OpenGripperRight(OpenGripper):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.arm = 'right'
        super(OpenGripperRight, self).__init__(name, params, expected_param_types, env, debug)

class InGripper(PosePredicate):
    """
        Format: InGripper, Robot, Item

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            set_robot_poses[Function]:Function that sets robot's poses
            get_robot_info[Function]:Function that returns robot's transformations and arm indices
            eval_f[Function]:Function returns predicate value
            eval_grad[Function]:Function returns predicate gradient
            coeff[Float]:In Gripper coeffitions, used during optimization
            opt_coeff[Float]:In Gripper coeffitions, used during optimization
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self._env = env
        if params[1].is_symbol():
            self.robot, self.targ = params
            attr_inds, attr_dim = init_robot_pred(self, self.robot, [self.targ])
            self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}
        else:
            self.robot, self.obj = params
            attr_inds, attr_dim = init_robot_pred(self, self.robot, [self.obj])
            self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                                   self.obj: self.lazy_spawn_or_body(self.obj, self.obj.name, self.obj.geom)}

        self.arm = self.robot.geom.arms[0]
        if not hasattr(self, 'coeff'): self.coeff = const.IN_GRIPPER_COEFF
        if not hasattr(self, 'rot_coeff'): self.rot_coeff = const.IN_GRIPPER_ROT_COEFF
        if hasattr(params[1], 'geom') and hasattr(params[1].geom, 'near_coeff'):
            self.coeff *= params[1].geom.near_coeff
            #self.rot_coeff *= params[1].geom.near_coeff

        if hasattr(params[1], 'geom') and 'sphere' in params[1].geom.get_types():
            self.rot_coeff = 0.

        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.rel_pt = np.array(params[1].geom.grasp_point) if hasattr(params[1], 'geom') and hasattr(params[1].geom, 'grasp_point') else np.zeros(3)
        self.eval_dim = 3 # 4
        e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        super(InGripper, self).__init__(name, e, self.attr_inds, params, expected_param_types, ind0=0, ind1=1, priority=2)
        self.spacial_anchor = True
        self._init_include = False

    def stacked_f(self, x):
        if self.eval_dim == 3:
            return self.coeff * self.pos_check_f(x, self.rel_pt)
        else:
            return np.vstack([self.coeff * self.pos_check_f(x, self.rel_pt), self.rot_coeff * self.ee_rot_check_f(x, robot_off=self.inv_mats[self.arm])])

    def stacked_grad(self, x):
        if self.eval_dim == 3:
            return self.coeff * self.pos_check_jac(x, self.rel_pt)
        else:
            return np.vstack([self.coeff * self.pos_check_jac(x, self.rel_pt), self.rot_coeff * self.ee_rot_check_jac(x, robot_off=self.inv_mats[self.arm])])

    def tile_f(self, x):
        val = self.eval_f(x)
        return np.r_[val, -1.*val]

    def tile_grad(self, x):
        grad = self.eval_grad(x)
        return np.r_[grad, -1.*grad]

    def resample(self, negated, t, plan):
        if negated: return None, None
        return robot_sampling.resample_in_gripper(self, negated, t, plan)

class InGripperLeft(InGripper):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(InGripperLeft, self).__init__(name, params, expected_param_types, env, debug)
        self.arm = "left" if "left" in self.robot.geom.arms else self.robot.geom.arms[0]

class InGripperRight(InGripper):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(InGripperRight, self).__init__(name, params, expected_param_types, env, debug)
        self.arm = "right" if "right" in self.robot.geom.arms else self.robot.geom.arms[0]

class NearGripper(InGripper):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.coeff = const.NEAR_GRIP_COEFF
        self.rot_coeff = const.NEAR_GRIP_ROT_COEFF
        super(NearGripper, self).__init__(name, params, expected_param_types, env, debug)
        self._rollout = True

class NearGripperLeft(InGripperLeft):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.coeff = const.NEAR_GRIP_COEFF
        self.rot_coeff = const.NEAR_GRIP_COEFF
        super(NearGripperLeft, self).__init__(name, params, expected_param_types, env, debug)
        self._rollout = True

class NearGripperRight(InGripperRight):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.coeff = const.NEAR_GRIP_COEFF
        self.rot_coeff = const.NEAR_GRIP_ROT_COEFF
        super(NearGripperRight, self).__init__(name, params, expected_param_types, env, debug)
        self._rollout = True

class AlmostInGripper(PosePredicate):
    """
        Format: AlmostInGripper, Robot, Item

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            set_robot_poses[Function]:Function that sets robot's poses
            get_robot_info[Function]:Function that returns robot's transformations and arm indices
            eval_f[Function]:Function returns predicate value
            eval_grad[Function]:Function returns predicate gradient
            coeff[Float]:In Gripper coeffitions, used during optimization
            opt_coeff[Float]:In Gripper coeffitions, used during optimization
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 2
        self._env = env
        self.robot, self.obj = params

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.obj: self.lazy_spawn_or_body(self.obj, self.obj.name, self.obj.geom)}

        e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        # e = LEqExpr(Expr(self.eval_f, self.eval_grad), self.max_dist.reshape(-1,1))
        super(AlmostInGripper, self).__init__(name, e, self.attr_inds, params, expected_param_types, ind0=0, ind1=1, priority = 2)
        self.spacial_anchor = True

class EEAt(PosePredicate):
    """
        Format: EEAt, Robot
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 1
        self._env = env
        self.robot, = params

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        super(EEAt, self).__init__(name, e, self.attr_inds, params, expected_param_types, ind0=0, ind1=1, priority = 2)
        self.spacial_anchor = True

class GripperAt(PosePredicate):
    """
        Format: GripperAt, Robot, EEPose
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 2
        self._env = env
        self.robot, self.pose = params

        attr_inds, attr_dim = init_robot_pred(self, self.robot, [], [self.pose])
        self.coeff = const.GRIPPER_AT_COEFF
        self.rot_coeff = const.GRIPPER_AT_ROT_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        super(GripperAt, self).__init__(name, e, self.attr_inds, params, expected_param_types, ind0=0, ind1=1, priority = 2)
        self.spacial_anchor = True

    def stacked_f(self, x):
        return np.vstack([self.coeff * self.pos_check_f(x)])

    def stacked_grad(self, x):
        return np.vstack([10*self.coeff * self.pos_check_jac(x)])

    def resample(self, negated, t, plan):
        return robot_sampling.resample_gripper_at(self, negated, t, plan)

class GripperAtLeft(GripperAt):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(GripperAtLeft, self).__init__(name, params, expected_param_types, env, debug)
        self.arm = "left"

class GripperAtRight(GripperAt):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(GripperAtRight, self).__init__(name, params, expected_param_types, env, debug)
        self.arm = "right"

class EEGraspValid(PosePredicate):

    # EEGraspValid EEPose Washer

    #@profile
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 2
        self._env = env
        self.ee_pose, self.robot = params

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        super(EEGraspValid, self).__init__(name, e, self.attr_inds, params, expected_param_types, ind0=0, ind1=1, priority = 0)
        self.spacial_anchor = True

class EEValid(PosePredicate):
    """
        Format: InGripper, Robot, Item

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            set_robot_poses[Function]:Function that sets robot's poses
            get_robot_info[Function]:Function that returns robot's transformations and arm indices
            eval_f[Function]:Function returns predicate value
            eval_grad[Function]:Function returns predicate gradient
            coeff[Float]:In Gripper coeffitions, used during optimization
            opt_coeff[Float]:In Gripper coeffitions, used during optimization
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, = params
        attr_inds, attr_dim = init_robot_pred(self, self.robot, [])
        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        self.arm = self.robot.geom.arms[0]
        if not hasattr(self, 'coeff'): self.coeff = 1e-2
        if not hasattr(self, 'rot_coeff'): self.rot_coeff = 1e-2
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.rel_pt = np.zeros(3)
        self.ee_ref = True
        if hasattr(self, 'dist'):
            self.eval_dim = 12
            e = LEqExpr(Expr(self.tile_f, self.tile_grad), self.coeff*self.dist*np.ones((self.eval_dim, 1)))
        else:
            self.eval_dim = 6 # 4
            e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        super(EEValid, self).__init__(name, e, self.attr_inds, params, expected_param_types, ind0=0, ind1=0, priority=2)
        self.spacial_anchor = True

    def stacked_f(self, x):
        #return self.coeff * self.pos_check_f(x, self.rel_pt)
        val = np.vstack([self.coeff * self.pos_check_f(x), self.rot_coeff * self.ee_rot_check_f(x)])
        return val

    def stacked_grad(self, x):
        #return self.coeff * self.pos_check_jac(x, self.rel_pt)
        return np.vstack([self.coeff * self.pos_check_jac(x), self.rot_coeff * self.ee_rot_check_jac(x)])

    def tile_f(self, x):
        val = self.eval_f(x)
        return np.r_[val, -1.*val]

    def tile_grad(self, x):
        grad = self.eval_grad(x)
        return np.r_[grad, -1.*grad]

class LeftEEValid(EEValid):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(LeftEEValid, self).__init__(name, params, expected_param_types, env, debug)
        self.arm = "left" if "left" in self.robot.geom.arms else self.robot.geom.arms[0]

class RightEEValid(EEValid):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(RightEEValid, self).__init__(name, params, expected_param_types, env, debug)
        self.arm = "right" if "right" in self.robot.geom.arms else self.robot.geom.arms[0]

class InContactRobotTable(InGripper):
    """
    Format: InContactRobotTable, Robot, Item

    Robot related

    Requires:
        attr_inds[OrderedDict]: robot attribute indices
        set_robot_poses[Function]:Function that sets robot's poses
        get_robot_info[Function]:Function that returns robot's transformations and arm indices
        eval_f[Function]:Function returns predicate value
        eval_grad[Function]:Function returns predicate gradient
        coeff[Float]:In Gripper coefficients, used during optimization
        opt_coeff[Float]:In Gripper coefficients, used during optimization
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(InContactRobotTable, self).__init__(name, params, expected_param_types, env=env, debug=debug)
        self.spacial_anchor = True
        self._init_include = False
        self.rel_pt = np.array([0, 0, params[1].geom.height])

    def stacked_f(self, x):
        # NOTE: We override this because we only care about contact in the z-direction, so 
        # we only return the last element!
        if self.eval_dim == 3:
            # NOTE: when the below expression == 0, this overall predicate
            # will evaluate to True (you can see this by going down the stack of
            # functions that are being called from pos_check_f).
            # import ipdb; ipdb.set_trace()
            return self.coeff * self.pos_check_f(x, self.rel_pt)[-1,None]
        else:
            raise ValueError('stacked_f case for self.eval_dim being > 3 has not been handled yet')
            # return np.vstack([self.coeff * self.pos_check_f(x, self.rel_pt), self.rot_coeff * self.ee_rot_check_f(x, robot_off=self.inv_mats[self.arm])])

    def stacked_grad(self, x):
        if self.eval_dim == 3:
            return self.coeff * self.pos_check_jac(x, self.rel_pt)[-1,None]
        else:
            raise ValueError('stacked_grad case for self.eval_dim being > 3 has not been handled yet')
            # return np.vstack([self.coeff * self.pos_check_jac(x, self.rel_pt), self.rot_coeff * self.ee_rot_check_jac(x, robot_off=self.inv_mats[self.arm])])

    def resample(self, negated, t, plan):
        return None, None  # There is no valid resample


### EEREACHABLE CONSTRAINTS - defined linear approach paths

class EEReachable(PosePredicate):
    """
        Format: EEReachable Robot, StartPose, EEPose

        Robot related

        Requires:
            attr_inds[OrderedDict]: attribute indices for constructing x
            set_robot_poses[Function]:Function that sets robot's poses
            get_robot_info[Function]:Function that returns robot's transformations and arm indices
            eval_f[Function]:Function returns predicate's value
            eval_grad[Function]:Function returns predicate's gradient
            coeff[Float]:pose coeffitions
            rot_coeff[Float]:rotation coeffitions
    """
    #@profile
    def __init__(self, name, params, expected_param_types, active_range=(-const.EEREACHABLE_STEPS, const.EEREACHABLE_STEPS), env=None, debug=False):
        self._env = env
        self.robot = params[0]
        if params[1].is_symbol():
            self.targ = params[1]
        else:
            self.obj = params[1]

        attrs = {self.robot: self.robot.geom.arms + self.robot.geom.grippers + ['pose', 'rotation']}
        attr_inds, attr_dim = init_robot_pred(self, self.robot, [params[1]], attrs=attrs)
        self.attr_dim = attr_dim

        if not hasattr(self, 'coeff'): self.coeff = const.EEREACHABLE_COEFF
        if not hasattr(self, 'rot_coeff'): self.rot_coeff = const.EEREACHABLE_ROT_COEFF
        self.approach_dist = const.APPROACH_DIST
        self.retreat_dist = const.RETREAT_DIST
        self.axis_coeff = 0. # For LEq, allow a little more slack in the gripper direction
        self.mask = np.ones((3,1))
        if not hasattr(self, 'rel_pt'):
            self.rel_pt = np.array(params[1].geom.grasp_point) if hasattr(params[1], 'geom') and hasattr(params[1].geom, 'grasp_point') else np.zeros(3)
        if not hasattr(self, 'axis'): self.axis = np.array([0, 0, -1])
        if not hasattr(self, 'f_tol'): self.f_tol = 0
        if not hasattr(self, 'pause'): self.pause = 0 # extra time ee is at target pos
        if not hasattr(self, 'arm'): self.arm = self.robot.geom.arms[0]
        if hasattr(params[1], 'geom'):
            self.coeff *= params[1].geom.eereachable_coeff

        self.eval_rel = True
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.n_steps = 1 + (active_range[1] - active_range[0])
        self.eval_dim = 3 * self.n_steps
        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        if self.f_tol > 0:
            self.eval_dim *= 2
            pos_expr = Expr(self.eval_f, self.eval_grad)
            self._coeffs = self.coeff * (self.f_tol*np.ones(3) + self.axis_coeff * self.f_tol*np.abs(self.axis))
            self._coeffs = np.tile(self._coeffs, 2*self.n_steps).reshape((-1,1))
            e = LEqExpr(pos_expr, self._coeffs)
        else:
            pos_expr = Expr(self.eval_f, self.eval_grad)
            e = EqExpr(pos_expr, np.zeros((self.eval_dim, 1)))

        super(EEReachable, self).__init__(name, e, self.attr_inds, params, expected_param_types, active_range = active_range, priority = 1)
        self.spacial_anchor = True
        self._rollout = True
        self._init_include = False

    #@profile
    def stacked_f(self, x):
        """
            Stacking values of all EEReachable timesteps in following order:
            pos_val(t - EEReachableStep)
            pos_val(t - EEReachableStep + 1)
            ...
            pos_val(t)
            rot_val(t)
            pos_val(t + 1)
            ...
            pos_val(t + EEReachableStep)
        """
        i, index = 0, 0
        f_res = []
        start, end = self.active_range
        for s in range(start, end+1):
            rel_pt = self.rel_pt + self.get_rel_pt(s)
            if self.eval_rel:
                obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x[i:i+self.attr_dim])
                f_res.append(self.coeff * self.rel_ee_pos_check_f(x[i:i+self.attr_dim], rel_pt))
            else:
                f_res.append(self.mask * self.coeff * self.abs_ee_pos_check_f(x[i:i+self.attr_dim], rel_pt))
            i += self.attr_dim

        f_res = np.vstack(f_res)
        if self.f_tol > 0:
            f_res = np.r_[f_res, -1.*f_res]
        return f_res

    #@profile
    def stacked_grad(self, x):
        """
            Stacking jacobian of all EEReachable timesteps in following order:
            pos_jac(t - EEReachableStep)
            pos_jac(t - EEReachableStep + 1)
            ...
            pos_jac(t)
            rot_jac(t)
            pos_jac(t + 1)
            ...
            pos_jac(t + EEReachableStep)
        """
        start, end = self.active_range
        dim, step = 3, end + 1 - start
        i, j = 0, 0
        grad = np.zeros((dim*step, self.attr_dim*step))
        for s in range(start, end+1):
            rel_pt = self.rel_pt + self.get_rel_pt(s)
            if self.eval_rel:
                grad[j:j+dim, i:i+self.attr_dim] = self.coeff * self.rel_ee_pos_check_jac(x[i:i+self.attr_dim], rel_pt)
            else:
                grad[j:j+dim, i:i+self.attr_dim] = self.coeff * self.abs_ee_pos_check_jac(x[i:i+self.attr_dim], rel_pt)
            j += dim
            i += self.attr_dim

        if self.f_tol > 0:
            grad = np.r_[grad, -1.*grad]

        return grad

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*self.approach_dist*self.axis
        else:
            return -rel_step*self.retreat_dist*self.axis

    def resample(self, negated, t, plan):
        if negated: return None, None
        #if hasattr(self, 'obj') and 'door' in self.obj.geom.get_types():
        #    return None, None
        return robot_sampling.resample_eereachable(self, negated, t, plan, inv=False, rel=self.eval_rel, use_rot=False)

class EEReachableRot(EEReachable):
    def __init__(self, name, params, expected_param_types, active_range=(-const.EEREACHABLE_STEPS, const.EEREACHABLE_STEPS), env=None, debug=False):
        super(EEReachableRot, self).__init__(name, params, expected_param_types, active_range, env, debug)

    def stacked_f(self, x):
        i, index = 0, 0
        f_res = []
        start, end = self.active_range
        for s in range(start, end+1):
            rel_pt = self.get_rel_pt(s)
            f_res.append(self.rot_coeff * self.ee_rot_check_f(x[i:i+self.attr_dim], robot_off=self.inv_mats[self.arm]))
            i += self.attr_dim

        f_res = np.vstack(f_res)
        if self.f_tol > 0:
            f_res = np.r_[f_res, -1.*f_res]
        return f_res

    def stacked_grad(self, x):
        start, end = self.active_range
        dim, step = 3, end + 1 - start
        i, j = 0, 0
        grad = np.zeros((dim*step, self.attr_dim*step))
        for s in range(start, end+1):
            rel_pt = self.get_rel_pt(s)
            grad[j:j+3, i:i+self.attr_dim] = self.rot_coeff *  self.ee_rot_check_jac(x[i:i+self.attr_dim], robot_off=self.inv_mats[self.arm])
            j += dim
            i += self.attr_dim

        if self.f_tol > 0:
            grad = np.r_[grad, -1.*grad]

        return grad

    def resample(self, negated, t, plan):
        if negated: return None, None
        if hasattr(self, 'obj') and 'door' in self.obj.geom.get_types():
            return None, None
        return robot_sampling.resample_eereachable(self, negated, t, plan, inv=False, use_pos=False)

class Approach(EEReachable):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(Approach, self).__init__(name, params, expected_param_types, env, debug, 0)
        self.approach_dist = const.GRASP_DIST

    def get_rel_pt(self, rel_step):
        return self.approach_dist*self.axis

class ApproachRot(EEReachableRot):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(ApproachRot, self).__init__(name, params, expected_param_types, env, debug, 0)
        self.approach_dist = const.GRASP_DIST

    def get_rel_pt(self, rel_step):
        return self.approach_dist*self.axis

class NearApproach(EEReachable):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = const.NEAR_APPROACH_COEFF
        super(NearApproach, self).__init__(name, params, expected_param_types, env, debug)
        self._rollout = True

class NearApproachRot(EEReachable):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.rot_coeff = const.NEAR_APPROACH_ROT_COEFF
        super(NearApproachRot, self).__init__(name, params, expected_param_types, env, debug)
        self._rollout = True

class EEReachableLeft(EEReachable):
    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.arm = "left"
        super(EEReachableLeft, self).__init__(name, params, expected_param_types, (-steps, steps), env, debug)

class EEApproachLeft(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(EEApproachLeft, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)

class EEApproachAbsLeft(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(EEApproachAbsLeft, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)
        self.eval_rel = False

class EEApproachStackLeft(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        self.stack_obj = params[2]
        super(EEApproachStackLeft, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)
        self.eval_rel = False
        h2 = params[2].geom.height if hasattr(params[2].geom, 'height') else params[2].geom.radius
        base_h = (params[1].geom.height + h2) 
        self.rel_pt += np.array([0., 0., base_h])

class EEApproachInDoorLeft(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(EEApproachInDoorLeft, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)
        self.eval_rel = True
        self.ref_orn = T.quat2mat(T.euler_to_quaternion(params[1].geom.in_orn, 'xyzw'))
        self.axis = self.ref_orn.dot(self.axis).round(4)
        self.rel_pt = params[1].geom.in_pos

class NearApproachInDoorLeft(EEApproachInDoorLeft):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.coeff = const.NEAR_GRIP_COEFF
        self.approach_dist = const.GRASP_DIST
        super(NearApproachInDoorLeft, self).__init__(name, params, expected_param_types, 0, env, debug)

    def get_rel_pt(self, rel_step):
        return -self.approach_dist*self.axis

class EERetreatLeft(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(EERetreatLeft, self).__init__(name, params, expected_param_types, (0, steps), env, debug)

class EERetreatAbsLeft(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(EERetreatAbsLeft, self).__init__(name, params, expected_param_types, (0, steps), env, debug)
        self.eval_rel = False

class EERetreatStackLeft(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        self.stack_obj = params[2]
        super(EERetreatStackLeft, self).__init__(name, params, expected_param_types, (0, steps), env, debug)
        h2 = params[2].geom.height if hasattr(params[2].geom, 'height') else params[2].geom.radius
        base_h = (params[1].geom.height + h2)
        self.rel_pt += np.array([0., 0., base_h])

class EERetreatInDoorLeft(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(EERetreatInDoorLeft, self).__init__(name, params, expected_param_types, (0, steps), env, debug)
        self.ref_orn = T.quat2mat(T.euler_to_quaternion(params[1].geom.in_orn, 'xyzw'))
        self.axis = self.ref_orn.dot(self.axis).round(4)
        self.rel_pt = params[1].geom.in_pos

class EEReachableLeftRot(EEReachableRot):
    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.arm = "left"
        super(EEReachableLeftRot, self).__init__(name, params, expected_param_types, (-steps, steps), env, debug)

class EEAtLeftRot(EEReachableRot):
    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.arm = "left"
        super(EEReachableLeftRot, self).__init__(name, params, expected_param_types, (0, 0), env, debug)

class ApproachLeft(EEReachableLeft):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(ApproachLeft, self).__init__(name, params, expected_param_types, env, debug, 0)
        self.approach_dist = const.GRASP_DIST

    def get_rel_pt(self, rel_step):
        return -self.approach_dist*self.axis

class EEAtXRelLeft(EEReachableLeft):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = const.EEATXY_COEFF
        super(EEAtXRelLeft, self).__init__(name, params, expected_param_types, env=env, debug=debug, steps=0)
        self.mask = np.array([1., 0., 0.]).reshape((3,1))
        self.approach_dist = const.GRASP_DIST
        self.eval_rel = True

    def get_rel_pt(self, rel_step):
        return np.zeros(3)

class EEAtYRelLeft(EEReachableLeft):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = const.EEATXY_COEFF
        super(EEAtYRelLeft, self).__init__(name, params, expected_param_types, env=env, debug=debug, steps=0)
        self.mask = np.array([0., 1., 0.]).reshape((3,1))
        self.approach_dist = const.GRASP_DIST
        self.eval_rel = True

    def get_rel_pt(self, rel_step):
        return np.zeros(3)

class EEAtZRelLeft(EEReachableLeft):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = const.EEATXY_COEFF
        super(EEAtZRelLeft, self).__init__(name, params, expected_param_types, env=env, debug=debug, steps=0)
        self.mask = np.array([0., 0., 1.]).reshape((3,1))
        self.approach_dist = const.GRASP_DIST
        self.eval_rel = True

    def get_rel_pt(self, rel_step):
        return np.zeros(3)

class EEAtXYLeft(EEReachableLeft):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = const.EEATXY_COEFF
        super(EEAtXYLeft, self).__init__(name, params, expected_param_types, env=env, debug=debug, steps=0)
        self.mask = np.array([1., 1., 0.]).reshape((3,1))
        self.approach_dist = const.GRASP_DIST
        self.eval_rel = False

    def get_rel_pt(self, rel_step):
        return np.zeros(3)

class EEAtRelXYLeft(EEAtXYLeft):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(EEAtRelXYLeft, self).__init__(name, params, expected_param_types, env, debug)
        self.eval_rel = True

class EEAtXZLeft(EEAtXYLeft):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.axis = np.array([0., -1., 0.])
        super(EEAtXZLeft, self).__init__(name, params, expected_param_types, env, debug)
        self.mask = np.array([1., 0., 1.]).reshape((3,1))

class EEAtRelXZLeft(EEAtXZLeft):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(EEAtRelXZLeft, self).__init__(name, params, expected_param_types, env, debug)
        self.eval_rel = True

class EEAtYZLeft(EEAtXYLeft):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.axis = np.array([-1., 0, 0.])
        super(EEAtXZLeft, self).__init__(name, params, expected_param_types, env, debug)
        self.mask = np.array([0., 1., 1.]).reshape((3,1))

class EEAtRelYZLeft(EEAtYZLeft):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(EEAtRelYZLeft, self).__init__(name, params, expected_param_types, env, debug)
        self.eval_rel = True

class NearApproachLeft(ApproachLeft):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # self.f_tol = 0.04
        self.coeff = const.NEAR_APPROACH_COEFF
        super(NearApproachLeft, self).__init__(name, params, expected_param_types, env, debug)
        self._rollout = True

class ApproachLeftRot(EEReachableLeftRot):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(ApproachLeftRot, self).__init__(name, params, expected_param_types, env, debug, 0)
        self.approach_dist = const.GRASP_DIST

    def get_rel_pt(self, rel_step):
        return -self.approach_dist*self.axis

class NearApproachLeftRot(ApproachLeftRot):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.rot_coeff = const.NEAR_APPROACH_ROT_COEFF
        super(NearApproachLeftRot, self).__init__(name, params, expected_param_types, env, debug)
        self._rollout = True

class EEReachableRight(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        super(EEReachableRight, self).__init__(name, params, expected_param_types, (-steps, steps), env, debug)

class EEApproachRight(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        super(EEApproachRight, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)

class EEApproachAbsRight(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        super(EEApproachAbsRight, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)
        self.eval_rel = False

class EEApproachStackRight(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        self.stack_obj = params[2]
        super(EEApproachStackRight, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)
        self.eval_rel = False
        h2 = params[2].geom.height if hasattr(params[2].geom, 'height') else params[2].geom.radius
        base_h = (params[1].geom.height + h2)
        self.rel_pt += np.array([0., 0., base_h])

class EEApproachInDoorRight(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        super(EEApproachInDoorRight, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)
        self.eval_rel = True
        self.ref_orn = T.quat2mat(T.euler_to_quaternion(params[1].geom.in_orn, 'xyzw'))
        self.axis = self.ref_orn.dot(self.axis).round(5)
        self.rel_pt = params[1].geom.in_pos

class NearApproachInDoorRight(EEApproachInDoorRight):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.coeff = 2e-2 # const.NEAR_APPROACH_COEFF
        super(NearApproachInDoorRight, self).__init__(name, params, expected_param_types, steps=0, env=env, debug=debug)
        self.approach_dist = const.PLACE_DIST # const.GRASP_DIST

    def get_rel_pt(self, rel_step):
        return -self.approach_dist*self.axis

class EERetreatRight(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        super(EERetreatRight, self).__init__(name, params, expected_param_types, (0, steps), env, debug)

class EEWeakRetreatRight(EERetreatRight):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        self.coeff = const.WEAK_EEREACHABLE_COEFF
        super(EEWeakRetreatRight, self).__init__(name, params, expected_param_types, steps, env, debug)
        self.approach_dist = const.QUICK_RETREAT_DIST
        self.retreat_dist = const.QUICK_RETREAT_DIST

class EERetreatAbsRight(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        super(EERetreatAbsRight, self).__init__(name, params, expected_param_types, (0, steps), env, debug)
        self.eval_rel = False

class EERetreatStackRight(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        self.stack_obj = params[2]
        super(EERetreatStackRight, self).__init__(name, params, expected_param_types, (0, steps), env, debug)
        h2 = params[2].geom.height if hasattr(params[2].geom, 'height') else params[2].geom.radius
        base_h = (params[1].geom.height + h2)
        self.rel_pt += np.array([0., 0., base_h])

class EERetreatInDoorRight(EEReachable):
    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        super(EERetreatInDoorRight, self).__init__(name, params, expected_param_types, (0, steps), env, debug)
        self.ref_orn = T.quat2mat(T.euler_to_quaternion(params[1].geom.in_orn, 'xyzw'))
        self.axis = self.ref_orn.dot(self.axis).round(4)
        self.rel_pt = params[1].geom.in_pos

class EEReachableRightRot(EEReachableRot):
    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.arm = "right"
        super(EEReachableRightRot, self).__init__(name, params, expected_param_types, (-steps, steps), env, debug)

class EEAtRightRot(EEReachableRot):
    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.arm = "right"
        super(EEAtRightRot, self).__init__(name, params, expected_param_types, (0, 0), env, debug)

class ApproachRight(EEReachableRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(ApproachRight, self).__init__(name, params, expected_param_types, 0, env, debug)
        self.approach_dist = const.GRASP_DIST

    def get_rel_pt(self, rel_step):
        return -self.approach_dist*self.axis

class NearApproachRight(ApproachRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        #self.f_tol = 0.04
        self.coeff = const.NEAR_APPROACH_COEFF
        super(NearApproachRight, self).__init__(name, params, expected_param_types, env, debug)
        self._rollout = True

class NearRetreatRight(ApproachRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        #self.f_tol = 0.04
        self.coeff = const.NEAR_RETREAT_COEFF
        super(NearRetreatRight, self).__init__(name, params, expected_param_types, env, debug)
        self.approach_dist = const.PLACE_DIST

class ApproachRightRot(EEReachableRightRot):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(ApproachRightRot, self).__init__(name, params, expected_param_types, env, debug, 0)
        self.approach_dist = const.GRASP_DIST

    def get_rel_pt(self, rel_step):
        return -self.approach_dist*self.axis

class NearApproachRightRot(ApproachRightRot):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # self.f_tol = 0.04
        self.rot_coeff = const.NEAR_APPROACH_ROT_COEFF
        super(NearApproachRightRot, self).__init__(name, params, expected_param_types, env, debug)
        self._rollout = True

class EEAtXRelRight(EEReachableRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = const.EEATXY_COEFF
        super(EEAtXRelRight, self).__init__(name, params, expected_param_types, env=env, debug=debug, steps=0)
        self.mask = np.array([1., 0., 0.]).reshape((3,1))
        self.approach_dist = const.GRASP_DIST
        self.eval_rel = True

    def get_rel_pt(self, rel_step):
        return np.zeros(3)

class EEAtYRelRight(EEReachableRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = const.EEATXY_COEFF
        super(EEAtYRelRight, self).__init__(name, params, expected_param_types, env=env, debug=debug, steps=0)
        self.mask = np.array([0., 1., 0.]).reshape((3,1))
        self.approach_dist = const.GRASP_DIST
        self.eval_rel = True

    def get_rel_pt(self, rel_step):
        return np.zeros(3)

class EEAtZRelRight(EEReachableRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = const.EEATXY_COEFF
        super(EEAtZRelRight, self).__init__(name, params, expected_param_types, env=env, debug=debug, steps=0)
        self.mask = np.array([0., 0., 1.]).reshape((3,1))
        self.approach_dist = const.GRASP_DIST
        self.eval_rel = True

    def get_rel_pt(self, rel_step):
        return np.zeros(3)

class EEAtXYRight(EEReachableRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = const.EEATXY_COEFF
        super(EEAtXYRight, self).__init__(name, params, expected_param_types, env=env, debug=debug, steps=0)
        self.mask = np.array([1., 1., 0.]).reshape((3,1))
        self.approach_dist = const.GRASP_DIST
        self.eval_rel = False

    def get_rel_pt(self, rel_step):
        return np.zeros(3)

class EEAtRelXYRight(EEAtXYRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(EEAtRelXYRight, self).__init__(name, params, expected_param_types, env, debug)
        self.eval_rel = True

class EEAtXZRight(EEAtXYRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.axis = np.array([0., -1., 0.])
        super(EEAtXZRight, self).__init__(name, params, expected_param_types, env, debug)
        self.mask = np.array([1., 0., 1.]).reshape((3,1))
        self.eval_rel = False

class EEAtRelXZRight(EEAtXZRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(EEAtRelXZRight, self).__init__(name, params, expected_param_types, env, debug)
        self.eval_rel = True

class EEAtYZRight(EEAtXYRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.axis = np.array([-1., 0., 0.])
        super(EEAtXZRight, self).__init__(name, params, expected_param_types, env, debug)
        self.mask = np.array([0., 1., 1.]).reshape((3,1))
        self.eval_rel = False

class EEAtRelYZRight(EEAtYZRight):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(EEAtRelYZRight, self).__init__(name, params, expected_param_types, env, debug)
        self.eval_rel = True

class EEReachableLeftInv(EEReachableLeft):
    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([0, 0, -const.RETREAT_DIST])
        else:
            return rel_step*np.array([-const.APPROACH_DIST, 0, 0])

    def resample(self, negated, t, plan):
        return robot_sampling.resample_eereachable(self, negated, t, plan, inv=True)

class EEReachableRightInv(EEReachableRight):
    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([0, 0, -const.RETREAT_DIST])
        else:
            return rel_step*np.array([-const.APPROACH_DIST, 0, 0])

    def resample(self, negated, t, plan):
        return robot_sampling.resample_eereachable(self, negated, t, plan, inv=True)

class EEReachableLeftVer(EEReachableLeft):
    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

    def resample(self, negated, t, plan):
        return robot_sampling.resample_eereachable_ver(self, negated, t, plan)

class EEReachableRightVer(EEReachableRight):
    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

    def resample(self, negated, t, plan):
        return robot_sampling.resample_eereachable_ver(self, negated, t, plan)


### COLLISION CONSTRAINTS

class Obstructs(CollisionPredicate):
    """
        Format: Obstructs, Robot, RobotPose, RobotPose, Can

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]:number of attribute in robot's full pose
            set_robot_poses[Function]:Function that sets robot's poses
            set_active_dof_inds[Function]:Function that sets robot's active dof indices
            coeff[Float]:EEReachable coeffitions, used during optimization
            neg_coeff[Float]:EEReachable coeffitions, used during optimization
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False, tol=const.COLLISION_TOL):
        self._env = env
        self.robot, self.obstacle = params

        self.coeff = -const.OBSTRUCTS_COEFF
        self.neg_coeff = const.OBSTRUCTS_COEFF

        attrs = {self.robot: self.robot.geom.arms + self.robot.geom.grippers + ['pose', 'rotation']}
        attr_inds, attr_dim = init_robot_pred(self, self.robot, [self.obstacle], attrs=attrs)
        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.obstacle: self.lazy_spawn_or_body(self.obstacle, self.obstacle.name, self.obstacle.geom)}

        col_expr = Expr(self.f, self.grad)
        links = len(self.robot.geom.col_links)

        self.col_link_pairs = [x for x in itertools.product(self.robot.geom.col_links, self.obstacle.geom.col_links)]
        self.col_link_pairs = sorted(self.col_link_pairs)

        val = np.zeros((len(self.col_link_pairs),1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(self.f_neg, self.grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(Obstructs, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1, debug=debug, tol=tol, priority=3)
        self.spacial_anchor = False
        self._init_include = False

    def f(self, x):
        return self.coeff*self.robot_obj_collision(x)[0]

    def grad(self, x):
        return self.coeff*self.robot_obj_collision(x)[1]

    def f_neg(self, x):
        return self.neg_coeff*self.robot_obj_collision(x)[0]

    def grad_neg(self, x):
        return self.neg_coeff*self.robot_obj_collision(x)[1]

    #@profile
    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

    def resample(self, negated, t, plan):
        return robot_sampling.resample_obstructs(self, negated, t, plan)

class ObstructsHolding(CollisionPredicate):
    """
        Format: ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can

        Robot related

        Requires:
            attr_dim[Int]:number of attribute in robot's full pose
            attr_inds[OrderedDict]: robot attribute indices
            set_robot_poses[Function]:Function that sets robot's poses
            set_active_dof_inds[Function]:Function that sets robot's active dof indices
            OBSTRUCTS_OPT_COEFF[Float]: Obstructs_holding coeffitions, used during optimization problem
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False, tol=const.COLLISION_TOL):
        self._env = env
        self.robot, self.obstacle, self.obj = params
        attr_inds, attr_dim = init_robot_pred(self, self.robot, [self.obstacle, self.obj])
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attr_inds = self.attr_inds

        self.coeff = -const.OBSTRUCTS_COEFF
        self.neg_coeff = const.OBSTRUCTS_COEFF

        attrs = {self.robot: self.robot.geom.arms + self.robot.geom.grippers + ['pose', 'rotation']}
        if self.obstacle is not self.obj:
            attr_inds, attr_dim = init_robot_pred(self, self.robot, [self.obstacle, self.obj], attrs=attrs)
        else:
            attr_inds, attr_dim = init_robot_pred(self, self.robot, [self.obj], attrs=attrs)
        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.obstacle: self.lazy_spawn_or_body(self.obstacle, self.obstacle.name, self.obstacle.geom),
                               self.obj: self.lazy_spawn_or_body(self.obj, self.obj.name, self.obj.geom)}

        # self.col_link_pairs = [x for x in itertools.product(self.robot.geom.col_links, self.obstacle.geom.col_links)]
        def exclude_f(c):
            c = list(c)
            return (('left_gripper_r_finger_tip' in c or 'left_gripper_r_finger' in c) and 'short_1' in c) \
                   or (('right_gripper_l_finger_tip' in c or 'right_gripper_l_finger' in c) and 'short_2' in c)
        self.col_link_pairs = [x for x in itertools.product(self.robot.geom.col_links, self.obstacle.geom.col_links) if not exclude_f(x)]
        self.col_link_pairs = sorted(self.col_link_pairs)

        self.obj_obj_link_pairs = [x for x in itertools.product(self.obj.geom.col_links, self.obstacle.geom.col_links)]
        self.obj_obj_link_pairs = sorted(self.obj_obj_link_pairs)


        if self.obj.name == self.obstacle.name:
            links = len(self.col_link_pairs)

            self.offset = const.DIST_SAFE - const.COLLISION_TOL
            val = np.zeros((links,1))
        else:
            links = len(self.col_link_pairs) + len(self.obj_obj_link_pairs)
            self.offset = 0
            val = np.zeros((links,1))

        col_expr, col_expr_neg = Expr(self.f, self.grad), Expr(self.f_neg, self.grad_neg)
        e, self.neg_expr = LEqExpr(col_expr, val), LEqExpr(col_expr_neg, val)
        super(ObstructsHolding, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1, debug = debug, tol=tol, priority=3)
        self.spacial_anchor = False
        self._init_include = False

    def f(self, x):
        return self.coeff * (self.col_fn(x)[0] - self.offset)

    def grad(self, x):
        return self.coeff * self.col_fn(x)[1]

    def f_neg(self, x):
        return self.neg_coeff * (self.col_fn(x)[0] - self.offset)

    def grad_neg(self, x):
        return self.neg_coeff * self.col_fn(x)[1]

    def col_fn(self, x):
        if self.obj.name == self.obstacle.name:
            return (np.zeros((1,1)), np.zeros((1, self.attr_dim)))
        else:
            return self.robot_obj_held_collision(x)

    #@profile
    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None
    
    def resample(self, negated, t, plan):
        return robot_sampling.resample_obstructs(self, negated, t, plan)

class Collides(CollisionPredicate):
    """
        Format: Collides Item Item

        Non-robot related
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.coeff = -const.COLLIDE_COEFF
        self.neg_coeff = const.COLLIDE_COEFF
        if len(params) == 2:
            self.obj, self.obstacle = params
        else:
            self.obj = params[0]
            self.obstacle = self.obj

        attr_inds = OrderedDict([(self.obj, [("pose", np.array([0, 1, 2], dtype='int32')),
                                             ("rotation", np.array([0, 1, 2], dtype='int32'))]),
                                 (self.obstacle, [("pose", np.array([0, 1, 2], dtype='int32')),
                                                  ("rotation", np.array([0, 1, 2], dtype='int32'))])])
        self._param_to_body = {self.obj: self.lazy_spawn_or_body(self.obj, self.obj.name, self.obj.geom),
                               self.obstacle: self.lazy_spawn_or_body(self.obstacle, self.obstacle.name, self.obstacle.geom)}

        self.obj_obj_link_pairs = [x for x in itertools.product(self.obj.geom.col_links, self.obstacle.geom.col_links)]
        self.obj_obj_link_pairs = sorted(self.obj_obj_link_pairs)

        links = len(self.obj_obj_link_pairs)

        col_expr, val = Expr(self.f, self.grad), np.zeros((1,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(self.f_neg, self.grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(Collides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1, debug=debug, priority = 3)
        self.spacial_anchor = False
        self.dsafe = const.COLLIDES_DSAFE

    def f(self, x):
        if self.obj is self.obstacle:
            return np.zeros((1,1))
        else:
            return self.coeff * self.obj_obj_collision(x)[0]

    def grad(self, x):
        if self.obj is self.obstacle:
            return np.zeros((1, self.attr_dim))
        else:
            return self.coeff * self.obj_obj_collision(x)[1]

    def f_neg(self, x):
        if self.obj is self.obstacle:
            return np.zeros((1,1))
        else:
            return self.neg_coeff * self.obj_obj_collision(x)[0]

    def grad_neg(self, x):
        if self.obj is self.obstacle:
            return np.zeros((1, self.attr_dim))
        else:
            return self.neg_coeff * self.obj_obj_collision(x)[1]

    #@profile
    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

class RCollides(CollisionPredicate):
    """
        Format: RCollides Robot Obstacle

        Robot related

        Requires:
            attr_dim[Int]:number of attribute in robot's full pose
            attr_inds[OrderedDict]: robot attribute indices
            set_robot_poses[Function]:Function that sets robot's poses
            set_active_dof_inds[Function]:Function that sets robot's active dof indices
            RCOLLIDES_OPT_COEFF[Float]: Obstructs_holding coeffitions, used during optimization problem
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, self.obstacle = params
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)

        self.coeff = -const.RCOLLIDE_COEFF
        self.neg_coeff = const.RCOLLIDE_COEFF
        attrs = {self.robot: self.robot.geom.arms + self.robot.geom.grippers + ['pose', 'rotation'], \
                 self.obstacle: ['pose', 'rotation']}

        if 'door' in self.obstacle.geom.get_types():
            attrs[self.obstacle].append('hinge')

        attr_inds, attr_dim = init_robot_pred(self, self.robot, [self.obstacle], attrs=attrs)
        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.obstacle: self.lazy_spawn_or_body(self.obstacle, self.obstacle.name, self.obstacle.geom)}

        col_expr = Expr(self.f, self.grad)

        self.col_link_pairs = [x for x in itertools.product(self.robot.geom.col_links, self.obstacle.geom.col_links)]
        self.col_link_pairs = sorted(self.col_link_pairs)
        links = len(self.col_link_pairs)

        val = np.zeros((links,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(self.f_neg, self.grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(RCollides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1, priority = 3)
        self.spacial_anchor = False
        self.dsafe = const.RCOLLIDES_DSAFE
        self._init_include = False

    def f(self, x):
        return self.coeff * self.robot_obj_collision(x)[0]

    def grad(self, x):
        return self.coeff * self.robot_obj_collision(x)[1]

    def f_neg(self, x):
        return self.neg_coeff * self.robot_obj_collision(x)[0]

    def grad_neg(self, x):
        return self.neg_coeff * self.robot_obj_collision(x)[1]

    #@profile
    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

    def resample(self, negated, t, plan):
        return robot_sampling.resample_obstructs(self, negated, t, plan)

class RSelfCollides(CollisionPredicate):
    """
        Format: RCollides Robot

        Robot related

        Requires:
            attr_dim[Int]:number of attribute in robot's full pose
            attr_inds[OrderedDict]: robot attribute indices
            set_robot_poses[Function]:Function that sets robot's poses
            set_active_dof_inds[Function]:Function that sets robot's active dof indices
            RCOLLIDES_OPT_COEFF[Float]: Obstructs_holding coeffitions, used during optimization problem
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot = params[0]
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attrs = {self.robot: self.robot.geom.arms + self.robot.geom.grippers + ['pose', 'rotation']}
        attr_inds, attr_dim = init_robot_pred(self, self.robot, [], attrs=attrs)
        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        self.coeff = -const.RCOLLIDE_COEFF
        self.neg_coeff = const.RCOLLIDE_COEFF

        col_expr = Expr(self.f, self.grad)

        self.col_link_pairs = [x for x in itertools.product(self.robot.geom.col_links, self.robot.geom.col_links)]
        self.col_link_pairs = sorted(self.col_link_pairs)
        links = len(self.col_link_pairs)

        val = np.zeros((links,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(self.f_neg, self.grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(RSelfCollides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=0, priority = 3)
        self.spacial_anchor = False
        self.dsafe = const.RCOLLIDES_DSAFE

    def f(self, x):
        return self.coeff * self.robot_self_collision(x)[0]

    def grad(self, x):
        return self.coeff * self.robot_self_collision(x)[1]

    def f_neg(self, x):
        return self.neg_coeff * self.robot_self_collision(x)[0]

    def grad_neg(self, x):
        return self.neg_coeff * self.robot_self_collision(x)[1]

    #@profile
    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

class BasketLevel(ExprPredicate):
    '''
    Format: BasketLevel Basket
    '''
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        attr_inds = self.attr_inds
        A = np.c_[np.eye(self.attr_dim)]
        A[0, 0] = 0
        b, val = np.zeros((self.attr_dim,1)), np.array([[0], [0], [np.pi/2]])
        # b, val = np.zeros((self.attr_dim,1)), np.array([[np.pi/2], [0], [np.pi/2]])
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(BasketLevel, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)

        self.spacial_anchor = False

class ObjectWithinRotLimit(ExprPredicate):
    '''
    Format: ObjectWithinRotLimit Object
    '''
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        attr_inds = self.attr_inds
        A = np.r_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim*2,1)), np.array([[np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]]).T
        pos_expr = AffExpr(A, b)
        e = LEqExpr(pos_expr, val)
        super(ObjectWithinRotLimit, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = False

#class EEValid(PosePredicate):
#    def __init__(self, name, params, expected_param_types, env=None, debug=False):
#        self.robot = params[0]
#        self.coeff = 0.1
#        self.eval_dim = 3
#        if not hasattr(self, 'arm'): self.arm = self.robot.geom.arms[0]
#        attr_inds, attr_dim = init_robot_pred(self, params[0], [])
#        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}
#        pos_expr, val = Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim,1))
#        e = EqExpr(pos_expr, val)
#        super(EEValid, self).__init__(name, e, attr_inds, params, expected_param_types, priority=1)
#
#    def eval_f(self, x):
#        x = x.flatten()
#        body = self.robot.openrave_body
#        self.set_robot_poses(x, body)
#        ee_pos = np.array(body.fwd_kinematics(self.arm)['pos'])
#        inds = self.attr_map[self.robot, '{}_ee_pos'.format(self.arm)]
#        return self.coeff*(np.array(ee_pos) - x[inds]).reshape((-1,1))
#
#    def eval_grad(self, x):
#        jac = np.zeros((3, self.attr_dim))
#        inds = self.attr_map[self.robot, '{}_ee_pos'.format(self.arm)]
#        jac[:, inds] = -np.eye(3)
#        return self.coeff*jac
#
#class LeftEEValid(EEValid):
#    arm = 'left'
#
#class RightEEValid(EEValid):
#    arm = 'right'

class GrippersLevel(PosePredicate):
    '''
    Format: GrippersLevel Robot
    '''
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self._env = env
        self.robot = params[0]
        attr_inds = self.attr_inds

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        pos_expr, val = Expr(self.f, self.grad), np.zeros((self.eval_dim,1))
        e = EqExpr(pos_expr, val)

        super(GrippersLevel, self).__init__(name, e, attr_inds, params, expected_param_types, priority=0)
        self.spacial_anchor = False

class EERetiming(PosePredicate):
    """
        Format: EERetiming Robot EEVel

        Robot related
        Requires:
        self.attr_inds
        self.coeff
        self.eval_f
        self.eval_grad
        self.eval_dim
    """
    #@profile
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 2
        self._env = env
        self.robot, self.ee_vel = params
        attr_inds = self.attr_inds

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        f = lambda x: self.coeff*self.eval_f(x)
        grad = lambda x: self.coeff*self.eval_grad(x)

        pos_expr, val = Expr(f, grad), np.zeros((self.eval_dim,1))
        e = EqExpr(pos_expr, val)
        super(EERetiming, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1, active_range = (0,1), priority = 3)
        self.spacial_anchor = False

class ObjRelPoseConstant(ExprPredicate):
    '''
    Format: ObjRelPoseConstant Basket Cloth
    '''
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        attr_inds = self.attr_inds
        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim), -np.eye(self.attr_dim), np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim,1)), np.zeros((self.attr_dim,1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(ObjRelPoseConstant, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority = -2)

        self.spacial_anchor = False

class IsPushing(PosePredicate):
    '''
    Format: IsPushing Robot Robot
    '''
    #@profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self._env = env
        self.robot1 = params[0]
        self.robot2 = params[1]
        attr_inds = self.attr_inds

        self._param_to_body = {self.robot1: self.lazy_spawn_or_body(self.robot1, self.robot1.name, self.robot1.geom), self.robot2: self.lazy_spawn_or_body(self.robot2, self.robot2.name, self.robot2.geom)}

        f = lambda x: self.coeff*self.eval_f(x)
        grad = lambda x: self.coeff*self.eval_grad(x)

        pos_expr, val = Expr(f, grad), np.zeros((self.eval_dim,1))
        e = EqExpr(pos_expr, val)

        super(IsPushing, self).__init__(name, e, attr_inds, params, expected_param_types, priority = 1)
        self.spacial_anchor = False

class GrippersDownRot(GrippersLevel):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = 1e-2
        self.opt_coeff = 1e-2
        attr_inds, attr_dim = init_robot_pred(self, params[0], [])
        self.local_dir = np.array([0,0,1])

        geom = params[0].geom
        self.quats = {}
        self.mats = {}
        geom = params[0].geom
        if not hasattr(self, 'arms'): self.arms = geom.arms
        if not hasattr(self, 'axis'): self.axis = [0, 0, -1]

        for arm in self.arms:
            axis = geom.get_gripper_axis(arm)
            quat = OpenRAVEBody.quat_from_v1_to_v2(axis, self.axis)
            self.quats[arm] = quat
            self.mats[arm] = T.quat2mat(quat)

        self.eval_dim = 3 * len(self.arms)
        super(GrippersDownRot, self).__init__(name, params, expected_param_types, env, debug)

    def f(self, x):
        return self.coeff*self.both_arm_rot_check_f(x)

    def grad(self, x):
        return self.coeff*self.both_arm_rot_check_jac(x)
    
    def resample(self, negated, t, plan):
        return robot_sampling.resample_gripper_down_rot(self, negated, t, plan)

    def both_arm_rot_check_f(self, x):
        robot_body = self._param_to_body[self.params[self.ind0]]
        self.set_robot_poses(x, robot_body)
        obj_trans = np.zeros((4,4))
        obj_trans[3,3] = 1
        trans = []
        for arm in self.arms:
            obj_trans[:3,:3] = self.mats[arm]
            robot_trans, arm_inds = self.get_robot_info(robot_body, arm)
            rot_val = self.rot_error_f(obj_trans, robot_trans, self.local_dir, robot_off=self.inv_mats[self.arm])
            trans.append(rot_val)
        return np.concatenate(trans, axis=0)

    def both_arm_rot_check_jac(self, x):
        robot_body = self._param_to_body[self.params[self.ind0]]
        self.set_robot_poses(x, robot_body)
        geom = robot_body._geom
        obj_trans = np.zeros((4,4))
        obj_trans[3,3] = 1
        trans = []
        jacs = []
        axises = np.eye(3)
        for arm in self.arms:
            obj_trans[:3,:3] = self.mats[arm]
            robot_trans, arm_inds = self.get_robot_info(robot_body, arm)
            rot_jacs = []
            lb, ub = geom.get_arm_bnds(arm)
            axes = [p.getJointInfo(robot_body.body_id, jnt_id)[13] for jnt_id in arm_inds]
            jacs.append(self.rot_error_jac(obj_trans, robot_trans, axises, arm_inds, self.local_dir, robot_off=self.inv_mats[self.arm]))

        rot_jac = np.concatenate(jacs, axis=-1)
        return rot_jac

class RightGripperDownRot(GrippersDownRot):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.arms = ['right']
        super(RightGripperDownRot, self).__init__(name, params, expected_param_types, env, debug)

class LeftGripperDownRot(GrippersDownRot):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.arms = ['left']
        super(LeftGripperDownRot, self).__init__(name, params, expected_param_types, env, debug)

class HeightBlock(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.goal_obj, self.block_obj, = params
        goal_geom, block_geom = self.goal_obj.geom, self.block_obj.geom
        self.goal_h = goal_geom.height if hasattr(goal_geom, 'height') else goal_geom.radius
        self.block_h = block_geom.height if hasattr(block_geom, 'height') else block_geom.radius
        self.dist = 0.1

        attr_inds = OrderedDict([(self.goal_obj, [("pose", np.array([0,1,2], dtype='int32'))]),
                                 (self.block_obj, [("pose", np.array([0,1,2], dtype='int32'))])])

        A = np.c_[np.r_[np.eye(3), -np.eye(3)], np.r_[-np.eye(3), np.eye(3)]]
        A *= 0
        b, val = np.zeros((6, 1)), NEAR_TOL*np.ones((6, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(HeightBlock, self).__init__(name, e, attr_inds, params, expected_param_types, tol=1e-3, priority=-2, active_range=(0,0))

    def test(self, time, negated=False, tol=1e-3):
        # Move taller objects first
        if self.block_h <= self.goal_h:
            return negated

        # Ignore objects already placed
        block_pos = self.block_obj.pose[:, time]
        goal_pos = self.goal_obj.pose[:, time] if not self.goal_obj.is_symbol() else self.goal_obj.value[:,0]
        # For now, hardcode in the fact positive means placed
        if block_pos[1] > 0.0 and not self.goal_obj.is_symbol():
            return negated

        if block_pos[1] < 0.0 and self.goal_obj.is_symbol():
            return negated

        #if np.sum((block_pos - self.block_targ.value[:,0])**2) < self.dist**2:
        #    return negated
      
        return not negated

class DeskHeightBlock(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.goal_obj, self.block_obj, = params
        goal_geom, block_geom = self.goal_obj.geom, self.block_obj.geom
        self.goal_h = goal_geom.height if hasattr(goal_geom, 'height') else goal_geom.radius
        self.block_h = block_geom.height if hasattr(block_geom, 'height') else block_geom.radius
        self.dist = 0.1

        attr_inds = OrderedDict([(self.goal_obj, [("pose", np.array([0,1,2], dtype='int32'))]),
                                 (self.block_obj, [("pose", np.array([0,1,2], dtype='int32'))])])

        A = np.c_[np.r_[np.eye(3), -np.eye(3)], np.r_[-np.eye(3), np.eye(3)]]
        A *= 0
        b, val = np.zeros((6, 1)), NEAR_TOL*np.ones((6, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(DeskHeightBlock, self).__init__(name, e, attr_inds, params, expected_param_types, tol=1e-3, priority=-2, active_range=(0,0))
        self._init_include = False

    def test(self, time, negated=False, tol=1e-3):
        # Move taller objects first
        if self.block_h <= self.goal_h:
            return negated

        block_pos = self.block_obj.pose[:, time]
        goal_pos = self.goal_obj.pose[:, time]
        if self.goal_obj.name.find('shelf_handle') >= 0 and \
           self.block_obj.name.find('upright') >= 0:
            if block_pos[2] < 0.75 or block_pos[0] > 0.55 or \
               block_pos[1] < 0.75 or block_pos[1] > 0.86:
                return negated

            if goal_pos[0] > 0.1:
                return negated
            return not negated
                      
        if 'box' not in self.block_obj.geom.get_types():
            return negated

        if 'box' not in self.goal_obj.geom.get_types():
            return negated

        if block_pos[1] > 0.86:
            return negated

        if block_pos[2] < 0.75:
            return negated

        if block_pos[0] > 0.55:
            return negated
      
        if block_pos[1] < 0.6:
            return negated

        return not negated

class AboveTable(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self.obj, = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([2], dtype='int32'))])])
        A = -np.ones((1,1))
        z = 1.0 if self.obj.name in ['milk', 'cereal'] else 0.95
        b = z * np.ones((1,1))
        val = np.zeros((1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(AboveTable, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True
        self._init_include = False

class LiftedAboveTable(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.obj, self.table = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([2], dtype='int32'))]),
                                 (self.table, [("pose", np.array([2], dtype='int32'))])])
            
        A = np.array([[-1.,1.]])

        b = -0.1 * np.ones((1,1))
        val = np.zeros((1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(LiftedAboveTable, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True

class Lifted(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.obj, self.robot = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([2], dtype='int32'))])])
            
        A = np.array([[-1.]])

        if self.obj.name.lower().find('upright') >= 0:
            #b = 0.925 * np.ones((1,1))
            b = 0.9 * np.ones((1,1))
        elif self.obj.name.lower().find('flat') >= 0:
            b = 0.8 * np.ones((1,1))
        else:
            #b = 0.875 * np.ones((1,1))
            b = 0.85 * np.ones((1,1))
        val = np.zeros((1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(Lifted, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True
        self._init_include = False

class InReach(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.obj, self.robot = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([2], dtype='int32'))])])
            
        A = np.array([[-1.]])
        if self.obj.name.find('upright') >= 0:
            b = 0.8 * np.ones((1,1))
        elif self.robot.name.find('panda') >= 0:
            b = 0.6 * np.ones((1,1))
        else:
            b = np.zeros((1,1))
        val = np.zeros((1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(InReach, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True
        self._init_include = False

class Stackable(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.obj, self.item = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([2], dtype='int32'))])])
            
        A = np.array([[1.]])
        if self.obj.name.find('flat') >= 0 and self.item.name.find('upright') >= 0:
            b = -1. * np.ones((1,1))
        else:
            b = np.zeros((1,1))
        val = np.zeros((1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(Stackable, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True
        self._init_include = True

class OffDesk(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self.obj, self.robot = params
        attr_inds = OrderedDict([(self.obj, [("pose", np.array([0], dtype='int32'))])])
            
        A = np.array([[-1.]])
        b = 0.62 * np.ones((1,1))
        val = np.zeros((1,1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(OffDesk, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
        self.spacial_anchor = True

