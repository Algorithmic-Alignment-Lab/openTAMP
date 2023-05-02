import numpy as np
import pybullet as p

from sco_py.expr import BoundExpr, QuadExpr, AffExpr
from opentamp.core.internal_repr.parameter import Object
from opentamp.core.util_classes.matrix import Vector
import opentamp.core.util_classes.common_constants as const
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
import opentamp.core.util_classes.robot_sampling as robot_sampling
import opentamp.core.util_classes.transform_utils as T
from opentamp.pma import backtrack_ll_solver_OSQP

try:
    USE_GRB = True
    from opentamp.pma import backtrack_ll_solver_gurobi
except:
    USE_GRB = False

PANDA_REF_JNTS = [-0.30, -0.4, 0.28, -2.5, 0.13, 1.87, 0.91]
#PANDA_REF_JNTS = [0.3, -0.8, 1.0, -2.5, 1.5, 1.87, 0.91]
#PANDA_REF_JNTS = [0.3, -0.8, 1.0, -2.5, 0.5, 1.87, 0.2]
#PANDA_REF_JNTS = [0.5, -0.8, 1.0, -2.5, 0.5, 1.87, 0.2]
PANDA_REF_JNTS = [0.5, -0.8, 0.9, -2.3, 0.5, 1.6, 0.2]


def get_robot_resample_param(a):
    if a.name.lower().find('move') < 0:
        if a.name.lower().find('grasp') >= 0: return [a.params[0], a.params[1]]
        if a.name.lower().find('lift') >= 0: return [a.params[0], a.params[1]]
        #if a.name.lower().find('place') >= 0: return [a.params[0], a.params[2]]
        if a.name.lower().find('slide') >= 0: return [a.params[0], a.params[1]]

    if a.name.lower().find('move') >= 0:
        if a.name.lower().find('put') >= 0: return [a.params[0], a.params[2]]
        if a.name.lower().find('place') >= 0: return [a.params[0], a.params[2]]

    return a.params[0]


def cleanup_robot_plan(plan, active_ts):
    for param in plan.params.values():
        if 'Robot' not in param.get_type(True): continue
        for arm in param.geom.arms:
            attr = '{}_ee_pos'.format(arm)
            rot_attr = '{}_ee_rot'.format(arm)
            if not hasattr(param, attr): continue
            for t in range(active_ts[0], active_ts[1]):
                if np.any(np.isnan(getattr(param, arm)[:,t])): continue
                param.openrave_body.set_dof({arm: getattr(param, arm)[:,t]})
                info = param.openrave_body.fwd_kinematics(arm)
                getattr(param, attr)[:,t] = info['pos']


def vertical_gripper(a_name, robot, arm, obj, obj_geom, gripper_open=True, ts=(0,20), rand=False, null_zero=True, disp=np.array([0, 0, const.GRASP_DIST]), rel_pos=True):
    start_ts, end_ts = ts
    robot_body = robot.openrave_body
    robot_body.set_from_param(robot, start_ts)

    old_arm_pose = getattr(robot, arm)[:, start_ts].copy()

    iks = []
    attempt = 0
    while not len(iks) and attempt < 20:
        off_mat = np.eye(3) if attempt == 0 else T.quat2mat(T.euler_to_quaternion([np.random.uniform(-np.pi/8, np.pi/8), 0., 0.], 'xyzw'))

        if 'door' in obj_geom.get_types():
            if a_name.find('slide') >= 0:
                euler = obj_geom.handle_orn
            else:
                euler = obj_geom.in_orn
        #elif a_name.find('lift') >= 0:
        #    euler = [1.57, 0., 0.]
        else:
            euler = obj.rotation[:,ts[0]] if not obj.is_symbol() else obj.rotation[:,0]

        obj_quat = T.euler_to_quaternion(euler, 'xyzw')
        obj_mat = off_mat.dot(T.quat2mat(obj_quat))
        quat = convert_orn(robot, arm, euler, off_mat=off_mat)
        offset = obj_geom.grasp_point if hasattr(obj_geom, 'grasp_point') else np.zeros(3)
        cur_disp = disp + offset
        if rel_pos:
            cur_disp = obj_mat.dot(cur_disp)

        if a_name.lower().find('slide') >= 0 and 'door' in obj_geom.get_types():
            target_loc = obj.pose[:, start_ts] + obj_geom.handle_pos + cur_disp
            if a_name.lower().find('open') >= 0:
                target_loc += np.abs(obj_geom.open_val) * np.array(obj_geom.open_dir)
            if a_name.lower().find('close') >= 0:
                target_loc -= np.abs(obj_geom.close_val) * np.array(obj_geom.open_dir)
        elif a_name.lower().find('place_in') >= 0 and 'door' in obj_geom.get_types():
            target_loc = obj.pose[:, start_ts] + obj_geom.in_pos + cur_disp
        elif obj.is_symbol():
            target_loc = obj.value[:, 0] + cur_disp
        else:
            target_loc = obj.pose[:, start_ts] + cur_disp

        robot_body.set_pose(robot.pose[:,ts[0]], robot.rotation[:,ts[0]])
        if a_name.lower().find('move') >= 0:
            robot_body.set_dof({arm: PANDA_REF_JNTS})
            #robot_body.set_dof({arm: (getattr(robot, arm)[:, ts[0]]+PANDA_REF_JNTS)/2.})
        else:
            robot_body.set_dof({arm: getattr(robot, arm)[:, ts[0]]})

        iks = robot_body.get_ik_from_pose(target_loc, quat, arm)
        robot_body.set_dof({arm: iks})
        info = robot_body.fwd_kinematics(arm)
        #if np.linalg.norm(target_loc-info['pos']) > 5e-2:
        #    iks = []
        attempt += 1

    if not len(iks): return None
    if a_name.find('lift') >= 0 and obj.name.find('upright') >= 0:
        iks[-1] += 1.57 if iks[-1] < -1.25 else -1.57
        #iks[-1] += 1.57 if iks[-1] < 0. else -1.57
    arm_pose = np.array(iks).reshape((-1,1))
    pose = {arm: arm_pose}
    gripper = robot.geom.get_gripper(arm)
    if gripper is not None:
        pose[gripper] = robot.geom.get_gripper_open_val() if gripper_open else robot.geom.get_gripper_closed_val()
        pose[gripper] = np.array(pose[gripper]).reshape((-1,1))

    for aux_arm in robot.geom.arms:
        if aux_arm == arm: continue
        old_pose = getattr(robot, aux_arm)[:, start_ts].reshape((-1,1))
        pose[aux_arm] = old_pose
        aux_gripper = robot.geom.get_gripper(aux_arm)
        if aux_gripper is not None:
            pose[aux_gripper] = getattr(robot, aux_gripper)[:, start_ts].reshape((-1,1))

    robot_body.set_pose(robot.pose[:,ts[0]], robot.rotation[:,ts[0]])
    for arm in robot.geom.arms:
        robot_body.set_dof({arm: pose[arm].flatten().tolist()})
        info = robot_body.fwd_kinematics(arm)
        pose['{}_ee_pos'.format(arm)] = np.array(info['pos']).reshape((-1,1))
        pose['{}_ee_rot'.format(arm)] = np.array(T.quaternion_to_euler(info['quat'], 'xyzw')).reshape((-1,1))

    pose['pose'] = robot.pose[:, ts[0]:ts[0]+1]
    pose['rotation'] = robot.rotation[:, ts[0]:ts[0]+1]

    return pose

def convert_orn(robot, arm, rot, to_robot=True, off_mat=np.eye(3)): 
    if len(rot) == 3:
        rot = T.euler_to_quaternion(rot, 'xyzw')

    targ_mat = off_mat.dot(T.quat2mat(rot))
    gripper_axis = robot.geom.get_gripper_axis(arm)
    target_axis = [0, 0, -1]
    quat = OpenRAVEBody.quat_from_v1_to_v2(gripper_axis, target_axis)
    robot_mat = T.quat2mat(quat)
    if to_robot:
        quat = T.mat2quat(targ_mat.dot(robot_mat))
    else:
        quat = T.mat2quat(targ_mat.dot(np.linalg.inv(robot_mat)))

    return quat


def objs_stacked(obj, base_obj, st):
    h1 = obj.geom.height if hasattr(obj.geom, 'height') else obj.geom.radius
    h2 = base_obj.geom.height if hasattr(base_obj.geom, 'height') else base_obj.geom.radius
    disp = (h1 + h2) / 2.
    pos = base_obj.pose[:,st] + np.array([0., 0., disp])
    rot = base_obj.rotation[:,st]
    pose = {'pose': pos.reshape((-1,1)), 'rotation': rot}
    return pose


def obj_in_gripper(ee_pos, targ_rot, obj):
    pose = {}
    targ_quat = targ_rot
    if len(targ_rot) == 3:
        targ_quat = T.euler_to_quaternion(targ_rot, 'xyzw')
    if len(targ_rot) == 4:
        targ_rot = T.quaternion_to_euler(targ_rot, 'xyzw')

    obj_mat = T.quat2mat(targ_quat)
    grasp_pt = obj_mat.dot(obj.geom.grasp_point).flatten()
    pose['pose'] = ee_pos.flatten() - grasp_pt
    pose['pose'] = pose['pose'].reshape((-1,1))
    pose['rotation'] = np.array(targ_rot).reshape((-1,1))
    return pose


def robot_gripper_open(a_name):
    gripper_open = False
    if a_name.find('move_to_grasp') >= 0:
        gripper_open = True
        
    if a_name.find('move') < 0 and \
       (a_name.find('putdown') >= 0 or \
        a_name.find('place') >= 0 or \
        a_name.find('slide') >= 0):
        gripper_open = True
    return gripper_open


def door_val(door, is_open=True, st=0):
    pose = {'pose': door.pose[:,st:st+1], 'rotation': door.rotation[:,st:st+1]}
    pose['hinge'] = door.geom.open_val if is_open else door.geom.close_val
    pose['hinge'] = np.array(pose['hinge']).reshape((1,1))
    return pose


def robot_obj_pose_suggester(plan, anum, resample_size=20, st=0):
    robot_pose = []
    assert anum + 1 <= len(plan.actions)

    act = plan.actions[anum]

    robot = act.params[0]
    robot_body = robot.openrave_body
    act_st, et = act.active_timesteps
    st = max(act_st, st)
    zero_null = True
    if hasattr(plan, 'freeze_ts') and plan.freeze_ts > 0:
        st = max(st, plan.freeze_ts)
        zero_null = False

    robot_body.set_pose(robot.pose[:,st], robot.rotation[:,st])
    obj = act.params[1]
    targ = act.params[2]
    obj_geom = obj.geom if hasattr(obj, 'geom') else targ.geom
    st, et = act.active_timesteps
    for param in plan.params.values():
        if hasattr(param, 'openrave_body') and param.openrave_body is not None:
            if param.is_symbol():
                if hasattr(param, 'rotation'):
                    param.openrave_body.set_pose(param.value[:,0], param.rotation[:,0])
                else:
                    param.openrave_body.set_pose(param.value[:,0])
            else:
                if hasattr(param, 'rotation'):
                    param.openrave_body.set_pose(param.pose[:,st], param.rotation[:,st])
                else:
                    param.openrave_body.set_pose(param.pose[:,st])

    a_name = act.name.lower()

    arm = None
    if a_name.find('left') >= 0: 
        arm = 'left'

    if a_name.find('right') >= 0: 
        arm = 'right'

    if arm:
        gripper_open = robot_gripper_open(a_name)

    rel_pos = True
    disp = np.array([0., 0., const.GRASP_DIST])
    if a_name.find('place') >= 0 or a_name.find('slide') >= 0:
        disp = np.array([0., 0., const.PLACE_DIST])

    if a_name.find('move') < 0 and \
       a_name.find('lift') >= 0:
        rel_pos = False
        y_offset = max(-0.14, 0.55-obj.pose[1,st])
        disp[0] = -obj.pose[0,st] / 3.
        #disp[1] = y_offset
        #disp[1] = (0.575-obj.pose[1,st]) / 2.
        disp[1] = (0.575-obj.pose[1,st]) / 2.

    if a_name.find('move') < 0 and \
        a_name.find('hold') >= 0:
        disp = np.zeros(3)

    rand = False
    ### Sample poses
    for i in range(resample_size):
        pose = {}

        if a_name == 'moveto':
            target_pose = act.params[2]
            pose['pose'] = target_pose.value.copy()
            for attr in ['rotation'] + robot.geom.arms + robot.geom.pos_jnts + robot.geom.rot_jnts + robot.geom.grippers:
                pose[attr] = getattr(target_pose, attr).copy()

        elif arm:
            pose.update(vertical_gripper(a_name, robot, arm, obj, obj_geom, gripper_open, 
                                         (st, et), rand=(rand or (i>0)), null_zero=zero_null, rel_pos=rel_pos, disp=disp))

            targ_pos = pose['{}_ee_pos'.format(arm)]
            targ_rot = convert_orn(robot, arm, pose['{}_ee_rot'.format(arm)].flatten(), to_robot=False)
            targ_rot = T.quaternion_to_euler(targ_rot, 'xyzw')

            if a_name.find('move') < 0 and \
                (a_name.find('grasp') >= 0 or \
                a_name.find('lift') >= 0):
                obj = act.params[1]
                targ = act.params[2]
                pose = {robot: pose, obj: obj_in_gripper(targ_pos, targ_rot, obj)}

            if a_name.find('move') >= 0 and \
            (a_name.find('put') >= 0 or \
                a_name.find('place') >= 0):
                obj = act.params[2]
                targ = act.params[1]
                if hasattr(targ, 'geom') and 'door' in targ.geom.get_types():
                    targ_rot = targ.geom.in_orn
                pose = {robot: pose, obj: obj_in_gripper(targ_pos, targ_rot, obj)}
                obj = act.params[1]
                targ = act.params[2]

            if a_name.find('slide') >= 0:
                door = act.params[1]
                is_open = a_name.find('open') >= 0
                door_pose = door_val(door, is_open, st)
                pose = {robot: pose, door: door_pose}

            # if a_name.find('place_in_door') >= 0:
            #     door = act.params[1]
            #     obj = act.params[2]
            #     obj_pose = {'pose': (door.pose[:, st] + np.array(door.geom.in_pos)).reshape((3,1)),
            #                'rotation': np.array(door.geom.in_orn).reshape((3,1))}
            #     pose = {robot: pose, obj: obj_pose}

        if pose is None: break
        robot_pose.append(pose)

    return robot_pose


if USE_GRB:
    class RobotSolverGurobi(backtrack_ll_solver_gurobi.BacktrackLLSolverGurobi):
        def get_resample_param(self, a):
            return get_robot_resample_param(a)

        def freeze_rs_param(self, act):
            return True


        def obj_pose_suggester(self, plan, anum, resample_size=20, st=0):
            return robot_obj_pose_suggester(plan, anum, resample_size=20, st=0)


        def _cleanup_plan(self, plan, active_ts):
            return cleanup_robot_plan(plan, active_ts)


        def _get_trajopt_obj(self, plan, active_ts=None):
            if active_ts == None:
                active_ts = (0, plan.horizon-1)
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
                            KT = K*T
                            v = -1 * np.ones((KT - K, 1))
                            d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                            # [:,0] allows numpy to see v and d as one-dimensional so
                            # that numpy will create a diagonal matrix with v and d as a diagonal
                            P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                            Q = np.dot(np.transpose(P), P)
                            Q *= self.trajopt_coeff/float(plan.horizon)

                            quad_expr = None
                            coeff = 1.
                            if attr_name.find('ee_pos') >= 0:
                                coeff = 7e-3
                            elif attr_name.find('ee_rot') >= 0:
                                coeff = 2e-3
                            elif attr_name.find('right') >= 0 or attr_name.find('left') >= 0:
                                coeff = 1e1
                            else:
                                coeff = 1e-2

                            quad_expr = QuadExpr(coeff*Q, np.zeros((1,KT)), np.zeros((1,1)))
                            param_ll = self._param_to_ll[param]
                            ll_attr_val = getattr(param_ll, attr_name)
                            param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
                            attr_val = getattr(param, attr_name)
                            init_val = attr_val[:, start:end+1].reshape((KT, 1), order='F')
                            sco_var = self.create_variable(param_ll_grb_vars, init_val)
                            bexpr = BoundExpr(quad_expr, sco_var)
                            traj_objs.append(bexpr)

            return traj_objs


class RobotSolverOSQP(backtrack_ll_solver_OSQP.BacktrackLLSolverOSQP):
    def get_resample_param(self, a):
        return get_robot_resample_param(a)

    def freeze_rs_param(self, act):
        return True


    def obj_pose_suggester(self, plan, anum, resample_size=20, st=0):
        return robot_obj_pose_suggester(plan, anum, resample_size=20, st=0)


    def _cleanup_plan(self, plan, active_ts):
        return cleanup_robot_plan(plan, active_ts)


    def _get_trajopt_obj(self, plan, active_ts=None):
        if active_ts == None:
            active_ts = (0, plan.horizon-1)
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
                        KT = K*T
                        v = -1 * np.ones((KT - K, 1))
                        d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                        # [:,0] allows numpy to see v and d as one-dimensional so
                        # that numpy will create a diagonal matrix with v and d as a diagonal
                        P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                        Q = np.dot(np.transpose(P), P)
                        Q *= self.trajopt_coeff/float(plan.horizon)

                        quad_expr = None
                        coeff = 1.
                        if attr_name.find('ee_pos') >= 0:
                            coeff = 7e-3
                        elif attr_name.find('ee_rot') >= 0:
                            coeff = 2e-3
                        elif attr_name.find('right') >= 0 or attr_name.find('left') >= 0:
                            coeff = 1e1
                        else:
                            coeff = 1e-2

                        quad_expr = QuadExpr(coeff*Q, np.zeros((1,KT)), np.zeros((1,1)))
                        param_ll = self._param_to_ll[param]
                        ll_attr_val = getattr(param_ll, attr_name)
                        param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
                        attr_val = getattr(param, attr_name)
                        init_val = attr_val[:, start:end+1].reshape((KT, 1), order='F')
                        sco_var = self.create_variable(param_ll_grb_vars, init_val)
                        bexpr = BoundExpr(quad_expr, sco_var)
                        traj_objs.append(bexpr)

        return traj_objs
