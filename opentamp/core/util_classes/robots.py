from collections import OrderedDict
import os

import pybullet as p

import numpy as np
import opentamp
import opentamp.core.util_classes.transform_utils as T


JNT_SLACK = 1e-4

class Robot(object):
    """
    Base class of every robot parameter
    """
    def __init__(self, shape):
        self.shape = shape
        self.file_type = 'urdf'
        self.initialized = False
        self.arms = []
        self.pos_jnts = []
        self.rot_jnts = []
        self.grippers = []
        self.ee_attrs = []
        self.id = -1
        self.attr_map = {}
        self._base_type = "Robot"
        self._type = "Robot"

    def get_types(self):
        return [self._type, self._base_type]

    def _init_attr_map(self):
        robot_map = []
        robot_pose_map = []
        for arm in self.arms:
            arm_dim = len(self.get_arm_inds(arm)) 
            robot_map.append((arm, np.array(range(arm_dim), dtype=np.int)))
            robot_pose_map.append((arm, np.array(range(arm_dim), dtype=np.int)))

        for gripper in self.grippers:
            gripper_dim = 1
            robot_map.append((gripper, np.array(range(gripper_dim), dtype=np.int)))
            robot_pose_map.append((gripper, np.array(range(gripper_dim), dtype=np.int)))

        for pos_jnt in self.pos_jnts:
            dim = len(self.jnt_names[pos_jnt])
            robot_map.append((pos_jnt, np.array(range(dim), dtype=np.int)))
            robot_pose_map.append((pos_jnt, np.array(range(dim), dtype=np.int)))

        for rot_jnt in self.rot_jnts:
            dim = len(self.jnt_names[rot_jnt])
            robot_map.append((rot_jnt, np.array(range(dim), dtype=np.int)))
            robot_pose_map.append((rot_jnt, np.array(range(dim), dtype=np.int)))

        base = self.get_base_move_limit().flatten()
        robot_map.append(('pose', np.array(range(len(base)),dtype=np.int))) 
        robot_pose_map.append(('value', np.array(range(len(base)),dtype=np.int)))
        self.attr_map['robot'] = robot_map
        self.attr_map['robot_pose'] = robot_pose_map

    def is_initialized(self):
        return self.initialized

    def setup(self, robot=None):
        self.setup_arms()
        self.intialized = True

    def get_joint_move_factor(self):
        return 5

    def get_base_limit(self):
        return np.array([-10, -10, -5*np.pi]), np.array([10, 10, 5*np.pi])
    
    def get_base_move_limit(self):
        return np.array([0.1, 0.1, np.pi/8])

    def get_arm_inds(self, arm):
        return self.arm_inds[arm]

    def get_pos_jnt_inds(self, pos_jnt):
        return self.pos_jnt_inds[pos_jnt]

    def get_rot_jnt_inds(self, rot_jnt):
        return self.rot_jnt_inds[rot_jnt]

    def get_free_inds(self, key=None):
        if key is None:
            return list(sorted(self.free_joints.keys()))
        
        if key in self.arms:
            inds = self.get_arm_inds(key)
        elif key in self.pos_jnts:
            inds = self.get_pos_jnt_inds(key)
        elif key in self.rot_jnts:
            inds = self.get_rot_jnt_inds(key)
        else:
            raise ValueError(f'Invalid inds key {key}')

        return [self.free_joints[ind] for ind in inds]

    def get_free_jnts(self):
        jnt_ids = sorted(self.free_joints.keys())
        return list([jnt[0] for jnt in p.getJointStates(self.id, jnt_ids)]) 

    def get_ee_link(self, arm):
        return self.ee_links[arm]

    def gripper_dim(self, arm):
        gripper = self.get_gripper(arm)
        return len(self.jnt_names[gripper])

    def get_gripper(self, arm):
        return self.ee_link_names[arm]

    def get_gripper_axis(self, arm):
        return self.gripper_axis[arm]

    def get_arm_bnds(self, arm=None):
        if arm is None:
            return self.lb, self.ub
        return self.arm_bnds[arm]

    def get_joint_limits(self, arm=None):
        if arm is not None: return self.jnt_limits[arm]

        limits = []
        for arm in self.arms:
            limits.append(self.jnt_limits[arm])
        return limits

    def get_shape(self):
        return self.shape

    def get_type(self):
        return self.file_type

    def get_dof_inds(self):
        return list(self.dof_inds.items())

    def get_gripper_open_val(self, arm=None, scalar=False):
        if arm is None:
            arm = self.arms[0]

        gripper = self.ee_link_names[arm]
        #ub = np.max(self.jnt_limits[gripper][1])
        if not scalar: return self.jnt_limits[gripper][1]

        ub = self.jnt_limits[gripper][1][0]
        return ub

    def get_gripper_closed_val(self, arm=None, scalar=False):
        if arm is None:
            arm = self.arms[0]

        gripper = self.ee_link_names[arm]
        #lb = np.min(self.jnt_limits[gripper][0])
        if not scalar: return self.jnt_limits[gripper][0]

        lb = self.jnt_limits[gripper][0][0]
        return lb

    def _init_pybullet(self):
        if self.shape.endswith('urdf'):
            self.id = p.loadURDF(self.shape)
        elif self.shape.endswith('mjcf'):
            self.id = p.loadMJCF(self.shape)
        elif self.shape.endswith('xml'):
            self.id = p.loadMJCF(self.shape)

        if type(self.id) is not int:
            for i in range(len(self.id)):
                if p.getNumJoints(self.id[i]) > 0:
                    self.id = self.id[i]
                    break

    def infer_gripper_axis(self, arm):
        """
        Helper function to infer direction gripper points from current rotation
        """
        ee_link = self.get_ee_link(arm)
        parent = p.getJointInfo(self.id, ee_link)[-1]
        parent_pos = p.getLinkState(self.id, parent)[0]
        ee_pos = p.getLinkState(self.id, ee_link)[0]
        axis = np.array(ee_pos) - np.array(parent_pos)
        axis = axis.round(4)
        axis = axis / np.linalg.norm(axis)
        return axis

    def setup_arms(self):
        if self.id < 0:
            self._init_pybullet()

        self.grippers = list(self.ee_link_names.values())

        # Setup dof inds, which are used to idnex into state vectors
        self.dof_inds = OrderedDict()
        cur_ind = 0
        for arm in self.arms:
            n_jnts = len(self.jnt_names[arm])
            self.dof_inds[arm] = np.array(range(cur_ind, cur_ind+n_jnts))
            cur_ind += n_jnts
            ee_name = self.ee_link_names[arm]
            self.dof_inds[ee_name] = np.array([cur_ind])
            cur_ind += 1
        self.dof_inds['pose'] = np.array(range(cur_ind, cur_ind+3))

        self.dof_map = {}
        self.arm_inds = {}
        self.gripper_inds = {}
        self.pos_jnt_inds = {}
        self.rot_jnt_inds = {}
        self.ee_links = {}
        self.jnt_limits = {}
        self.free_joints = {}
        self.lb, self.ub = [], []
        self.arm_links = {}

        # Setup tracking for pybullet indices <-> link/jnt names
        self.jnt_to_id = {}
        self.id_to_jnt = {}
        self.link_to_id = {}
        self.id_to_link = {}
        self.jnt_parents = {}
        self.bounds = {}
        cur_free = 0
        n_decs = 6
        for i in range(p.getNumJoints(self.id)):
            jnt_info = p.getJointInfo(self.id, i)
            jnt_name = jnt_info[1].decode('utf-8')
            self.jnt_to_id[jnt_name] = i
            self.id_to_jnt[i] = jnt_name
            self.link_to_id[jnt_info[12].decode('utf-8')] = i
            self.id_to_link[i] = jnt_info[12].decode('utf-8')
            self.jnt_parents[jnt_name] = jnt_info[-1]
            jnt_lb = np.trunc(10**n_decs * jnt_info[8]) / 10**n_decs - JNT_SLACK
            jnt_ub = np.trunc(10**n_decs * jnt_info[9]) / 10**n_decs + JNT_SLACK
            self.bounds[jnt_name] = (jnt_lb, jnt_ub)

            # Track free joints (necessary for IK solves & similar)
            if jnt_info[2] != p.JOINT_FIXED:
                self.free_joints[i] = cur_free
                self.lb.append(jnt_lb)
                self.ub.append(jnt_ub)
                cur_free += 1

        # Setup tracking for pybullet indices <-> arm attributes
        for arm in self.arms:
            jnt_names = self.jnt_names[arm]
            self.dof_map[arm] = [self.jnt_to_id[jnt] for jnt in jnt_names]
            self.arm_inds[arm] = [self.jnt_to_id[jnt] for jnt in jnt_names]
            self.ee_links[arm] = self.link_to_id[self.ee_link_names[arm]]
            self.jnt_limits[arm] = (np.array([self.bounds[jnt][0] for jnt in jnt_names]), np.array([self.bounds[jnt][1] for jnt in jnt_names]))

            # Assumes parent link ids are always less than child link ids
            self.arm_links[arm] = [self.jnt_parents[jnt] for jnt in jnt_names if self.jnt_parents[jnt] >= 0]
            for i in range(p.getNumJoints(self.id)):
                info = p.getJointInfo(self.id, i)
                parent_link = info[-1]
                if parent_link in self.arm_links[arm]:
                    self.arm_links[arm].append(i)

        for gripper in self.grippers:
            jnt_names = self.jnt_names[gripper]
            self.dof_map[gripper] = [self.jnt_to_id[jnt] for jnt in jnt_names]
            self.gripper_inds[gripper] = [self.jnt_to_id[jnt] for jnt in jnt_names]
            self.jnt_limits[gripper] = (np.array([self.bounds[jnt][0] for jnt in jnt_names]), np.array([self.bounds[jnt][1] for jnt in jnt_names]))

        for pos_jnt in self.pos_jnts:
            jnt_names = self.jnt_names[pos_jnt]
            self.dof_map[pos_jnt] = [self.jnt_to_id[jnt] for jnt in jnt_names]
            self.pos_jnt_inds[pos_jnt] = [self.jnt_to_id[jnt] for jnt in jnt_names]
            self.jnt_limits[pos_jnt] = (np.array([self.bounds[jnt][0] for jnt in jnt_names]), np.array([self.bounds[jnt][1] for jnt in jnt_names]))

        for rot_jnt in self.rot_jnts:
            jnt_names = self.jnt_names[rot_jnt]
            self.dof_map[rot_jnt] = [self.jnt_to_id[jnt] for jnt in jnt_names]
            self.rot_jnt_inds[rot_jnt] = [self.jnt_to_id[jnt] for jnt in jnt_names]
            self.jnt_limits[rot_jnt] = (np.array([self.bounds[jnt][0] for jnt in jnt_names]), np.array([self.bounds[jnt][1] for jnt in jnt_names]))

        self.col_links = set([self.link_to_id[name] for name in self.col_link_names])
        self._init_attr_map()
        
        # Store axis pointing from gripper base to finger tips
        self.gripper_axis = {}
        for arm in self.arms:
            ee_link = self.get_ee_link(arm)
            cur_quat = p.getLinkState(self.id, ee_link)[1]
            mat = T.quat2mat(cur_quat)
            cur_axis = self.infer_gripper_axis(arm)
            axis = np.linalg.inv(mat).dot(cur_axis)
            self.gripper_axis[arm] = axis.round(4)


class NAMO(Robot):
    def __init__(self):
        self._type = "robot"
        self._base_type = "robot"
        self.file_type = 'mjcf'
        self.radius = 0.3
        self.shape = opentamp.__path__._path[0]+'/robot_info/lidar_namo.xml'
        self.dof_map = {'xpos': 0, 'ypos': 1, 'robot_theta': 2, 'left_grip': 6, 'right_grip': 4}
        self._init_pybullet()
        self.initialized = True


class TwoLinkArm(Robot):
    def __init__(self):
        self._type = "robot"
        self.file_type = 'mjcf'
        self.radius = 0.3
        self.shape = opentamp.__path__._path[0]+'/robot_info/lidar_arm.xml'
        self.dof_map = {'joint1': 0, 'joint2': 2, 'wrist': 4, 'left_grip': 11, 'right_grip': 8}
        self.link_to_ind = {'link1': 1, 'link2': 3, 'wrist': 5, 'ee': 6, 'ee_far': 7, 'right_finger': 8, 'right_finger_tip': 9, 'left_finger': 11, 'left_finger_tip': 12}
        self.ind_to_link = {v:k for k, v in self.link_to_ind.items()}
        self.jnt_to_body = {0:0, 2:2, 4:4}
        self.upper_bounds = np.array([10, 3.1, 3.1, 0.4, 0.4])
        self.lower_bounds = np.array([-10, -3.1, -3.1, -0.4, -0.4])
        self.arm_links = [1, 3, 5]
        self.ee_link = 6
        self.far_ee_link = 7
        self.ee_links = [6, 7, 8, 9, 11, 12]
        self.col_links = [1, 3, 8, 9, 10, 11, 12, 13] # [1, 3, 8, 10]


class PR2(Robot):
    """
    Defines geometry used in the PR2 domain.
    """
    def __init__(self):
        self._type = "pr2"
        pr2_shape = "../models/pr2/pr2.zae"
        # down to 30 links from 45
        self.col_links = set(['base_link', 'torso_lift_link', 'l_shoulder_pan_link',
                              'l_shoulder_lift_link', 'l_upper_arm_roll_link', 'l_upper_arm_link',
                              'l_elbow_flex_link', 'l_forearm_roll_link', 'l_forearm_link',
                              'l_wrist_flex_link', 'l_wrist_roll_link', 'l_gripper_palm_link',
                              'l_gripper_l_finger_link', 'l_gripper_l_finger_tip_link',
                              'l_gripper_r_finger_link', 'l_gripper_r_finger_tip_link', 'r_shoulder_pan_link',
                              'r_shoulder_lift_link', 'r_upper_arm_roll_link', 'r_upper_arm_link',
                              'r_elbow_flex_link', 'r_forearm_roll_link', 'r_forearm_link', 'r_wrist_flex_link',
                              'r_wrist_roll_link', 'r_gripper_palm_link', 'r_gripper_l_finger_link',
                              'r_gripper_l_finger_tip_link', 'r_gripper_r_finger_link',
                              'r_gripper_r_finger_tip_link'])
        self.dof_map = {"backHeight": [12], "lArmPose": list(range(15,22)), "lGripper": [22], "rArmPose": list(range(27,34)), "rGripper":[34]}
        super(PR2, self).__init__(pr2_shape)

    def setup(self, robot):
        """
        Nothing to setup for pr2
        """
        return


class Baxter(Robot):
    """
    Defines geometry used in the Baxter domain.
    """
    def __init__(self):
        self._type = "baxter"
        baxter_shape = opentamp.__path__._path[0] + "/robot_info/baxter_model.xml"
        super(Baxter, self).__init__(baxter_shape)

        self.jnt_names = {'left':  ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2'],
                          'left_gripper': ['left_gripper_l_finger_joint', 'left_gripper_r_finger_joint'],
                          'right': ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2'],
                          'right_gripper': ['right_gripper_l_finger_joint', 'right_gripper_r_finger_joint'],
                          }
        self.ee_link_names = {'left': 'left_gripper', 'right': 'right_gripper'}
        self.arms = ['left', 'right']
        self.ee_attrs = ['left_ee_pos', 'right_ee_pos', 'left_ee_rot', 'right_ee_rot']
        #self.arm_inds = {'left':  [31, 32, 33, 34, 35, 37, 38],
        #                 'right': [13, 14, 15, 16, 17, 19, 20]}
        #self.ee_links = {'left': 45, 'right': 27}
        self.arm_bnds = {'left': (0,7), 'right': (8, 15)}
        #self.jnt_limits = {'left':  ([-1.701, -2.145, -3.05, -0.05, -3.059, -1.57, -3.059], [1.70, 1.04, 3.05, 2.61, 3.059, 2.094, 3.059]),
        #                   'right': ([-1.701, -2.145, -3.05, -0.05, -3.059, -1.57, -3.059], [1.70, 1.04, 3.05, 2.61, 3.059, 2.094, 3.059])}

        #self.col_link_names = set(["torso", "head", "sonar_ring", "screen", "collision_head_link_1",
        #                      "collision_head_link_2", "right_upper_shoulder", "right_lower_shoulder",
        #                      "right_upper_elbow", "right_upper_elbow_visual", "right_lower_elbow",
        #                      "right_upper_forearm", "right_upper_forearm_visual", "right_lower_forearm",
        #                      "right_wrist", "right_hand", "right_gripper_base", "right_gripper",
        #                      "right_gripper_l_finger", "right_gripper_r_finger", "right_gripper_l_finger_tip",
        #                      "right_gripper_r_finger_tip", "left_upper_shoulder", "left_lower_shoulder",
        #                      "left_upper_elbow", "left_upper_elbow_visual", "left_lower_elbow",
        #                      "left_upper_forearm", "left_upper_forearm_visual", "left_lower_forearm",
        #                      "left_wrist", "left_hand", "left_gripper_base", "left_gripper",
        #                      "left_gripper_l_finger", "left_gripper_r_finger", "left_gripper_l_finger_tip",
        #                      "left_gripper_r_finger_tip"])

        self.col_link_names = set(["torso", "head", "screen", "collision_head_link_1",
                              "collision_head_link_2", "right_upper_shoulder", "right_lower_shoulder",
                              "right_upper_elbow", "right_lower_elbow",
                              "right_upper_forearm", "right_lower_forearm",
                              "right_wrist", "right_hand", "right_gripper_base", "right_gripper_l_finger_tip",
                              "right_gripper_r_finger_tip", "left_upper_shoulder", "left_lower_shoulder",
                              "left_upper_elbow", "left_lower_elbow",
                              "left_upper_forearm", "left_lower_forearm",
                              "left_wrist", "left_hand", "left_gripper_base", "left_gripper_l_finger_tip",
                              "left_gripper_r_finger_tip"])
        #self.ik_solver = BaxterIKController(lambda: np.zeros(14))
        #self.col_links = set([self.ik_solver.name2id(name) for name in self.col_link_names])
        #self.dof_map = {"lArmPose": [31, 32, 33, 34, 35, 37, 38],
        #                "rArmPose": [13, 14, 15, 16, 17, 19, 20]}


class Sawyer(Robot):
    """
    Defines geometry used in the Baxter domain.
    """
    def __init__(self):
        self._type = "sawyer"
        sawyer_shape = opentamp.__path__._path[0] + "/robot_info/sawyer/robot.xml"
        super(Sawyer, self).__init__(sawyer_shape)

        self.jnt_names = {'right':  ['right_j0','right_j1','right_j2','right_j3','right_j4','right_j5','right_j6'],
                          'right_gripper': ['l_finger_joint', 'r_finger_joint'],
                          }
        self.ee_link_names = {'right': 'right_gripper'}
        self.arms = ['right']
        self.ee_attrs = ['right_ee_pos', 'right_ee_rot']
        self.arm_bnds = {'right': (0,7)}
        self.col_link_names = set(['right_arm_base_link', 'right_l0', 'right_l1', 
                                   'right_l2', 'right_l3', 'right_l4', 'right_l5',
                                   'right_l6', 'gripper_base'])


class Panda(Robot):
    """
    Defines geometry used in the Baxter domain.
    """
    def __init__(self):
        self._type = "panda"
        panda_shape = opentamp.__path__._path[0] + "/robot_info/robodesk/panda.xml"
        super(Panda, self).__init__(panda_shape)

        self.jnt_names = {'right': ['panda0_joint1', 'panda0_joint2', 'panda0_joint3', 
                                    'panda0_joint4', 'panda0_joint5', 'panda0_joint6',
                                    'panda0_joint7'],
                          'right_gripper': ['panda0_finger_joint1', 'panda0_finger_joint2'],
                          }
        self.ee_link_names = {'right': 'right_gripper'}
        self.arms = ['right']
        self.ee_attrs = ['right_ee_pos', 'right_ee_rot']
        self.arm_bnds = {'right': (0,7)}
        self.col_link_names = set(['panda0_link1', 'panda0_link2', 'panda0_link3',
                                   'panda0_link4', 'panda0_link5', 'panda0_link6', 'panda0_link7', 
                                   'panda0_leftfinger', 'panda0_rightfinger'])


class Spot(Robot):
    """
    Defines geometry used in the Spot domain.
    """
    def __init__(self):
        self._type = "spot"
        spot_shape = opentamp.__path__._path[0] + "/robot_info/spot_simple.xml"
        super(Spot, self).__init__(spot_shape)

        self.jnt_names = {'position': ['robot_x', 'robot_y'],
                          'theta': ['robot_theta'],
                          }
        self.pos_jnts = ['position']
        self.rot_jnts = ['theta']
        self.ee_link_names = {}
        self.arms = []
        self.ee_attrs = []
        self.arm_bnds = {}
        self.col_link_names = set(['spot'])

    def get_base_limit(self):
        return np.array([-10, -10, -5*np.pi]), np.array([10, 10, 5*np.pi])
    
    def get_base_move_limit(self):
        return np.array([0.5, 0.5, np.pi/4])
    

class HSR(Robot):
    """
    Defines geometry used in the HSR domain.
    """
    def __init__(self):
        self._type = "hsr"
        shape = "../models/hsr/hsrb4s.xml"
        self.col_links = set(['arm_lift_link', 'arm_flex_link', 'arm_roll_link',
                              'wrist_flex_link', 'wrist_roll_link', 'hand_palm_link',
                              'hand_l_proximal_link', 'hand_l_spring_proximal_link',
                              'hand_l_mimic_distal_link', 'hand_l_distal_link',
                              'hand_motor_dummy_link', 'hand_r_proximal_link',
                              'hand_r_spring_proximal_link', 'hand_r_mimic_distal_link',
                              'hand_r_distal_link', 'base_roll_link',
                              'base_l_drive_wheel_link', 'base_l_passive_wheel_z_link',
                              'base_r_drive_wheel_link', 'base_r_passive_wheel_z_link',
                              'torso_lift_link', 'head_pan_link', 'head_tilt_link'])
        self.dof_map = {'arm': [0, 1, 2, 3, 4], 'gripper': [6]}
        super(HSR, self).__init__(shape)


class Washer(Robot):
    """
        Object stores all the information to for a Washer model
    """

    def __init__(self, mockup = True, topload = False):
        self._type = "washer"
        if mockup:
            self.shape = "../models/items/washer_mock_up/washer_col_2.xml"
            """
            to variate the depth of the mockup washer, simply change the y-cord
            of the washer_bottom body in washer.xml
            """
            if topload:
                self.good_pos = np.array([0.416, 1.509, 0.825])
                self.good_rot = np.array([np.pi, 0, np.pi/2])
            else:
                self.good_pos = np.array([0.505, 1.161, 1.498])
                self.good_rot = np.array([np.pi, 0, 0])
        else:
            self.shape = "../models/items/washer.xml"
            self.up_right_rot = [1.57, 0, 0]
            self.good_pos = np.array([0.5, 0.8, 0])
            self.good_rot = np.array([np.pi/2, 0, 0])
        self.dof_map = {"door": [0]}

        self.col_links = set(["external_1", "external_2", "external_3", "external_4", "back",
                              "corner_1", "corner_2", "corner_3", "corner_4",
                              "strip_1", "strip_2", "strip_3", "strip_4", "washer_door",
                              "washer_handle", "barrel_1", "barrel_2",
                              "barrel_3", "barrel_4", "barrel_5",
                              "barrel_6", "barrel_7", "barrel_8",
                              "barrel_9", "barrel_10", "barrel_11",
                              "barrel_12", "barrel_13", "barrel_14",
                              "barrel_15", "barrel_16", "barrel_17",
                              "barrel_18", "barrel_19", "barrel_20",
                              "barrel_21", "barrel_22", "barrel_23",
                              "barrel_24", "barrel_25", "barrel_26",
                              "barrel_27", "barrel_28", "barrel_29",
                              "barrel_30", "barrel_back", "base"])

    def setup(self, robot):
        pass
