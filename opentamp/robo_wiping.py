import os
import sys
import time

import numpy as np
import pybullet as P
import robosuite
import robosuite.utils.transform_utils as robo_T
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation

import opentamp.core.util_classes.transform_utils as T
import main
from opentamp.core.parsing import parse_domain_config, parse_problem_config
from opentamp.core.util_classes.openrave_body import *
from opentamp.core.util_classes.transform_utils import *
from opentamp.core.util_classes.viewer import PyBulletViewer
from pma import backtrack_ll_solver_gurobi as bt_ll
from pma.hl_solver import *
from pma.pr_graph import *
from pma.robosuite_solver import RobotSolver
from sco_py.expr import *
import faulthandler

SAWYER_INIT_POSE = [-0.41, 0.0, 0.912]
SAWYER_END_POSE = [0, 0, 0]
R_ARM_INIT = [-0.3962099, -0.97739413, 0.04612799, 1.742205 , -0.03562013, 0.8089644, 0.45207395]
OPEN_GRIPPER = [0.02, -0.01]
CLOSE_GRIPPER = [0.01, -0.02]
GRIPPER_SIZE = [0.05, 0.12, 0.015]
EE_POS = [0.11338, -0.16325, 1.03655]
EE_ROT = [3.139, 0.00, -2.182]
TABLE_GEOM = [0.25, 0.40, 0.820]
TABLE_POS = [0.15, 0.00, 0.00]
TABLE_ROT = [0,0,0]
NUM_ROWS = int((TABLE_GEOM[0] * 2) / GRIPPER_SIZE[0])
NUM_COLS = int((TABLE_GEOM[1] * 2) / GRIPPER_SIZE[1])

REF_QUAT = np.array([0, 0, -0.7071, -0.7071])

def theta_error(cur_quat, next_quat):
    sign1 = np.sign(cur_quat[np.argmax(np.abs(cur_quat))])
    sign2 = np.sign(next_quat[np.argmax(np.abs(next_quat))])
    next_quat = np.array(next_quat)
    cur_quat = np.array(cur_quat)
    angle = -(sign1 * sign2) * robo_T.get_orientation_error(
        sign1 * next_quat, sign2 * cur_quat
    )
    return angle


# controller_config = load_controller_config(default_controller="OSC_POSE")
# controller_config = load_controller_config(default_controller="JOINT_VELOCITY")
# controller_config['control_delta'] = False
# controller_config['kp'] = 500
# controller_config['kp'] = [750, 750, 500, 5000, 5000, 5000]

ctrl_mode = "JOINT_POSITION"
true_mode = "JOINT"

# ctrl_mode = 'OSC_POSE'
# true_mode = 'IK'

controller_config = load_controller_config(default_controller=ctrl_mode)
if ctrl_mode.find("JOINT") >= 0:
    controller_config["kp"] = [7500, 6500, 6500, 6500, 6500, 6500, 12000]
    controller_config["output_max"] = 0.2
    controller_config["output_min"] = -0.2
else:
    controller_config["kp"] = 5000  # [8000, 8000, 8000, 4000, 4000, 4000]
    controller_config["input_max"] = 0.2  # [0.05, 0.05, 0.05, 4, 4, 4]
    controller_config["input_min"] = -0.2  # [-0.05, -0.05, -0.05, -4, -4, -4]
    controller_config["output_max"] = 0.02  # [0.1, 0.1, 0.1, 2, 2, 2]
    controller_config["output_min"] = -0.02  # [-0.1, -0.1, -0.1, -2, -2, -2]


visual = len(os.environ.get("DISPLAY", "")) > 0
has_render = visual
obj_mode = 2
env = robosuite.make(
    "Wipe",
    robots=["Sawyer"],             # load a Sawyer robot
    controller_configs=controller_config,   # each arm is controlled using OSC
    has_renderer=has_render,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera
    has_offscreen_renderer=(not has_render),           # no off-screen rendering
    control_freq=50,                        # 50 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=True,                   # no observations needed
    use_camera_obs=False,                   # no observations needed
    ignore_done=True,
    reward_shaping=True,
    initialization_noise={'magnitude': 0., 'type': 'gaussian'},
    camera_widths=128,
    camera_heights=128,
)
obs, _, _, _ = env.step(np.zeros(7)) # Step a null action to 'boot' the environment.

# Get the locations of all dirt particles
dirt_locs = np.zeros((env.num_markers, 3))
for i, marker in enumerate(env.model.mujoco_arena.markers):
    marker_pos = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(marker.root_body)])
    dirt_locs[i,:] = marker_pos

# First, we reset the environment and then manually set the joint positions to their
# initial positions and all the joint velocities and accelerations to 0.
obs = env.reset()
jnts = env.sim.data.qpos[:7]
for _ in range(40):
    env.step(np.zeros(7))
    env.sim.data.qpos[:7] = jnts
    env.sim.forward()
env.sim.data.qvel[:] = 0
env.sim.data.qacc[:] = 0
env.sim.forward()

bt_ll.DEBUG = True
openrave_bodies = None
domain_fname = os.getcwd() + "/opentamp/domains/robot_wiping_domain/right_wipe_onlytable.domain"
prob = os.getcwd() + "/opentamp/domains/robot_wiping_domain/probs/simple_move_onlytable_prob.prob"
d_c = main.parse_file_to_dict(domain_fname)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
hls = FFSolver(d_c)
p_c = main.parse_file_to_dict(prob)
visual = len(os.environ.get('DISPLAY', '')) > 0
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=visual)
params = problem.init_state.params
body_ind = env.mjpy_model.body_name2id("robot0_base")
table_ind = env.mjpy_model.body_name2id("table")


# Computes the dirty regions set, which contains a tuple (row, col) for every
# region that is dirty. 
dirty_regions = set()
row_step_size = (TABLE_GEOM[0] * 2) / NUM_ROWS
col_step_size = (TABLE_GEOM[1] * 2) / NUM_COLS
for xyz_pose in dirt_locs.tolist():
    x_pos = xyz_pose[0]
    y_pos = xyz_pose[1]
    # Analytical way to compute.
    row_idx = (x_pos - (TABLE_POS[0] - TABLE_GEOM[0])) // row_step_size
    col_idx = (y_pos - (TABLE_POS[1] - TABLE_GEOM[1])) // col_step_size
    # Stupider way to do it.
    closest_row_col = (0, 0)
    dist_to_closest = float("inf")
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            xyz_pose = params[f"region_pose{row}_{col}"].right_ee_pos.squeeze()
            xy_pose = xyz_pose[:-1]
            dist_to_rowcol = np.linalg.norm(xy_pose - np.array([x_pos, y_pos]))
            if dist_to_rowcol < dist_to_closest:
                closest_row_col = (row, col)
                dist_to_closest = dist_to_rowcol

    # dirty_regions.add((int(row_idx), int(col_idx)))
    dirty_regions.add(closest_row_col)


# Resetting the initial state to match the robotsuite sim.
params["sawyer"].pose[:, 0] = env.sim.data.body_xpos[body_ind]
# NOTE: for the table, we only want to set the (x,y) poses to
# be equal, because we use a different geometry and thus the
# height must be different.
params["table"].pose[:2, 0] = env.sim.data.body_xpos[table_ind][:2]
jnts = params["sawyer"].geom.jnt_names["right"]
jnts = ["robot0_" + jnt for jnt in jnts]
jnt_vals = []
sawyer_inds = []
for jnt in jnts:
    jnt_adr = env.mjpy_model.joint_name2id(jnt)
    jnt_ind = env.mjpy_model.jnt_qposadr[jnt_adr]
    sawyer_inds.append(jnt_ind)
    jnt_vals.append(env.sim.data.qpos[jnt_ind])
params["sawyer"].right[:, 0] = jnt_vals
params["robot_init_pose"].right[:, 0] = jnt_vals
params["robot_init_pose"].value[:, 0] = params["sawyer"].pose[:, 0]
params["sawyer"].openrave_body.set_pose(params["sawyer"].pose[:, 0])
params["sawyer"].openrave_body.set_dof({"right": params["sawyer"].right[:, 0]})
info = params["sawyer"].openrave_body.fwd_kinematics("right")
params["sawyer"].right_ee_pos[:, 0] = info["pos"]
params["sawyer"].right_ee_pos[:, 0] = T.quaternion_to_euler(info["quat"], "xyzw")

goal = "(and"
for dirty_region in dirty_regions:
    # Regions start at (0,0), so anything with negative numbers
    # would indicate a bug.
    assert dirty_region[0] >= 0 and dirty_region[1] >= 0
    goal += f"(WipedSurface region_pose{dirty_region[0]}_{dirty_region[1]}) "
goal += ")"

# for row in range(NUM_ROWS):
#         for col in range(NUM_COLS):
#                 xyz_pose = params[f"region_pose{row}_{col}"].right_ee_pos.squeeze()
#                 quat = np.array([0.0, 1.0, 0.0, 0.0])
#                 print(f'("region_pose{row}_{col}", np.{repr(params["sawyer"].openrave_body.get_ik_from_pose(xyz_pose, quat, "right"))}),')
# exit()

solver = RobotSolver()
plan, descr = p_mod_abs(
    hls, solver, domain, problem, goal=goal, debug=True, n_resamples=10
)

print(plan)

if len(sys.argv) > 1 and sys.argv[1] == "end":
    sys.exit(0)

if plan is None:
    print("Could not find plan; terminating.")
    sys.exit(1)

sawyer = plan.params["sawyer"]
cmds = []
for t in range(plan.horizon):
    rgrip = sawyer.right_gripper[0, t]
    if true_mode.find("JOINT") >= 0:
        act = np.r_[sawyer.right[:, t]]
    else:
        pos, euler = sawyer.right_ee_pos[:, t], sawyer.right_ee_rot[:, t]
        quat = np.array(T.euler_to_quaternion(euler, "xyzw"))
        # angle = robosuite.utils.transform_utils.quat2axisangle(quat)
        rgrip = sawyer.right_gripper[0, t]
        act = np.r_[pos, quat]
        # act = np.r_[pos, angle, [-rgrip]]
        # act = np.r_[sawyer.right[:,t], [-rgrip]]
    cmds.append(act)

grip_ind = env.mjpy_model.site_name2id("gripper0_grip_site")
hand_ind = env.mjpy_model.body_name2id("robot0_right_hand")
env.reset()
env.sim.data.qpos[:7] = params["sawyer"].right[:, 0]
env.sim.data.qacc[:] = 0
env.sim.data.qvel[:] = 0
env.sim.forward()
rot_ref = T.euler_to_quaternion(params["sawyer"].right_ee_rot[:, 0], "xyzw")

for _ in range(40):
    env.step(np.zeros(7))
    env.sim.data.qpos[:7] = params["sawyer"].right[:, 0]  # This will help set the simulator joint sets!
    env.sim.forward()

if has_render:
    env.render()

print(dirty_regions)

nsteps = 60
cur_ind = 0
tol = 1e-3

true_lb, true_ub = plan.params["sawyer"].geom.get_joint_limits("right")
factor = (np.array(true_ub) - np.array(true_lb)) / 5
ref_jnts = env.sim.data.qpos[:7]
ref_jnts = np.array([0, -np.pi / 4, 0, np.pi / 4, 0, np.pi / 2, 0])
for act in plan.actions:
    t = act.active_timesteps[0]
    plan.params["sawyer"].right[:, t] = env.sim.data.qpos[:7]
    grip = env.sim.data.qpos[7:9].copy()
    failed_preds = plan.get_failed_preds(active_ts=(t, t), priority=3, tol=tol)
    oldqfrc = env.sim.data.qfrc_applied[:]
    oldxfrc = env.sim.data.xfrc_applied[:]
    oldacc = env.sim.data.qacc[:]
    oldvel = env.sim.data.qvel[:]
    oldwarm = env.sim.data.qacc_warmstart[:]
    oldctrl = env.sim.data.ctrl[:]
    print("FAILED:", t, failed_preds, act.name)
    old_state = env.sim.get_state()

    sawyer = plan.params["sawyer"]
    for t in range(act.active_timesteps[0], act.active_timesteps[1]):
        base_act = cmds[cur_ind]
        cur_ind += 1
        print("TIME:", t)
        init_jnts = env.sim.data.qpos[:7]
        if ctrl_mode.find("JOINT") >= 0 and true_mode.find("JOINT") < 0:
            cur_jnts = env.sim.data.qpos[:7]
            if t < plan.horizon:
                targ_pos, targ_rot = (
                    sawyer.right_ee_pos[:, t + 1],
                    sawyer.right_ee_rot[:, t + 1],
                )
            else:
                targ_pos, targ_rot = (
                    sawyer.right_ee_pos[:, t],
                    sawyer.right_ee_rot[:, t],
                )
            lb = env.sim.data.qpos[:7] - factor
            ub = env.sim.data.qpos[:7] + factor
            sawyer.openrave_body.set_dof({"right": np.zeros(7)})
            sawyer.openrave_body.set_dof({"right": ref_jnts})

            targ_jnts = sawyer.openrave_body.get_ik_from_pose(
                targ_pos, targ_rot, "right", bnds=(lb, ub)
            )
            base_act = np.r_[targ_jnts, base_act[-1]]

        true_act = base_act.copy()
        if ctrl_mode.find("JOINT") >= 0:
            targ_jnts = base_act[:7]  # + env.sim.data.qpos[:7]
            for n in range(nsteps):
                act = base_act.copy()
                act[:7] = targ_jnts - env.sim.data.qpos[:7]
                obs = env.step(act)
            end_jnts = env.sim.data.qpos[:7]

            ee_to_sim_discrepancy = (
                env.sim.data.site_xpos[grip_ind] - sawyer.right_ee_pos[:, t]
            )

            print(
                "EE PLAN VS SIM:",
                ee_to_sim_discrepancy,
                t,
            )

            # if ee_to_sim_discrepancy[2] > 0.01:
            #     from IPython import embed; embed()

            # print('\n\n\n')

        else:
            targ = base_act[3:7]
            cur = env.sim.data.body_xquat[hand_ind]
            cur = np.array([cur[1], cur[2], cur[3], cur[0]])
            truerot = Rotation.from_quat(targ)
            currot = Rotation.from_quat(cur)
            base_angle = (truerot * currot.inv()).as_rotvec()
            # base_angle = robosuite.utils.transform_utils.get_orientation_error(sign*targ, cur)
            rot = Rotation.from_rotvec(base_angle)
            targrot = (rot * currot).as_quat()
            # print('TARGETS:', targ, targrot)
            for n in range(nsteps):
                act = base_act.copy()
                act[:3] -= env.sim.data.site_xpos[grip_ind]
                # act[:3] *= 1e2
                cur = env.sim.data.body_xquat[hand_ind]
                cur = np.array([cur[1], cur[2], cur[3], cur[0]])
                # targ = act[3:7]
                sign = np.sign(targ[np.argmax(np.abs(targrot))])
                cur_sign = np.sign(targ[np.argmax(np.abs(cur))])
                targ = targrot
                # if sign != cur_sign:
                #    sign = -1.
                # else:
                #    sign = 1.
                rotmult = 1e0  # 1e1
                ##angle = 5e2*theta_error(cur, targ) #robosuite.utils.transform_utils.get_orientation_error(sign*targ, cur)
                # angle = robosuite.utils.transform_utils.get_orientation_error(sign*targ, cur)
                # rot = Rotation.from_rotvec(angle)
                # currot = Rotation.from_quat(cur)
                angle = (
                    -rotmult
                    * sign
                    * cur_sign
                    * robosuite.utils.transform_utils.get_orientation_error(
                        sign * targrot, cur_sign * cur
                    )
                )
                # a = np.linalg.norm(angle)
                # if a > 2*np.pi:
                #    angle = (a - 2*np.pi)  * angle / a
                act = np.r_[act[:3], angle, act[-1:]]
                # act[3:6] -= robosuite.utils.transform_utils.quat2axisangle(cur)
                # act[:7] = (act[:7] - np.array([env.sim.data.qpos[ind] for ind in sawyer_inds]))
                obs = env.step(act)
            print('EE PLAN VS SIM:', env.sim.data.site_xpos[grip_ind]-sawyer.right_ee_pos[:,t], t, env.reward())
        if has_render: env.render()
plan.params['sawyer'].right[:,t] = env.sim.data.qpos[:7]
