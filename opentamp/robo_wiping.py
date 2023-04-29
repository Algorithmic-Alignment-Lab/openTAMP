import os
import sys

import numpy as np
import pybullet as P
import robosuite
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation
from robosuite.utils.mjmod import DynamicsModder, CameraModder, LightingModder 
from robosuite.wrappers import DomainRandomizationWrapper
import opentamp.core.util_classes.transform_utils as T
import main
from opentamp.core.parsing import parse_domain_config, parse_problem_config
from opentamp.core.util_classes.openrave_body import *
from opentamp.core.util_classes.transform_utils import *
from pma.hl_solver import *
from pma.pr_graph import *
from pma.robosuite_solver import RobotSolverOSQP
from sco_py.expr import *

import pdb

from PIL import Image

# Constants
GRIPPER_SIZE = [0.05, 0.12, 0.015]
TABLE_GEOM = [0.25, 0.40, 0.820]
TABLE_POS = [0.15, 0.00, 0.00]
NUM_ROWS = int((TABLE_GEOM[0] * 2) / GRIPPER_SIZE[0])
NUM_COLS = int((TABLE_GEOM[1] * 2) / GRIPPER_SIZE[1])
REF_QUAT = np.array([0, 0, -0.7071, -0.7071])


def plan_wiping():
    ctrl_mode = "JOINT_POSITION"
    true_mode = "JOINT"
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

    # Set visualization variable.
    visual = len(os.environ.get("DISPLAY", "")) > 0
    has_render = visual
    print(has_render)
    # Create underlying robosuite environment. This is ultimately the environment
    # that we will execute plans in.
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
        hard_reset = False,
    )
    env = DomainRandomizationWrapper(env,randomize_every_n_steps=0, randomize_on_reset=True, randomize_dynamics=False)
    print(env.num_markers)
    obs, _, _, _ = env.step(np.zeros(7)) # Step a null action to 'boot' the environment.
    #insert calls to the modder
    #modder = CameraModder(sim=env.env.sim, random_state=np.random.RandomState(5))
    #modder = DynamicsModder(sim=env.env.sim, random_state=np.random.RandomState(5))

    # Define function for easy printing
    #import pdb; pdb.set_trace()
    #cube_body_id = env.sim.model.body_name2id(env.cube.root_body)
    #cube_geom_ids = [env.sim.model.geom_name2id(geom) for geom in env.cube.contact_geoms]
    #modder.randomize()
    #position = modder.get_pos('agentview')
    #position = modder.get_pos('robot0_eye_in_hand')

    #modder.set_pos('frontview', np.array([100, 200, -200]))


    #print(position)
    #def print_params():
    #   print(f"cube mass: {env.sim.model.body_mass[cube_body_id]}")
    #  print(f"cube frictions: {env.sim.model.geom_friction[cube_geom_ids]}")
    # print()

    # Print out initial parameter values
    #print("INITIAL VALUES")
    #print_params()

    # Modify the cube's properties
    #modder.mod(env.cube.root_body, "mass", 5.0)                                # make the cube really heavy
    #for geom_name in env.cube.contact_geoms:
    #   modder.mod(geom_name, "friction", [2.0, 0.2, 0.04])           # greatly increase the friction
    #modder.update()                                                   # make sure the changes propagate in sim

    # Print out modified parameter values
    #print("MODIFIED VALUES")
    #print_params()

    # We can also restore defaults (original values) at any time
    #modder.restore_defaults()

    # Print out restored initial parameter values
    #print("RESTORED VALUES")
    #print_params()
    # Before commending planning, we reset the environment and then manually set the
    # joint positions to their initial positions and all the joint velocities and
    # accelerations to 0.
    obs = env.reset()
    jnts = env.env.sim.data.qpos[:7]
    for _ in range(40):
        env.step(np.zeros(7))
        env.env.sim.data.qpos[:7] = jnts
        env.env.sim.forward()
    env.env.sim.data.qvel[:] = 0
    env.env.sim.data.qacc[:] = 0
    env.env.sim.forward()

    # Now, we load the domain and problem files, and also instantiate the
    # solvers.
    domain_fname = os.getcwd() + "/opentamp/domains/robot_wiping_domain/right_wipe_onlytable.domain"
    prob = os.getcwd() + "/opentamp/domains/robot_wiping_domain/probs/simple_move_onlytable_prob.prob"
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = FFSolver(d_c)
    p_c = main.parse_file_to_dict(prob)
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=visual)
    params = problem.init_state.params
    # We will use the robot body and table later.
    # pdb.set_trace()
    body_ind = env.env.mjpy_model.body_name2id("robot0_base")
    table_ind = env.env.mjpy_model.body_name2id("table")

    # Get the locations of all dirt particles.
    # NOTE: Important that this is done after the env.reset() call
    # because this call randomizes all dirt positions.
    dirt_locs = np.zeros((env.num_markers, 3))
    for i, marker in enumerate(env.model.mujoco_arena.markers):
        marker_pos = np.array(env.env.sim.data.body_xpos[env.env.sim.model.body_name2id(marker.root_body)])
        dirt_locs[i,:] = marker_pos

    # Computes the dirty regions set, which contains a tuple (row, col) for every
    # region that is dirty. 
    dirty_regions = set()
    row_step_size = (TABLE_GEOM[0] * 2) / NUM_ROWS
    col_step_size = (TABLE_GEOM[1] * 2) / NUM_COLS
    for xyz_pose in dirt_locs.tolist():
        x_pos = xyz_pose[0]
        y_pos = xyz_pose[1]
        # Compute the distances from all pre-defined regions.
        distance_dict = {}
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                region_xyz_pose = params[f"region_pose{row}_{col}"].right_ee_pos.squeeze()
                xy_pose = region_xyz_pose[:-1]
                dist_to_rowcol = np.linalg.norm(xy_pose - np.array([x_pos, y_pos]))
                distance_dict[(row, col)] = dist_to_rowcol
        top_regions = [k for k, _ in sorted(distance_dict.items(), key=lambda item: item[1])]
        # Add the two closest regions to the dirty-regions list. This is done to be maximally
        # covering of the dirt and not allow some dirt spots to slip through the cracks.
        dirty_regions.add(top_regions[0])
        dirty_regions.add(top_regions[1])


    # Resetting the initial state of the robot in our internal representation
    # to match the robotsuite sim.
    params["sawyer"].pose[:, 0] = env.env.sim.data.body_xpos[body_ind]
    # NOTE: for the table, we only want to set the (x,y) poses to
    # be equal, because we use a different geometry and thus the
    # height must be different.
    params["table"].pose[:2, 0] = env.env.sim.data.body_xpos[table_ind][:2]
    jnts = params["sawyer"].geom.jnt_names["right"]
    jnts = ["robot0_" + jnt for jnt in jnts]
    jnt_vals = []
    sawyer_inds = []
    for jnt in jnts:
        jnt_adr = env.env.mjpy_model.joint_name2id(jnt)
        jnt_ind = env.env.mjpy_model.jnt_qposadr[jnt_adr]
        sawyer_inds.append(jnt_ind)
        jnt_vals.append(env.env.sim.data.qpos[jnt_ind])
    params["sawyer"].right[:, 0] = jnt_vals
    params["robot_init_pose"].right[:, 0] = jnt_vals
    params["robot_init_pose"].value[:, 0] = params["sawyer"].pose[:, 0]
    params["sawyer"].openrave_body.set_pose(params["sawyer"].pose[:, 0])
    params["sawyer"].openrave_body.set_dof({"right": params["sawyer"].right[:, 0]})
    info = params["sawyer"].openrave_body.fwd_kinematics("right")
    params["sawyer"].right_ee_pos[:, 0] = info["pos"]
    params["sawyer"].right_ee_pos[:, 0] = T.quaternion_to_euler(info["quat"], "xyzw")
    # Dynamically set goal to be the set of all dirty regions.
    goal = "(and"
    for dirty_region in dirty_regions:
        # Regions start at (0,0), so anything with negative numbers
        # would indicate a bug.
        assert dirty_region[0] >= 0 and dirty_region[1] >= 0
        goal += f"(WipedSurface region_pose{dirty_region[0]}_{dirty_region[1]}) "
    goal += ")"

    # NOTE: Run the below code to generate the region pose numbers that need to be
    # placed into the generate_onlytable_prob.py file.

    # for row in range(NUM_ROWS):
    #         for col in range(NUM_COLS):
    #                 xyz_pose = params[f"region_pose{row}_{col}"].right_ee_pos.squeeze()
    #                 quat = np.array([0.0, 1.0, 0.0, 0.0])
    #                 print(f'("region_pose{row}_{col}", np.{repr(params["sawyer"].openrave_body.get_ik_from_pose(xyz_pose, quat, "right"))}),')
    # exit()

    # Instantiate the solver.
    solver = RobotSolverOSQP()
    # Run planning to obtain a final plan.
    plan, descr = p_mod_abs(
        hls, solver, domain, problem, goal=goal, debug=True, n_resamples=10
    )

    if len(sys.argv) > 1 and sys.argv[1] == "end":
        sys.exit(0)

    if plan is None:
        print("Could not find plan; terminating.")
        sys.exit(1)

    # Create a list of the commands from the plan that we want to
    # execute in the real simulation.
    sawyer = plan.params["sawyer"]
    cmds = []
    for t in range(plan.horizon):
        rgrip = sawyer.right_gripper[0, t]
        if true_mode.find("JOINT") >= 0:
            act = np.r_[sawyer.right[:, t]]
        else:
            pos, euler = sawyer.right_ee_pos[:, t], sawyer.right_ee_rot[:, t]
            quat = np.array(T.euler_to_quaternion(euler, "xyzw"))
            rgrip = sawyer.right_gripper[0, t]
            act = np.r_[pos, quat]
        cmds.append(act)

    grip_ind = env.env.mjpy_model.site_name2id("gripper0_grip_site")
    hand_ind = env.env.mjpy_model.body_name2id("robot0_right_hand")
    env.env.sim.data.qpos[:7] = params["sawyer"].right[:, 0]
    env.env.sim.data.qacc[:] = 0
    env.env.sim.data.qvel[:] = 0
    env.env.sim.forward()
    rot_ref = T.euler_to_quaternion(params["sawyer"].right_ee_rot[:, 0], "xyzw")

    for _ in range(40):
        env.step(np.zeros(7))
        env.env.sim.data.qpos[:7] = params["sawyer"].right[:, 0]  # This will help set the simulator joint sets!
        env.env.sim.forward()

    if has_render:
        env.render()

    nsteps = 60
    cur_ind = 0
    tol = 1e-3

    # for visualization
    gif_frames = []
    render_interval = 10
    render_t =0
    #modder.set_pos('frontview', np.array([100., 200., -200.]))
    env.env.sim.forward()
    # Loop to execute the plan's actions in the simulation.
    true_lb, true_ub = plan.params["sawyer"].geom.get_joint_limits("right")
    factor = (np.array(true_ub) - np.array(true_lb)) / 5
    ref_jnts = env.env.sim.data.qpos[:7]
    ref_jnts = np.array([0, -np.pi / 4, 0, np.pi / 4, 0, np.pi / 2, 0])
    for act in plan.actions:
        t = act.active_timesteps[0]
        plan.params["sawyer"].right[:, t] = env.env.sim.data.qpos[:7]
        grip = env.env.sim.data.qpos[7:9].copy()
        failed_preds = plan.get_failed_preds(active_ts=(t, t), priority=3, tol=tol)
        oldqfrc = env.env.sim.data.qfrc_applied[:]
        oldxfrc = env.env.sim.data.xfrc_applied[:]
        oldacc = env.env.sim.data.qacc[:]
        oldvel = env.env.sim.data.qvel[:]
        oldwarm = env.env.sim.data.qacc_warmstart[:]
        oldctrl = env.env.sim.data.ctrl[:]
        print("FAILED:", t, failed_preds, act.name)
        old_state = env.env.sim.get_state()

        sawyer = plan.params["sawyer"]
        for t in range(act.active_timesteps[0], act.active_timesteps[1]):
            base_act = cmds[cur_ind]
            cur_ind += 1
            print("TIME:", t)
            init_jnts = env.env.sim.data.qpos[:7]
            if ctrl_mode.find("JOINT") >= 0 and true_mode.find("JOINT") < 0:
                cur_jnts = env.env.sim.data.qpos[:7]
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
                lb = env.env.sim.data.qpos[:7] - factor
                ub = env.env.sim.data.qpos[:7] + factor
                sawyer.openrave_body.set_dof({"right": np.zeros(7)})
                sawyer.openrave_body.set_dof({"right": ref_jnts})

                targ_jnts = sawyer.openrave_body.get_ik_from_pose(
                    targ_pos, targ_rot, "right", bnds=(lb, ub)
                )
                base_act = np.r_[targ_jnts, base_act[-1]]

            true_act = base_act.copy()
            if ctrl_mode.find("JOINT") >= 0:
                targ_jnts = base_act[:7]  # + env.env.sim.data.qpos[:7]
                for n in range(nsteps):
                    act = base_act.copy()
                    act[:7] = targ_jnts - env.env.sim.data.qpos[:7]
                    obs = env.step(act)
                    print(obs)
                    if render_t == 0:
                        gif_frames.append(
                                Image.fromarray(
                                    env.env.sim.render(height=192, width=192, camera_name="frontview")
                                )
                        )

                    render_t = (render_t+1) % render_interval

                    # pdb.set_trace()
                end_jnts = env.env.sim.data.qpos[:7]

                ee_to_sim_discrepancy = (
                    env.env.sim.data.site_xpos[grip_ind] - sawyer.right_ee_pos[:, t]
                )

                print(
                    "EE PLAN VS SIM:",
                    ee_to_sim_discrepancy,
                    t,
                )

            else:
                targ = base_act[3:7]
                cur = env.env.sim.data.body_xquat[hand_ind]
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
                    act[:3] -= env.env.sim.data.site_xpos[grip_ind]
                    # act[:3] *= 1e2
                    cur = env.env.sim.data.body_xquat[hand_ind]
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
                    act = np.r_[act[:3], angle, act[-1:]]
                    obs = env.step(act)
                    #print(obs)
            # pdb.set_trace()
            if has_render: env.render()
            # pdb.set_trace()
    plan.params['sawyer'].right[:,t] = env.env.sim.data.qpos[:7]

    # Print out whether the task was successfully completed or not.
    if len(env.wiped_markers) == env.num_markers:
        print("Task Completed Successfully!")
    else:
        print(f"Task Failed: Num Missed Markers: {env.num_markers - len(env.wiped_markers)}")

    gif_frames[0].save("render_planner6.gif",
            save_all=True,
            append_images=gif_frames[1:],
            duration=50,
            loop=0)
    return env.num_markers - len(env.wiped_markers)

difference = []
for i in range(5):
   difference.append(plan_wiping())
print(np.mean(np.array(difference)))
