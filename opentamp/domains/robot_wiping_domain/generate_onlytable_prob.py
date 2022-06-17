# SEED = 1234
NUM_PROBS = 1
filename = "opentamp/domains/robot_wiping_domain/probs/simple_move_onlytable_prob.prob"
GOAL = "(RobotAt sawyer region_pose3_3)"


SAWYER_INIT_POSE = [-0.5, -0.1, 0.912]
SAWYER_END_POSE = [0, 0, 0]
R_ARM_INIT = [-0.3962099, -0.97739413, 0.04612799, 1.742205 , -0.03562013, 0.8089644, 0.45207395]
OPEN_GRIPPER = [0.02, -0.01]
CLOSE_GRIPPER = [0.01, -0.02]
EE_POS = [0.11338, -0.16325, 1.03655]
EE_ROT = [3.139, 0.00, -2.182]
ON_TABLE_POS = [0.4, -0.15, 0.912]

TABLE_GEOM = [0.5, 0.8, 0.9]
TABLE_POS = [0.5, 0.0, -0.040]
TABLE_ROT = [0,0,0]

num_rows = 10
num_cols = 8
xy_ontable_poses = [[] for _ in range(num_rows)]
row_step_size = TABLE_GEOM[0] / num_rows
col_step_size = TABLE_GEOM[1] / num_cols
for row in range(num_rows):
    for col in range(num_cols):
        xy_ontable_poses[row].append([TABLE_POS[0] + row * row_step_size, TABLE_POS[1] + col * col_step_size, 0.912])

def get_sawyer_pose_str(name, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = SAWYER_INIT_POSE):
    s = ""
    s += "(right {} {}), ".format(name, RArm)
    s += "(right_ee_pos {} {}), ".format(name, EE_POS)
    s += "(right_ee_rot {} {}), ".format(name, EE_ROT)
    s += "(right_gripper {} {}), ".format(name, G)
    s += "(value {} {}), ".format(name, Pos)
    s += "(rotation {} {}), ".format(name, [0.,0.,0.])
    return s

def get_sawyer_ontable_pose_str(name, ee_pos):
    # NOTE: All planning is happening in joint space.
    s = ""
    s += "(right {} {}), ".format(name, ik_joint_vals)
    s += "(right_ee_pos {} {}), ".format(name, ee_pos)
    s += "(right_ee_rot {} {}), ".format(name, EE_ROT)
    s += "(right_gripper {} undefined), ".format(name)
    s += "(value {} {}), ".format(name, robot_origin)
    s += "(rotation {} {}), ".format(name, [0.,0.,0.])
    return s


def get_sawyer_str(name, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = SAWYER_INIT_POSE):
    s = ""
    s += "(geom {})".format(name)
    s += "(right {} {}), ".format(name, RArm)
    s += "(right_ee_pos {} {}), ".format(name, EE_POS)
    s += "(right_ee_rot {} {}), ".format(name, EE_ROT)
    s += "(right_gripper {} {}), ".format(name, G)
    s += "(pose {} {}), ".format(name, Pos)
    s += "(rotation {} {}), ".format(name, [0.,0.,0.])
    return s

def get_undefined_robot_pose_str(name):
    s = ""
    s += "(right {} undefined), ".format(name)
    s += "(right_ee_pos {} undefined), ".format(name)
    s += "(right_ee_rot {} undefined), ".format(name)
    s += "(right_gripper {} undefined), ".format(name)
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    return s

def get_undefined_symbol(name):
    s = ""
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    return s

def main():
    s = "# AUTOGENERATED. DO NOT EDIT.\n# Blank lines and lines beginning with # are filtered out.\n\n"

    s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
    s += "Objects: "
    s += "Sawyer (name sawyer); "

    s += "SawyerPose (name {}); ".format("robot_init_pose")
    for row in range(num_rows):
        for col in range(num_cols):
            s += "SawyerPose (name {}); ".format(f"region_pose{row}_{col}")
    s += "Obstacle (name {}); ".format("table_obs")
    s += "Box (name {}) \n\n".format("table")

    s += "Init: "

    s += get_sawyer_str('sawyer', R_ARM_INIT, OPEN_GRIPPER, SAWYER_INIT_POSE)
    s += get_sawyer_pose_str('robot_init_pose', R_ARM_INIT, OPEN_GRIPPER, SAWYER_INIT_POSE)
    for row in range(num_rows):
        for col in range(num_cols):
            s += get_sawyer_ontable_pose_str(f"region_pose{row}_{col}", xy_ontable_poses[row][col])

    s += "(geom table {}), ".format(TABLE_GEOM)
    s += "(pose table {}), ".format(TABLE_POS)
    s += "(rotation table {}), ".format(TABLE_ROT)
    s += "(geom table_obs {}), ".format(TABLE_GEOM)
    s += "(pose table_obs {}), ".format(TABLE_POS)
    s += "(rotation table_obs {}); ".format(TABLE_ROT)

    s += "(RobotAt sawyer robot_init_pose),"
    s += "(StationaryBase sawyer), "
    s += "(IsMP sawyer), "
    s += "(WithinJointLimit sawyer), "
    s += "\n\n"

    s += "Goal: {}\n\n".format(GOAL)

    s += "Invariants: "
    s += "(StationaryBase sawyer), "
    s += "(Stationary table), "
    s += "(WithinJointLimit sawyer), "
    for row in range(num_rows):
        for col in range(num_cols):
            if row != 0:
                s += f"(PoseAdjacent region_pose{row}_{col} region_pose{row-1}_{col}), "
            if row != num_rows - 1:
                s += f"(PoseAdjacent region_pose{row}_{col} region_pose{row+1}_{col}), "
            if col != 0:
                s += f"(PoseAdjacent region_pose{row}_{col} region_pose{row}_{col-1}), "
            if col != num_cols - 1:
                s += f"(PoseAdjacent region_pose{row}_{col} region_pose{row}_{col+1}), "
    s += "\n\n"

    with open(filename, "w") as f:
        f.write(s)

if __name__ == "__main__":
    main()
