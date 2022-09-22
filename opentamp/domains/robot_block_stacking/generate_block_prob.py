import opentamp


N_BLOCKS = 3
FILENAME = opentamp.__path__._path[0] + "/domains/robot_block_stacking/probs/stack_{}_blocks.prob".format(N_BLOCKS)


PANDA_INIT_POSE = [0., 0.1, 0.55]
PANDA_INIT_ROT = [0, 0, 1.57]
PANDA_END_POSE = [0., 0.1, 0.55]
R_ARM_INIT = [-0.30, -0.4, 0.28, -2.5, 0.13, 1.87, 0.91]
OPEN_GRIPPER = [0.04, 0.04]
CLOSE_GRIPPER = [0., 0.]
EE_POS = [0.11338, -0.16325, 1.03655]
EE_ROT = [3.139, 0.00, -2.182]
BLOCK_DIM = [0.01, 0.01, 0.01]


def get_panda_pose_str(name, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = PANDA_INIT_POSE, Rot = PANDA_INIT_ROT):
    s = ""
    s += "(right {} {}), ".format(name, RArm)
    s += "(right_ee_pos {} {}), ".format(name, EE_POS)
    s += "(right_ee_rot {} {}), ".format(name, EE_ROT)
    s += "(right_gripper {} {}), ".format(name, G)
    s += "(value {} {}), ".format(name, Pos)
    s += "(rotation {} {}), ".format(name, Rot)
    return s

def get_panda_str(name, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = PANDA_INIT_POSE, Rot = PANDA_INIT_ROT):
    s = ""
    s += "(geom {})".format(name)
    s += "(right {} {}), ".format(name, RArm)
    s += "(right_ee_pos {} {}), ".format(name, EE_POS)
    s += "(right_ee_rot {} {}), ".format(name, EE_ROT)
    s += "(right_gripper {} {}), ".format(name, G)
    s += "(pose {} {}), ".format(name, Pos)
    s += "(rotation {} {}), ".format(name, Rot)
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
    s = "# AUTOGENERATED. DO NOT EDIT.\n# Configuration file for CAN problem instance. Blank lines and lines beginning with # are filtered out.\n\n"

    s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
    s += "Objects: "
    s += "Panda (name panda); "

    for n in range(N_BLOCKS):
        s += "Box (name block{}); ".format(n)
        s += "Target (name block{}_target); ".format(n)

    s += "PandaPose (name {}); ".format("robot_init_pose")
    s += "PandaPose (name {}); ".format("robot_end_pose")
    s += "\n\n"

    s += "Init: "

    for n in range(N_BLOCKS):
        s += "(geom block{0} {1}), ".format(n, BLOCK_DIM)
        s += "(pose block{} [{}, 0.3, 0.01]), ".format(n, n/25.)
        s += "(rotation block{} [0.0, 0.0, 0.0]), ".format(n)

        s += "(value block{}_target [{}, 0.3, 0.01]), ".format(n, n/25.)
        s += "(rotation block{}_target [0.0, 0.0, 0.0]), ".format(n)

    s += get_panda_str('panda', R_ARM_INIT, OPEN_GRIPPER, PANDA_INIT_POSE)
    s += get_panda_pose_str('robot_init_pose', R_ARM_INIT, OPEN_GRIPPER, PANDA_INIT_POSE)
    s += get_undefined_robot_pose_str('robot_end_pose')[:-2]
    s += ";"

    for n in range(N_BLOCKS):
        s += "(InReach block{0} panda), ".format(n)
        for m in range(n):
            s += "(Stackable block{} block{}), ".format(m, n)
            s += "(Stackable block{} block{}), ".format(n, m)

    s += "(StationaryBase panda), "
    s += "(IsMP panda), "
    s += "(WithinJointLimit panda)"
    s += "\n\n"

    s += "Goal: (Stacked block1 block2), (Stacked block0 block1)\n\n"
    
    s += "Invariants: "
    s += "(StationaryBase panda), "
    s += "\n\n"

    with open(FILENAME, "w") as f:
        f.write(s)

if __name__ == "__main__":
    main()
