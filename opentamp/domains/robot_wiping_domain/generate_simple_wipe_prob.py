import numpy as np

# SEED = 1234
NUM_PROBS = 1
filename = "opentamp/domains/robot_wiping_domain/probs/simple_prob_sawyer.prob"
GOAL = "(RobotAt sawyer region_pose3_3)"


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

num_rows = int((TABLE_GEOM[0] * 2) / GRIPPER_SIZE[0])
num_cols = int((TABLE_GEOM[1] * 2) / GRIPPER_SIZE[1])
xy_ontable_poses = [[] for _ in range(num_rows)]
row_step_size = (TABLE_GEOM[0] * 2) / num_rows
col_step_size = (TABLE_GEOM[1] * 2) / num_cols
for row in range(num_rows):
    for col in range(num_cols):
        xy_ontable_poses[row].append([(TABLE_POS[0] - TABLE_GEOM[0]) + row * row_step_size, (TABLE_POS[1] - TABLE_GEOM[1]) + col * col_step_size, TABLE_POS[-1] + TABLE_GEOM[-1]])

# NOTE: Below 7DOF poses obtained by running the following code from a breakpoint in robo_wiping.py:
# for row in range(NUM_ROWS):
#         for col in range(NUM_COLS):
#                 xyz_pose = params[f"region_pose{row}_{col}"].right_ee_pos.squeeze()
#                 quat = np.array([0.0, 1.0, 0.0, 0.0])
#                 print(f'("region_pose{row}_{col}", np.{repr(params["sawyer"].openrave_body.get_ik_from_pose(xyz_pose, quat, "right"))}),')

# Moreover, to visualize these, we can use:
# params["sawyer"].openrave_body.set_dof({"right": params["region_pose0_1"].right[:, 0]})
# This will update the visualizer to show the arm at region_pose0_1 for
# example.

region_name_to_jnt_vals = dict([
("region_pose0_0", np.array([-0.50416095, -0.32452293, -0.36989257,  1.55344335, -2.34304996,
       -0.49812387, -2.75653215])),
("region_pose0_1", np.array([-0.31683338, -0.42009106, -0.30813161,  1.79852161, -2.19754083,
       -0.34833873, -2.75590155])),
("region_pose0_2", np.array([-0.93705249, -0.72365063,  0.47105144,  2.26221757, -1.33421935,
        0.35843596, -3.8307928 ])),
("region_pose0_3", np.array([-0.46875414, -0.72604809,  0.42593153,  2.29923856, -1.44037337,
        0.31760854, -3.28030397])),
("region_pose0_4", np.array([-0.14226102, -0.727958  ,  0.48465131,  2.26473146, -1.32268534,
        0.36753458, -3.03717773])),
("region_pose0_5", np.array([ 0.07571793, -0.70748522,  0.55903198,  2.14439332, -1.11907932,
        0.46500513, -3.00217919])),
("region_pose1_0", np.array([-0.42724979, -0.26672196, -0.4156458 ,  1.44496581, -2.33281759,
       -0.56800065, -2.66204275])),
("region_pose1_1", np.array([-0.21566751, -0.3469984 , -0.38049837,  1.66982723, -2.19492693,
       -0.44444165, -2.64390496])),
("region_pose1_2", np.array([-0.97769824, -0.71421935,  0.56863374,  2.1985044 , -1.20851078,
        0.45128779, -3.94683533])),
("region_pose1_3", np.array([-0.59270942, -0.72260182,  0.53587836,  2.23681521, -1.26724961,
        0.41382352, -3.51501104])),
("region_pose1_4", np.array([-0.26637647, -0.71628875,  0.55502447,  2.1941097 , -1.19278133,
        0.44255934, -3.26062043])),
("region_pose1_5", np.array([-0.01961426, -0.68737527,  0.59296612,  2.06945066, -1.04638157,
        0.52296054, -3.17104012])),
("region_pose2_0", np.array([-0.36539033, -0.20507472, -0.46127461,  1.32763914, -2.32341106,
       -0.63929363, -2.57014882])),
("region_pose2_1", np.array([-1.25137709, -0.65796836,  0.63942806,  1.98658735, -1.00349576,
        0.59562953, -4.44735167])),
("region_pose2_2", np.array([-0.97799226, -0.69299642,  0.61551998,  2.11117734, -1.10595835,
        0.52121572, -4.044819  ])),
("region_pose2_3", np.array([-0.64952743, -0.70433091,  0.59438471,  2.1508048 , -1.14397143,
        0.48886133, -3.67949208])),
("region_pose2_4", np.array([-0.34336194, -0.69509825,  0.59888098,  2.10706151, -1.09103831,
        0.51057185, -3.43581728])),
("region_pose2_5", np.array([-0.08887657, -0.6614019 ,  0.61538027,  1.98160053, -0.98108096,
        0.58072228, -3.32092925])),
("region_pose3_0", np.array([-0.31618282, -0.13229123, -0.51750806,  1.19519169, -2.29981839,
       -0.71706163, -2.48734574])),
("region_pose3_1", np.array([-1.19716607, -0.62667469,  0.65143504,  1.88587069, -0.94359824,
        0.65313774, -4.48317456])),
("region_pose3_2", np.array([-0.95961006, -0.66462869,  0.64011799,  2.00992881, -1.0211549 ,
        0.585191  , -4.12958405])),
("region_pose3_3", np.array([-0.67260471, -0.67745295,  0.62672215,  2.05004014, -1.04750409,
        0.55673291, -3.81021176])),
("region_pose3_4", np.array([-0.38896189, -0.66671691,  0.6246497 ,  2.00673054, -1.00766487,
        0.57541141, -3.58143996])),
("region_pose3_5", np.array([-0.13652397, -0.62996863,  0.62826075,  1.88162574, -0.92231723,
        0.63901945, -3.45554366])),
("region_pose4_0", np.array([-1.38649715, -0.4839197 ,  0.84548524,  1.59661008, -0.97688596,
        0.92796406, -4.712499  ])),
("region_pose4_1", np.array([-1.13842296, -0.59047148,  0.65304538,  1.77302111, -0.88742632,
        0.71015608, -4.52342731])),
("region_pose4_2", np.array([-0.9292261 , -0.63039773,  0.65067736,  1.89685945, -0.94883558,
        0.64726777, -4.20621233])),
("region_pose4_3", np.array([-0.67434898, -0.6440184 ,  0.64261584,  1.93716788, -0.96819497,
        0.62184769, -3.92108622])),
("region_pose4_4", np.array([-0.41194801, -0.63225802,  0.63749547,  1.89446846, -0.93707805,
        0.63913954, -3.707449  ])),
("region_pose4_5", np.array([-0.16619358, -0.59318606,  0.63344044,  1.76966636, -0.86904146,
        0.69864014, -3.57813633])),
("region_pose5_0", np.array([-1.33809373, -0.42893873,  0.89481988,  1.47988782, -0.99255241,
        1.01267961, -4.712499  ])),
("region_pose5_1", np.array([-1.076321  , -0.54887387,  0.64791005,  1.64773354, -0.83511836,
        0.76909349, -4.56670663])),
("region_pose5_2", np.array([-0.89005332, -0.59062399,  0.65151456,  1.77223532, -0.88544816,
        0.70954689, -4.27789188])),
("region_pose5_3", np.array([-0.66098499, -0.6047564 ,  0.64725035,  1.81280371, -0.9005355 ,
        0.68645346, -4.01944984])),
("region_pose5_4", np.array([-0.41775451, -0.5920921 ,  0.64091842,  1.77045358, -0.87565551,
        0.70328174, -3.81999558])),
("region_pose5_5", np.array([-0.18077316, -0.55080266,  0.63261277,  1.64513163, -0.82033619,
        0.76060009, -3.69144375])),
("region_pose6_0", np.array([-1.29955938, -0.36306683,  0.9705087 ,  1.35658721, -1.03679667,
        1.11334592, -4.712499  ])),
("region_pose6_1", np.array([-1.0111532 , -0.50102334,  0.6380999 ,  1.50847349, -0.78630148,
        0.83175959, -4.61277662])),
("region_pose6_2", np.array([-0.8439083 , -0.5449857 ,  0.64549235,  1.63517048, -0.82866948,
        0.77380116, -4.34672265])),
("region_pose6_3", np.array([-0.63612955, -0.55961395,  0.64399264,  1.67633408, -0.8411371 ,
        0.75235874, -4.10972177])),
("region_pose6_4", np.array([-0.40996508, -0.54597788,  0.63754826,  1.63383989, -0.82104379,
        0.76939246, -3.9231932 ])),
("region_pose6_5", np.array([-0.18251008, -0.50210823,  0.62744922,  1.50650895, -0.7756367 ,
        0.82632248, -3.79776138])),
("region_pose7_0", np.array([-1.27160219, -0.28559576,  1.08140429,  1.23342202, -1.11875249,
        1.22978381, -4.712499  ])),
("region_pose7_1", np.array([-0.94300725, -0.4454386 ,  0.62565972,  1.35232903, -0.74111228,
        0.90047275, -4.66172115])),
("region_pose7_2", np.array([-0.79188635, -0.49252308,  0.6349223 ,  1.48358257, -0.77718919,
        0.84206818, -4.41421334])),
("region_pose7_3", np.array([-0.60204072, -0.5078509 ,  0.63541876,  1.52595838, -0.78801909,
        0.82155368, -4.19483388])),
("region_pose7_4", np.array([-0.39101672, -0.49303588,  0.62965235,  1.4826096 , -0.77192072,
        0.83940706, -4.0199941 ])),
("region_pose7_5", np.array([-0.17313846, -0.44576239,  0.61993352,  1.35097365, -0.73505251,
        0.89794018, -3.89907332])),
("region_pose8_0", np.array([-1.24507678, -0.20780531,  1.20976662,  1.12228433, -1.2246496 ,
        1.34276709, -4.712499  ])),
("region_pose8_1", np.array([-0.8723803 , -0.37982156,  0.6147665 ,  1.17488334, -0.70226774,
        0.9790557 , -4.712499  ])),
("region_pose8_2", np.array([-0.73467177, -0.43145683,  0.62231321,  1.31372365, -0.73087577,
        0.91718015, -4.48159754])),
("region_pose8_3", np.array([-0.56018006, -0.44792297,  0.62405713,  1.3583157 , -0.74049556,
        0.89669983, -4.27694592])),
("region_pose8_4", np.array([-0.36261045, -0.4315657 ,  0.61978364,  1.31308787, -0.72817674,
        0.91610269, -4.11270298])),
("region_pose8_5", np.array([-0.15408767, -0.37947337,  0.61334222,  1.17360596, -0.70026478,
        0.97895051, -3.99719772])),
("region_pose9_0", np.array([-1.2039696 , -0.15263122,  1.30634228,  1.02452144, -1.30863885,
        1.42084602, -4.712499  ])),
("region_pose9_1", np.array([-0.85882648, -0.29932351,  0.73891992,  0.99872262, -0.79724526,
        1.12001436, -4.712499  ])),
("region_pose9_2", np.array([-0.67287135, -0.3587622 ,  0.61198045,  1.11909728, -0.69199947,
        1.00376587, -4.5500446 ])),
("region_pose9_3", np.array([-0.51162566, -0.37712109,  0.6137244 ,  1.16755047, -0.69998805,
        0.98198655, -4.35786035])),
("region_pose9_4", np.array([-0.32613291, -0.35863304,  0.61238831,  1.11887589, -0.69217462,
        1.00407264, -4.20331364])),
("region_pose9_5", np.array([-0.12714268, -0.29951709,  0.61576152,  0.96596988, -0.67788757,
        1.07584384, -4.09391802]))
])

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
    s = ""
    s += "(right {} {}), ".format(name, list(region_name_to_jnt_vals[name]))
    s += "(right_ee_pos {} {}), ".format(name, ee_pos)
    s += "(right_ee_rot {} {}), ".format(name, EE_ROT)
    s += "(right_gripper {} undefined), ".format(name)
    s += "(value {} {}), ".format(name, SAWYER_INIT_POSE)
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
