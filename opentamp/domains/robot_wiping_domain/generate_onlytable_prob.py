import numpy as np

# SEED = 1234
NUM_PROBS = 1
filename = "opentamp/domains/robot_wiping_domain/probs/simple_move_onlytable_prob.prob"
GOAL = "(RobotAt sawyer region_pose3_3)"


SAWYER_INIT_POSE = [-0.41, 0.0, 0.912]
SAWYER_END_POSE = [0, 0, 0]
R_ARM_INIT = [-0.3962099, -0.97739413, 0.04612799, 1.742205 , -0.03562013, 0.8089644, 0.45207395]
OPEN_GRIPPER = [0.02, -0.01]
CLOSE_GRIPPER = [0.01, -0.02]
EE_POS = [0.11338, -0.16325, 1.03655]
EE_ROT = [3.139, 0.00, -2.182]

TABLE_GEOM = [0.25, 0.40, 0.875 + 0.00506513]
TABLE_POS = [0.15, 0.00, 0.00]
TABLE_ROT = [0,0,0]

num_rows = 10
num_cols = 8
xy_ontable_poses = [[] for _ in range(num_rows)]
row_step_size = (TABLE_GEOM[0] * 2) / num_rows
col_step_size = (TABLE_GEOM[1] * 2) / num_cols
for row in range(num_rows):
    for col in range(num_cols):
        xy_ontable_poses[row].append([(TABLE_POS[0] - TABLE_GEOM[0]) + row * row_step_size, (TABLE_POS[1] - TABLE_GEOM[1]) + col * col_step_size, TABLE_POS[-1] + TABLE_GEOM[-1]])

# NOTE: Below 7DOF poses obtained by running the following code from a breakpoint in robo_wiping.py:
# for row in range(10):
#         for col in range(8):
#             xyz_pose = params[f"region_pose{row}_{col}"].right_ee_pos.squeeze()
#             quat = np.array([0.0, 1.0, 0.0, 0.0])
#             print(f'("region_pose{row}_{col}", np.{repr(params["sawyer"].openrave_body.get_ik_from_pose(xyz_pose, quat, "right"))}),')

# Moreover, to visualize these, we can use:
# params["sawyer"].openrave_body.set_dof({"right": params["region_pose0_1"].right[:, 0]})
# This will update the visualizer to show the arm at region_pose0_1 for
# example.

region_name_to_jnt_vals = dict([
("region_pose0_0", np.array([-0.46956788, -0.43070158, -0.42120413,  1.5965524 , -2.38756966,
       -0.57306972, -2.71801806])),
("region_pose0_1", np.array([-0.30895724, -0.50047432, -0.40103992,  1.78155142, -2.28987182,
       -0.47204354, -2.6992443 ])),
("region_pose0_2", np.array([-0.11540553, -0.54719428, -0.38452454,  1.92015038, -2.16706197,
       -0.3970493 , -2.65490554])),
("region_pose0_3", np.array([-1.01341581, -0.92754568,  0.73300796,  2.45030296, -1.1813145 ,
        0.45006267, -3.78374239])),
("region_pose0_4", np.array([-0.67526   , -0.93665312,  0.70361356,  2.47523097, -1.21769416,
        0.42189813, -3.42649631])),
("region_pose0_5", np.array([-0.37848835, -0.93204737,  0.68952984,  2.43940974, -1.15774553,
        0.42780828, -3.20942561])),
("region_pose0_6", np.array([-0.14958386, -0.91104917,  0.69570114,  2.35204449, -1.04334809,
        0.47265497, -3.1124752 ])),
("region_pose0_7", np.array([ 0.0323777 , -0.86944801,  0.7033463 ,  2.21741515, -0.92812458,
        0.54902678, -3.07982701])),
("region_pose1_0", np.array([-0.39834459, -0.36916827, -0.46233627,  1.48735806, -2.37112592,
       -0.63962148, -2.63331449])),
("region_pose1_1", np.array([-0.22648322, -0.42716099, -0.45726842,  1.65816705, -2.27431885,
       -0.55454612, -2.60953128])),
("region_pose1_2", np.array([-1.26985154, -0.87354742,  0.77245937,  2.27272487, -1.002211  ,
        0.5623345 , -4.24186582])),
("region_pose1_3", np.array([-1.01993418, -0.90067539,  0.76349313,  2.35534425, -1.068683  ,
        0.51337364, -3.90555169])),
("region_pose1_4", np.array([-0.73542209, -0.91122701,  0.7435857 ,  2.38043903, -1.08890859,
        0.48830509, -3.60843424])),
("region_pose1_5", np.array([-0.465928  , -0.9053277 ,  0.72828222,  2.34836124, -1.04944739,
        0.49468382, -3.39912894])),
("region_pose1_6", np.array([-0.23657514, -0.88155966,  0.72142763,  2.2641247 , -0.9708184 ,
        0.53470988, -3.27867526])),
("region_pose1_7", np.array([-0.04307578, -0.83800075,  0.71461618,  2.1321528 , -0.88414026,
        0.60335509, -3.22015876])),
("region_pose2_0", np.array([-0.34078406, -0.3021976 , -0.5059118 ,  1.36803993, -2.3522776 ,
       -0.70944224, -2.5523745 ])),
("region_pose2_1", np.array([-1.37907724, -0.79385851,  0.75905389,  2.04062586, -0.87445772,
        0.6819718 , -4.57332955])),
("region_pose2_2", np.array([-1.21934374, -0.83869841,  0.77457322,  2.17022862, -0.9384103 ,
        0.61982581, -4.28978786])),
("region_pose2_3", np.array([-1.00439041, -0.86686736,  0.7718927 ,  2.24939888, -0.98351421,
        0.57456526, -4.00318785])),
("region_pose2_4", np.array([-0.75882439, -0.8778273 ,  0.75903745,  2.27442449, -0.99533396,
        0.55281494, -3.74638451])),
("region_pose2_5", np.array([-0.51500592, -0.87120198,  0.74459142,  2.24496595, -0.96724938,
        0.55962002, -3.55048452])),
("region_pose2_6", np.array([-0.29358493, -0.84612694,  0.73159595,  2.16403225, -0.91049783,
        0.5961102 , -3.42211717])),
("region_pose2_7", np.array([-0.09731834, -0.8017102 ,  0.71674002,  2.03525929, -0.84359838,
        0.65880457, -3.34897153])),
("region_pose3_0", np.array([-1.49870311, -0.65016974,  0.90720788,  1.78170023, -0.9106615 ,
        0.91930137, -4.712499  ])),
("region_pose3_1", np.array([-1.30573494, -0.75419839,  0.74800866,  1.93166762, -0.83376678,
        0.73525669, -4.59199607])),
("region_pose3_2", np.array([-1.16299887, -0.7994333 ,  0.76484582,  2.0575853 , -0.88313985,
        0.67628805, -4.33823979])),
("region_pose3_3", np.array([-0.97509732, -0.82768025,  0.76719532,  2.13414268, -0.91580253,
        0.63518078, -4.0863065 ])),
("region_pose3_4", np.array([-0.75919617, -0.83851582,  0.75922722,  2.15884548, -0.92314815,
        0.61637715, -3.85891918])),
("region_pose3_5", np.array([-0.53778013, -0.83139329,  0.7461559 ,  2.13123576, -0.90204634,
        0.62349382, -3.6771096 ])),
("region_pose3_6", np.array([-0.32734735, -0.80568951,  0.73053544,  2.05319154, -0.8592182 ,
        0.65729486, -3.54839797])),
("region_pose3_7", np.array([-0.13337459, -0.76086073,  0.71165703,  1.92730729, -0.80615991,
        0.71555128, -3.46776252])),
("region_pose4_0", np.array([-1.42943851, -0.60088758,  0.91692826,  1.66726016, -0.90714215,
        0.98335266, -4.712499  ])),
("region_pose4_1", np.array([-1.23067579, -0.71042574,  0.72981616,  1.8116728 , -0.79419071,
        0.78921976, -4.61788482])),
("region_pose4_2", np.array([-1.10269429, -0.75582648,  0.74809521,  1.93529047, -0.83426746,
        0.73337882, -4.38679838])),
("region_pose4_3", np.array([-0.93613499, -0.78386418,  0.75377277,  2.00992475, -0.85936302,
        0.69579361, -4.16079562])),
("region_pose4_4", np.array([-0.74363443, -0.79435968,  0.74920059,  2.03425169, -0.86437819,
        0.6794429 , -3.95601754])),
("region_pose4_5", np.array([-0.54149627, -0.78681328,  0.73770152,  2.00793601, -0.84793811,
        0.68690419, -3.78739965])),
("region_pose4_6", np.array([-0.34287541, -0.76069014,  0.72147552,  1.93210655, -0.81440866,
        0.71884959, -3.66192744])),
("region_pose4_7", np.array([-0.15399206, -0.71543544,  0.70113938,  1.80825742, -0.77132399,
        0.77409371, -3.57823609])),
("region_pose5_0", np.array([-1.36898662, -0.54282687,  0.94444698,  1.54379299, -0.92075919,
        1.06050637, -4.712499  ])),
("region_pose5_1", np.array([-1.15470171, -0.6617215 ,  0.70788809,  1.67999934, -0.75637487,
        0.84593704, -4.6488076 ])),
("region_pose5_2", np.array([-1.03926365, -0.70763199,  0.726806  ,  1.80281746, -0.79004031,
        0.79219414, -4.43589261])),
("region_pose5_3", np.array([-0.8899645 , -0.73555213,  0.73454625,  1.87634828, -0.81037598,
        0.75724379, -4.23014019])),
("region_pose5_4", np.array([-0.71643235, -0.74569162,  0.73238771,  1.90041548, -0.81426021,
        0.74283779, -4.04338239])),
("region_pose5_5", np.array([-0.53089303, -0.73770988,  0.72253257,  1.87493011, -0.80119716,
        0.75075391, -3.88664529])),
("region_pose5_6", np.array([-0.34389049, -0.71108518,  0.7068941 ,  1.80045475, -0.77425163,
        0.78167182, -3.76604161])),
("region_pose5_7", np.array([-0.16149168, -0.66505669,  0.68676962,  1.67735705, -0.73869709,
        0.83529589, -3.68206316])),
("region_pose6_0", np.array([-1.31848215, -0.4744665 ,  0.99727036,  1.41237834, -0.95867414,
        1.15321542, -4.712499  ])),
("region_pose6_1", np.array([-1.07781958, -0.60706576,  0.6839872 ,  1.53490858, -0.72043364,
        0.90703596, -4.68404402])),
("region_pose6_2", np.array([-0.97317572, -0.65418368,  0.70275757,  1.65880308, -0.74935616,
        0.85404708, -4.48598925])),
("region_pose6_3", np.array([-0.83816937, -0.68233204,  0.71167137,  1.7322545 , -0.76656475,
        0.82075712, -4.29651864])),
("region_pose6_4", np.array([-0.68035803, -0.69222871,  0.71127615,  1.75631221, -0.7699937 ,
        0.80780079, -4.12451122])),
("region_pose6_5", np.array([-0.50916984, -0.68375069,  0.70307353,  1.73121547, -0.75957276,
        0.81631241, -3.97835554])),
("region_pose6_6", np.array([-0.33311698, -0.65634335,  0.68875867,  1.65704068, -0.73762298,
        0.84703007, -3.8633099 ])),
("region_pose6_7", np.array([-0.15773559, -0.6089028 ,  0.67005488,  1.53294776, -0.7081751 ,
        0.90048867, -3.78082522])),
("region_pose7_0", np.array([-1.2798011 , -0.39391644,  1.08697165,  1.27745898, -1.03213575,
        1.26408687, -4.712499  ])),
("region_pose7_1", np.array([-1.00885828, -0.54347269,  0.6775124 ,  1.37648046, -0.70202433,
        0.98233703, -4.712499  ])),
("region_pose7_2", np.array([-0.90465523, -0.59429582,  0.67751464,  1.50082785, -0.71176542,
        0.92072299, -4.53758962])),
("region_pose7_3", np.array([-0.78174119, -0.62324509,  0.68688946,  1.5755349 , -0.72672194,
        0.88800699, -4.36146519])),
("region_pose7_4", np.array([-0.63720203, -0.63311017,  0.68779825,  1.59996677, -0.73005863,
        0.87600426, -4.20177596])),
("region_pose7_5", np.array([-0.47848194, -0.62403133,  0.68125374,  1.5747817 , -0.72186999,
        0.88528614, -4.06505269])),
("region_pose7_6", np.array([-0.31250755, -0.59537496,  0.66883644,  1.49957177, -0.7040798 ,
        0.91670981, -3.95582055])),
("region_pose7_7", np.array([-0.14417331, -0.5455417 ,  0.65277419,  1.37207137, -0.68024288,
        0.97173252, -3.87604576])),
("region_pose8_0", np.array([-1.25048047, -0.30540993,  1.21443526,  1.15135064, -1.14448383,
        1.38504566, -4.712499  ])),
("region_pose8_1", np.array([-0.97426875, -0.46529019,  0.7473384 ,  1.21084358, -0.75230885,
        1.09678781, -4.712499  ])),
("region_pose8_2", np.array([-0.83376122, -0.52602929,  0.65302265,  1.32485675, -0.67774935,
        0.99489673, -4.59130986])),
("region_pose8_3", np.array([-0.72125202, -0.55664031,  0.66200088,  1.40269018, -0.69063331,
        0.96138524, -4.42624997])),
("region_pose8_4", np.array([-0.58808898, -0.56678818,  0.66381555,  1.42806195, -0.69400584,
        0.9497867 , -4.27700479])),
("region_pose8_5", np.array([-0.44025872, -0.55694039,  0.65902478,  1.40219897, -0.68790277,
        0.96009682, -4.1487473 ])),
("region_pose8_6", np.array([-0.28345479, -0.52631911,  0.64929205,  1.32412268, -0.6741663 ,
        0.99340577, -4.04540903])),
("region_pose8_7", np.array([-0.12200757, -0.47260491,  0.63797692,  1.18967497, -0.65685935,
        1.05245869, -3.96926506])),
("region_pose9_0", np.array([-1.21314766, -0.23433371,  1.32848344,  1.04472243, -1.2518597 ,
        1.47898718, -4.712499  ])),
("region_pose9_1", np.array([-0.95520533, -0.37576457,  0.87402594,  1.03805373, -0.85745746,
        1.234942  , -4.712499  ])),
("region_pose9_2", np.array([-0.76062937, -0.44621832,  0.63321543,  1.12407438, -0.65005636,
        1.08112886, -4.64795635])),
("region_pose9_3", np.array([-0.65707458, -0.4797973 ,  0.63997359,  1.20789019, -0.65978729,
        1.04473299, -4.49211023])),
("region_pose9_4", np.array([-0.53376093, -0.49068948,  0.64213834,  1.23507664, -0.66299732,
        1.03281368, -4.35182106])),
("region_pose9_5", np.array([-0.3955206 , -0.4797959 ,  0.63951153,  1.20770313, -0.65922993,
        1.04461313, -4.23124228])),
("region_pose9_6", np.array([-0.24714121, -0.44610004,  0.63435503,  1.1240134 , -0.6508056 ,
        1.08170373, -4.13384384])),
("region_pose9_7", np.array([-0.09284781, -0.38629475,  0.63355194,  0.97717073, -0.64480857,
        1.1491077 , -4.0621171 ]))
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