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
ON_TABLE_POS = [0.4, -0.15, 0.912]

# TABLE_GEOM = [0.5, 0.8, 0.9]
TABLE_GEOM = [0.5, 0.8, 0.8 + 0.08909001]
# TABLE_POS = [0.5, 0.0, -0.040]
TABLE_POS = [0.5, 0.0, 0.025]
TABLE_ROT = [0,0,0]

num_rows = 10
num_cols = 8
xy_ontable_poses = [[] for _ in range(num_rows)]
row_step_size = TABLE_GEOM[0] / num_rows
col_step_size = TABLE_GEOM[1] / num_cols
for row in range(num_rows):
    for col in range(num_cols):
        xy_ontable_poses[row].append([TABLE_POS[0] + row * row_step_size, TABLE_POS[1] + col * col_step_size, 0.912])

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
("region_pose0_0", np.array([-0.39519857, -0.22609743,  0.84378463,  0.4778734 , -0.82086871,
        1.4680674 , -4.57941063])),
("region_pose0_1", np.array([-0.29019884, -0.21618316,  0.89249123,  0.45299579, -0.86739522,
        1.49136652, -4.47823824])),
("region_pose0_2", np.array([-0.20111156, -0.1897045 ,  1.05830069,  0.39024525, -1.02711434,
        1.55724569, -4.40000957])),
("region_pose0_3", np.array([-0.13348661, -0.14841578,  1.36971214,  0.31195321, -1.33388163,
        1.64516772, -4.36382108])),
("region_pose0_4", np.array([-0.06812136, -0.09204648,  1.79474744,  0.21826582, -1.76841573,
        1.70147649, -4.38995229])),
("region_pose0_5", np.array([ 3.53683451e-03, -3.23541831e-02,  2.27864693e+00,  1.20633175e-01,
       -2.26804235e+00,  1.67054483e+00, -4.44074731e+00])),
("region_pose0_6", np.array([ 0.07581602,  0.01154134,  2.64997222,  0.05613452, -2.64086765,
        1.59053679, -4.43425618])),
("region_pose0_7", np.array([ 0.1451642 ,  0.03327288,  2.83633643,  0.02787555, -2.82145171,
        1.53533912, -4.38362628])),
("region_pose1_0", np.array([-0.45109111, -0.14656268,  1.38100404,  0.30553595, -1.34565799,
        1.64801402, -4.68683608])),
("region_pose1_1", np.array([-0.35383763, -0.13806794,  1.44620919,  0.29209356, -1.41120373,
        1.66111751, -4.59938877])),
("region_pose1_2", np.array([-0.26681629, -0.11349386,  1.63179141,  0.25206913, -1.60035278,
        1.68909371, -4.54985845])),
("region_pose1_3", np.array([-0.18476893, -0.07537367,  1.92654928,  0.18976411, -1.90514982,
        1.70260607, -4.54175938])),
("region_pose1_4", np.array([-0.10427291, -0.03083466,  2.29226138,  0.11787317, -2.28193433,
        1.66824923, -4.55169841])),
("region_pose1_5", np.array([-0.02603918,  0.00676817,  2.60995857,  0.06252477, -2.6013596 ,
        1.60096825, -4.5308119 ])),
("region_pose1_6", np.array([ 0.04875788,  0.02896289,  2.79862296,  0.03338819, -2.78550358,
        1.54750519, -4.47697057])),
("region_pose1_7", np.array([ 0.11852336,  0.04055156,  2.90352381,  0.01818698, -2.88415719,
        1.51148643, -4.414827  ])),
("region_pose2_0", np.array([-0.53945961, -0.07604554,  1.82025837,  0.35400757, -1.77863607,
        1.71531847, -4.712499  ])),
("region_pose2_1", np.array([-0.41567129, -0.0606114 ,  1.95854493,  0.24912078, -1.93168437,
        1.71264473, -4.712499  ])),
("region_pose2_2", np.array([-0.30445195, -0.04208464,  2.17165629,  0.15289247, -2.15758635,
        1.68847642, -4.712499  ])),
("region_pose2_3", np.array([-0.21301725, -0.01685518,  2.41108348,  0.09613259, -2.40248603,
        1.64647922, -4.68498961])),
("region_pose2_4", np.array([-0.12943519,  0.0090154 ,  2.62927232,  0.05937056, -2.62044176,
        1.5959094 , -4.63679376])),
("region_pose2_5", np.array([-0.04984798,  0.02729763,  2.78438362,  0.03548795, -2.7718246 ,
        1.55190986, -4.57430486])),
("region_pose2_6", np.array([ 0.02486382,  0.03847332,  2.88389111,  0.02101789, -2.86604613,
        1.51877096, -4.50721612])),
("region_pose2_7", np.array([ 0.09399377,  0.04504462,  2.94940521,  0.01150869, -2.92543311,
        1.49272162, -4.44198064])),
("region_pose3_0", np.array([-0.55682256, -0.03800772,  1.98609727,  0.36709576, -1.94693811,
        1.72952262, -4.712499  ])),
("region_pose3_1", np.array([-0.44721183, -0.01432829,  2.15024305,  0.2864794 , -2.12451864,
        1.71996116, -4.712499  ])),
("region_pose3_2", np.array([-0.33904224,  0.00752255,  2.36014246,  0.19998004, -2.34566939,
        1.68510794, -4.712499  ])),
("region_pose3_3", np.array([-0.23594139,  0.02254728,  2.62071137,  0.09756028, -2.61094402,
        1.6115578 , -4.712499  ])),
("region_pose3_4", np.array([-0.14888697,  0.02903767,  2.7996416 ,  0.03323363, -2.78646277,
        1.54710918, -4.67463275])),
("region_pose3_5", np.array([-0.07029489,  0.03795053,  2.87915756,  0.02170346, -2.8616425 ,
        1.52045242, -4.60203025])),
("region_pose3_6", np.array([ 3.18988701e-03,  4.40149468e-02,  2.93851232e+00,  1.31198964e-02,
       -2.91578352e+00,  1.49739905e+00, -4.53215260e+00])),
("region_pose3_7", np.array([ 0.07124007,  0.0479031 ,  2.98284609,  0.00644781, -2.95419678,
        1.47687342, -4.46650267])),
("region_pose4_0", np.array([-0.55499994, -0.01770796,  2.08564387,  0.35701765, -2.0498457 ,
        1.72809656, -4.712499  ])),
("region_pose4_1", np.array([-0.45135382,  0.00611611,  2.25060746,  0.28494391, -2.22664408,
        1.71228512, -4.712499  ])),
("region_pose4_2", np.array([-0.34716286,  0.02665908,  2.45497475,  0.20282439, -2.44007037,
        1.6710704 , -4.712499  ])),
("region_pose4_3", np.array([-0.24756982,  0.03926022,  2.70395899,  0.10263211, -2.69131818,
        1.59344868, -4.712499  ])),
("region_pose4_4", np.array([-0.16527864,  0.03918762,  2.89090133,  0.02002094, -2.87251885,
        1.51613029, -4.69773811])),
("region_pose4_5", np.array([-0.08858681,  0.0439196 ,  2.93764249,  0.01325461, -2.91500128,
        1.4977352 , -4.62384692])),
("region_pose4_6", np.array([-0.016732  ,  0.04739343,  2.97661203,  0.00742039, -2.94893894,
        1.47998195, -4.55411483])),
("region_pose4_7", np.array([ 5.00396016e-02,  4.97499417e-02,  3.00845734e+00,  2.33706638e-03,
       -2.97507514e+00,  1.46278925e+00, -4.48905611e+00])),
("region_pose5_0", np.array([-5.48563275e-01, -3.94384466e-03,  2.16529341e+00,  3.40551273e-01,
       -2.13237306e+00,  1.72018713e+00, -4.71249900e+00])),
("region_pose5_1", np.array([-0.44946033,  0.0182016 ,  2.3278301 ,  0.27289568, -2.30496184,
        1.69883696, -4.712499  ])),
("region_pose5_2", np.array([-0.34955388,  0.03653051,  2.52748407,  0.19328353, -2.51143197,
        1.65244755, -4.712499  ])),
("region_pose5_3", np.array([-0.25501263,  0.04686889,  2.76643733,  0.09617756, -2.75041044,
        1.57261175, -4.712499  ])),
("region_pose5_4", np.array([-0.17940221,  0.04516433,  2.94428114,  0.01396832, -2.92079399,
        1.495254  , -4.712499  ])),
("region_pose5_5", np.array([-0.10526791,  0.04745288,  2.97747269,  0.00729586, -2.94966036,
        1.47952315, -4.64266347])),
("region_pose5_6", np.array([-3.51761583e-02,  4.95052090e-02,  3.00489902e+00,  2.93365775e-03,
       -2.97224330e+00,  1.46484458e+00, -4.57404839e+00])),
("region_pose5_7", np.array([ 3.04449813e-02,  4.82793965e-02,  3.01485965e+00, -6.83079083e-04,
       -2.97619900e+00,  1.45294586e+00, -4.50963473e+00])),
("region_pose6_0", np.array([-0.54072023,  0.0065565 ,  2.23615698,  0.32178154, -2.20542414,
        1.70833224, -4.712499  ])),
("region_pose6_1", np.array([-0.4456586 ,  0.02659578,  2.39610398,  0.25680644, -2.37359683,
        1.68202121, -4.712499  ])),
("region_pose6_2", np.array([-0.35009427,  0.04256957,  2.59202684,  0.1787757 , -2.5741664 ,
        1.63107436, -4.712499  ])),
("region_pose6_3", np.array([-0.26111895,  0.05077627,  2.81981175,  0.08550774, -2.79999445,
        1.55061368, -4.712499  ])),
("region_pose6_4", np.array([-0.19027404,  0.04956886,  2.971782  ,  0.01583494, -2.94409242,
        1.48315432, -4.712499  ])),
("region_pose6_5", np.array([-1.20631477e-01,  4.96085333e-02,  3.00658033e+00,  2.66438898e-03,
       -2.97357536e+00,  1.46383469e+00, -4.65956330e+00])),
("region_pose6_6", np.array([-5.21324342e-02,  4.84343356e-02,  3.01431094e+00, -3.76730095e-04,
       -2.97619900e+00,  1.45391829e+00, -4.59208086e+00])),
("region_pose6_7", np.array([ 1.20513913e-02,  4.68607751e-02,  3.02020292e+00, -3.43054218e-03,
       -2.97619900e+00,  1.44324261e+00, -4.52899761e+00])),
("region_pose7_0", np.array([-0.5324967 ,  0.01504063,  2.30225915,  0.30201369, -2.27297517,
        1.69356842, -4.712499  ])),
("region_pose7_1", np.array([-0.44119988,  0.03290826,  2.46006571,  0.23871598, -2.43719761,
        1.6627239 , -4.712499  ])),
("region_pose7_2", np.array([-0.34998218,  0.04657388,  2.65295998,  0.16172581, -2.6326802 ,
        1.60767338, -4.712499  ])),
("region_pose7_3", np.array([-0.26678718,  0.05283889,  2.8668534 ,  0.07350161, -2.84293343,
        1.52849773, -4.712499  ])),
("region_pose7_4", np.array([-0.20012378,  0.05188009,  2.99388091,  0.01525115, -2.9618996 ,
        1.47072431, -4.712499  ])),
("region_pose7_5", np.array([-1.34646180e-01,  4.82541759e-02,  3.01491358e+00, -6.89264685e-04,
       -2.97619900e+00,  1.45280760e+00, -4.67467371e+00])),
("region_pose7_6", np.array([-6.79597476e-02,  4.69315716e-02,  3.01989621e+00, -3.26691179e-03,
       -2.97619900e+00,  1.44378096e+00, -4.60892258e+00])),
("region_pose7_7", np.array([-0.00524857,  0.0456579 ,  3.02557129, -0.0060133 , -2.976199  ,
        1.43342356, -4.54728103])),
("region_pose8_0", np.array([-0.52430136,  0.02208859,  2.36548974,  0.28171861, -2.33690765,
        1.67643143, -4.712499  ])),
("region_pose8_1", np.array([-0.43659113,  0.03783334,  2.52173815,  0.21941175, -2.49780869,
        1.64136556, -4.712499  ])),
("region_pose8_2", np.array([-0.34975182,  0.04929446,  2.71183176,  0.14321873, -2.68856418,
        1.58268576, -4.712499  ])),
("region_pose8_3", np.array([-0.27235665,  0.05386967,  2.90805025,  0.06165878, -2.87980169,
        1.50706338, -4.712499  ])),
("region_pose8_4", np.array([-0.20924432,  0.05308955,  3.01246192,  0.01350583, -2.97612695,
        1.45824361, -4.712499  ])),
("region_pose8_5", np.array([-1.47712007e-01,  4.67332215e-02,  3.02069675e+00, -3.65269831e-03,
       -2.97619900e+00,  1.44230262e+00, -4.68879163e+00])),
("region_pose8_6", np.array([-0.08280758,  0.0456711 ,  3.02548925, -0.00596406, -2.976199  ,
        1.43355418, -4.62479579])),
("region_pose8_7", np.array([-0.02153812,  0.04463945,  3.03095942, -0.00847561, -2.976199  ,
        1.42348715, -4.56457735])),
("region_pose9_0", np.array([-0.51632544,  0.02801142,  2.42691577,  0.26106081, -2.39829645,
        1.65723943, -4.712499  ])),
("region_pose9_1", np.array([-0.43209066,  0.04172077,  2.58217594,  0.19923703, -2.5565101 ,
        1.61818623, -4.712499  ])),
("region_pose9_2", np.array([-0.34972394,  0.05110364,  2.7689255 ,  0.12396474, -2.74215106,
        1.55655625, -4.712499  ])),
("region_pose9_3", np.array([-0.2779264 ,  0.0542928 ,  2.94349166,  0.05079039, -2.91076617,
        1.48683832, -4.712499  ])),
("region_pose9_4", np.array([-0.21780051,  0.05200763,  3.01734141,  0.0125198 , -2.976199  ,
        1.44877631, -4.712499  ])),
("region_pose9_5", np.array([-0.15994793,  0.04546812,  3.02647243, -0.00641196, -2.976199  ,
        1.43172831, -4.70209041])),
("region_pose9_6", np.array([-0.09675582,  0.04461433,  3.03108618, -0.00852185, -2.976199  ,
        1.42323267, -4.63979114])),
("region_pose9_7", np.array([-0.03689393,  0.04378161,  3.03636246, -0.01084742, -2.976199  ,
        1.41343756, -4.58096978]))
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