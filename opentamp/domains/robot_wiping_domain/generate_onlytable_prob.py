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

# NOTE: Below 7DOF poses obtained by running the following code from a breakpoint in robo_wiping.py:
# for row in range(10):
#         for col in range(8):
#             xyz_pose = params[f"region_pose{row}_{col}"].right_ee_pos.squeeze()
#             quat = np.array([0.0, 0.0, 0.0, 1.0])
#             print((f"region_pose{row}_{col}", f'np.{params["sawyer"].openrave_body.get_ik_from_pose(xyz_pose, quat, "right")},'))

# Moreover, to visualize these, we can use:
# params["sawyer"].openrave_body.set_dof({"right": params["region_pose0_1"].right[:, 0]})
# This will update the visualizer to show the arm at region_pose0_1 for
# example.

region_name_to_jnt_vals = dict([
('region_pose0_0', np.array([-0.6422763 ,  0.22401143,  2.787137  ,  0.06004149, -2.92065789,
        2.75581453, -2.08565301])),
('region_pose0_1', np.array([-5.50382805e-01,  2.44458876e-01,  2.87235982e+00, -2.13176151e-03,
       -2.82486438e+00,  2.68078187e+00, -1.90833167e+00])),
('region_pose0_2', np.array([-0.45754352,  0.26137572,  2.96316342, -0.05796548, -2.7576311 ,
        2.5992671 , -1.75917769])),
('region_pose0_3', np.array([-0.36545055,  0.27194763,  3.042699  , -0.10799586, -2.69898648,
        2.51576935, -1.63374702])),
('region_pose0_4', np.array([-0.27108108,  0.26312206,  3.042699  , -0.15236627, -2.57758764,
        2.44753994, -1.52706489])),
('region_pose0_5', np.array([-0.17980947,  0.25279871,  3.042699  , -0.18953789, -2.47194146,
        2.37995923, -1.43738015])),
('region_pose0_6', np.array([-0.09297949,  0.24193011,  3.042699  , -0.21979282, -2.37975271,
        2.31368679, -1.36108744])),
('region_pose0_7', np.array([-0.01140503,  0.23110468,  3.042699  , -0.24381158, -2.29929662,
        2.24907306, -1.29552707])),
('region_pose1_0', np.array([-0.63780147,  0.22344573,  2.85657712,  0.05220148, -2.976199  ,
        2.74617194, -2.07407646])),
('region_pose1_1', np.array([-0.55095279,  0.24420274,  2.94822624, -0.00916585, -2.89250739,
        2.6694943 , -1.90179695])),
('region_pose1_2', np.array([-0.46275793,  0.25995405,  3.03340143, -0.06418451, -2.82327495,
        2.58885825, -1.75533293])),
('region_pose1_3', np.array([-0.37176433,  0.25665462,  3.042699  , -0.11350387, -2.69792763,
        2.52084126, -1.63141616])),
('region_pose1_4', np.array([-0.28130882,  0.24913569,  3.042699  , -0.15595709, -2.58073853,
        2.45427979, -1.52753241])),
('region_pose1_5', np.array([-0.19352067,  0.2403344 ,  3.042699  , -0.1914756 , -2.47827869,
        2.38824606, -1.43965737])),
('region_pose1_6', np.array([-0.10963196,  0.2309646 ,  3.042699  , -0.22052633, -2.38848592,
        2.32328992, -1.36456437])),
('region_pose1_7', np.array([-0.03042489,  0.22150625,  3.042699  , -0.24379239, -2.30978071,
        2.25974642, -1.29982077])),
('region_pose2_0', np.array([-0.63249742,  0.21529407,  2.87918535,  0.04424817, -2.976199  ,
        2.74334366, -2.05401827])),
('region_pose2_1', np.array([-0.55058233,  0.24061846,  2.99838355, -0.015769  , -2.93061346,
        2.66178158, -1.89130542])),
('region_pose2_2', np.array([-0.46525208,  0.24795209,  3.042699  , -0.06976969, -2.82559715,
        2.59003501, -1.74843961])),
('region_pose2_3', np.array([-0.37767516,  0.24328749,  3.042699  , -0.11759451, -2.69611621,
        2.52530879, -1.62842214])),
('region_pose2_4', np.array([-0.29080667,  0.23675873,  3.042699  , -0.15851378, -2.58295306,
        2.46021043, -1.52731394])),
('region_pose2_5', np.array([-0.20624674,  0.22912849,  3.042699  , -0.19276493, -2.48360727,
        2.39554786, -1.44132839])),
('region_pose2_6', np.array([-0.1251258 ,  0.22095334,  3.042699  , -0.22089448, -2.39619208,
        2.33179662, -1.3675326 ])),
('region_pose2_7', np.array([-0.04818894,  0.21262479,  3.042699  , -0.24357902, -2.31925155,
        2.26927047, -1.30368816])),
('region_pose3_0', np.array([-0.62775479,  0.20799105,  2.90128528,  0.03633537, -2.976199  ,
        2.73953929, -2.03428634])),
('region_pose3_1', np.array([-0.54993791,  0.2356278 ,  3.03496964, -0.02209895, -2.95364035,
        2.65566985, -1.87923582])),
('region_pose3_2', np.array([-0.46732372,  0.23548247,  3.042699  , -0.07470057, -2.81839687,
        2.59240269, -1.74116924])),
('region_pose3_3', np.array([-0.38319748,  0.23132082,  3.042699  , -0.12099121, -2.69377319,
        2.52899796, -1.62501262])),
('region_pose3_4', np.array([-0.29963496,  0.22559971,  3.042699  , -0.16055263, -2.58443671,
        2.46525349, -1.52661672])),
('region_pose3_5', np.array([-0.21807851,  0.21892403,  3.042699  , -0.19372248, -2.48809232,
        2.40186345, -1.44253255])),
('region_pose3_6', np.array([-0.13956639,  0.21173941,  3.042699  , -0.22107486, -2.40299989,
        2.33925022, -1.3700747 ])),
('region_pose3_7', np.array([-0.06480452,  0.20436809,  3.042699  , -0.24326785, -2.3278108 ,
        2.27770792, -1.30717488])),
('region_pose4_0', np.array([-0.62350371,  0.2014395 ,  2.9228851 ,  0.02848956, -2.976199  ,
        2.73488269, -2.01489842])),
('region_pose4_1', np.array([-0.54894407,  0.22585339,  3.042699  , -0.02816217, -2.94805522,
        2.65460414, -1.86671684])),
('region_pose4_2', np.array([-0.46930465,  0.22418223,  3.042699  , -0.07921022, -2.81105626,
        2.59401294, -1.73376367])),
('region_pose4_3', np.array([-0.38835634,  0.22046831,  3.042699  , -0.12401422, -2.69100942,
        2.53187884, -1.62129848])),
('region_pose4_4', np.array([-0.30785271,  0.21542974,  3.042699  , -0.16231921, -2.58530882,
        2.46941396, -1.5255421 ])),
('region_pose4_5', np.array([-0.22909846,  0.20955691,  3.042699  , -0.19451203, -2.49184472,
        2.407229  , -1.44334533])),
('region_pose4_6', np.array([-0.15304825,  0.20321311,  3.042699  , -0.22116778, -2.40900693,
        2.34570613, -1.37224052])),
('region_pose4_7', np.array([-0.08036848,  0.19666593,  3.042699  , -0.2429184 , -2.33554245,
        2.2851224 , -1.31031166])),
('region_pose5_0', np.array([-0.61968417,  0.1955556 ,  2.943994  ,  0.02072941, -2.976199  ,
        2.72947611, -1.99586666])),
('region_pose5_1', np.array([-0.54797206,  0.21544164,  3.042699  , -0.03398882, -2.93518725,
        2.65439183, -1.85436481])),
('region_pose5_2', np.array([-0.47118995,  0.21385817,  3.042699  , -0.08346303, -2.80361219,
        2.59488079, -1.72626473])),
('region_pose5_3', np.array([-0.39317858,  0.21054567,  3.042699  , -0.12682526, -2.68789192,
        2.53397102, -1.61733863])),
('region_pose5_4', np.array([-0.31551421,  0.20609544,  3.042699  , -0.16394312, -2.58564967,
        2.4727266 , -1.5241475 ])),
('region_pose5_5', np.array([-0.23938039,  0.20091079,  3.042699  , -0.19522498, -2.49494675,
        2.4116936 , -1.44381302])),
('region_pose5_6', np.array([-0.16565581,  0.1952917 ,  3.042699  , -0.2212329 , -2.41429147,
        2.35122248, -1.37406341])),
('region_pose5_7', np.array([-0.09496802,  0.18946209,  3.042699  , -0.24256869, -2.3425178 ,
        2.29157547, -1.31312107])),
('region_pose6_0', np.array([-0.61624455,  0.19026719,  2.96462238,  0.01306846, -2.976199  ,
        2.72340467, -1.97719832])),
('region_pose6_1', np.array([-0.54711401,  0.20584675,  3.042699  , -0.03964722, -2.92254203,
        2.65348245, -1.84217879])),
('region_pose6_2', np.array([-0.4729788 ,  0.20437201,  3.042699  , -0.08754845, -2.79608748,
        2.59504147, -1.71869697])),
('region_pose6_3', np.array([-0.39769013,  0.20142097,  3.042699  , -0.12951343, -2.68446694,
        2.53531165, -1.6131692 ])),
('region_pose6_4', np.array([-0.32266847,  0.19748434,  3.042699  , -0.16549765, -2.58551893,
        2.47523626, -1.52246992])),
('region_pose6_5', np.array([-0.24899003,  0.19289716,  3.042699  , -0.19591494, -2.49746354,
        2.41530954, -1.44396733])),
('region_pose6_6', np.array([-0.17746447,  0.18790945,  3.042699  , -0.22130688, -2.41891876,
        2.35585613, -1.37556793])),
('region_pose6_7', np.array([-0.10868173,  0.18270978,  3.042699  , -0.24224344, -2.34879864,
        2.29712528, -1.31562106])),
('region_pose7_0', np.array([-0.6131402 ,  0.18551191,  2.9847817 ,  0.00551657, -2.976199  ,
        2.71673957, -1.95889677])),
('region_pose7_1', np.array([-0.54634868,  0.19697254,  3.042699  , -0.04517071, -2.91011115,
        2.65192599, -1.83015926])),
('region_pose7_2', np.array([-0.47467252,  0.19561764,  3.042699  , -0.09151863, -2.7884981 ,
        2.59453603, -1.71107681])),
('region_pose7_3', np.array([-0.40191512,  0.19299325,  3.042699  , -0.13213071, -2.68076957,
        2.53594304, -1.60881516])),
('region_pose7_4', np.array([-0.32935937,  0.18950901,  3.042699  , -0.1670261 , -2.58496401,
        2.47698984, -1.52053592])),
('region_pose7_5', np.array([-0.25798595,  0.18544537,  3.042699  , -0.19661434, -2.49944897,
        2.41812809, -1.44383219])),
('region_pose7_6', np.array([-0.18854167,  0.18101218,  3.042699  , -0.22141263, -2.4229446 ,
        2.35966088, -1.37677371])),
('region_pose7_7', np.array([-0.12158047,  0.17636913,  3.042699  , -0.2419589 , -2.35443924,
        2.30182606, -1.31782689])),
('region_pose8_0', np.array([-6.10332440e-01,  1.81235715e-01,  3.00448418e+00, -1.91920386e-03,
       -2.97619900e+00,  2.70954062e+00, -1.94096235e+00])),
('region_pose8_1', np.array([-0.54565889,  0.18873993,  3.042699  , -0.05057995, -2.89788622,
        2.64976906, -1.81830529])),
('region_pose8_2', np.array([-0.47627353,  0.18750998,  3.042699  , -0.09540548, -2.78085637,
        2.59340564, -1.70341644])),
('region_pose8_3', np.array([-0.40587564,  0.18518173,  3.042699  , -0.13470866, -2.67682822,
        2.53590751, -1.60429568])),
('region_pose8_4', np.array([-0.33562615,  0.18209874,  3.042699  , -0.16855475, -2.58402432,
        2.47803287, -1.51836642])),
('region_pose8_5', np.array([-0.26642032,  0.17849715,  3.042699  , -0.19734301, -2.50094896,
        2.42019769, -1.44342712])),
('region_pose8_6', np.array([-0.19894776,  0.17455416,  3.042699  , -0.22156448, -2.42641757,
        2.36268686, -1.37769747])),
('region_pose8_7', np.array([-0.13372818,  0.17040552,  3.042699  , -0.24172561, -2.35948779,
        2.30572801, -1.31975219])),
('region_pose9_0', np.array([-0.60778754,  0.17739144,  3.02374266, -0.009234  , -2.976199  ,
        2.7018582 , -1.92339296])),
('region_pose9_1', np.array([-0.54503062,  0.18108245,  3.042699  , -0.05588835, -2.8858591 ,
        2.64705377, -1.80661506])),
('region_pose9_2', np.array([-0.47778491,  0.17997883,  3.042699  , -0.09922917, -2.77317236,
        2.59168919, -1.6957257 ])),
('region_pose9_3', np.array([-0.40959183,  0.17791982,  3.042699  , -0.13726679, -2.67266688,
        2.53524524, -1.59962662])),
('region_pose9_4', np.array([-0.34150387,  0.17519474,  3.042699  , -0.17009963, -2.5827337 ,
        2.47840806, -1.51597905])),
('region_pose9_5', np.array([-0.27433976,  0.17200336,  3.042699  , -0.19811299, -2.50200349,
        2.42156334, -1.44276905])),
('region_pose9_6', np.array([-0.20873689,  0.16849598,  3.042699  , -0.2217712 , -2.42938062,
        2.3649803 , -1.37835418])),
('region_pose9_7', np.array([-0.14518267,  0.16478849,  3.042699  , -0.24155031, -2.36398746,
        2.30887742, -1.3214096]))
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