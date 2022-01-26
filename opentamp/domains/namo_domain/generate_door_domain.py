dom_str = """
# AUTOGENERATED. DO NOT EDIT.
# Configuration file for CAN domain. Blank lines and lines beginning with # are filtered out.

# implicity, all types require a name
Types: Can, Target, RobotPose, Robot, Grasp, Obstacle, Rotation, Door

# Define the class location of each non-standard attribute type used in the above parameter type descriptions.

Attribute Import Paths: RedCircle core.util_classes.items, BlueCircle core.util_classes.items, GreenCircle core.util_classes.items, Vector1d core.util_classes.matrix, Vector2d core.util_classes.matrix, Wall core.util_classes.items, NAMO core.util_classes.robots, Door2d core.util_classes.items

Predicates Import Path: core.util_classes.namo_grip_predicates

"""

prim_pred_str = 'Primitive Predicates: geom, Can, RedCircle; pose, Can, Vector2d; geom, Target, BlueCircle; value, Target, Vector2d; value, RobotPose, Vector2d; gripper, RobotPose, Vector1d; geom, RobotPose, BlueCircle; geom, Robot, NAMO; pose, Robot, Vector2d; gripper, Robot, Vector1d; value, Grasp, Vector2d; geom, Obstacle, Wall; pose, Obstacle, Vector2d; value, Rotation, Vector1d; theta, Robot, Vector1d; theta, RobotPose, Vector1d; vel, RobotPose, Vector1d; acc, RobotPose, Vector1d; vel, Robot, Vector1d; acc, Robot, Vector1d; geom, Door, Door2d; theta, Door, Vector1d; pose, Door, Vector2d'
dom_str += prim_pred_str + '\n\n'

der_pred_str = 'Derived Predicates: At, Can, Target; AtInit, Can, Target; RobotAt, Robot, RobotPose; InGripper, Robot, Can, Grasp; Obstructs, Robot, RobotPose, RobotPose, Can; ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can; WideObstructsHolding, Robot, RobotPose, RobotPose, Can, Can; StationaryRot, Robot; Stationary, Can; RobotStationary, Robot; StationaryNEq, Can, Can; IsMP, Robot; StationaryW, Obstacle; Collides, Can, Obstacle; CanCollides, Can, Can; RCollides, Robot, Obstacle; GripperClosed, Robot; Near, Can, Target;  RobotAtGrasp, Robot, Can, Grasp; RobotWithinReach, Robot, Target; RobotNearGrasp, Robot, Can, Grasp; RobotWithinBounds, Robot; WideObstructs, Robot, RobotPose, RobotPose, Can; AtNEq, Can, Can, Target; PoseCollides, RobotPose, Obstacle; TargetCollides, Target, Obstacle; TargetGraspCollides, Target, Obstacle, Grasp; TargetCanGraspCollides, Target, Can, Grasp; CanGraspCollides, Can, Obstacle, Grasp; HLPoseUsed, RobotPose; HLAtGrasp, Robot, Can, Grasp; RobotPoseAtGrasp, RobotPose, Target, Grasp; HLPoseAtGrasp, RobotPose, Target, Grasp; RobotRetreat, Robot, Grasp; RobotApproach, Robot, Grasp; LinearRetreat, Robot; LinearApproach, Robot; InGraspAngle, Robot, Can; NearGraspAngle, Robot, Can; ThetaDirValid, Robot; ForThetaDirValid, Robot; RevThetaDirValid, Robot; ScalarVelValid, Robot; DoorClosed, Door; InDoorAngle, Robot, Door; DoorIsMP, Door; StationaryDoor, Door; DoorInGrasp, Robot, Door; HandleAngleValid, Door; DoorNearClosed, Door; StationaryDoorPos, Door; OpenDoorReady, Robot; CloseDoorReady, Robot; OpenDoorApproach, Robot; CloseDoorApproach, Robot; AtCloset, Robot; InCloset, Can; DoorObstructs, Robot, RobotPose, RobotPose, Door'
dom_str += der_pred_str + '\n'

dom_str += """

# The first set of parentheses after the colon contains the
# parameters. The second contains preconditions and the third contains
# effects. This split between preconditions and effects is only used
# for task planning purposes. Our system treats all predicates
# similarly, using the numbers at the end, which specify active
# timesteps during which each predicate must hold

"""

class Action(object):
    def __init__(self, name, timesteps, pre=None, post=None):
        pass

    def to_str(self):
        time_str = ''
        cond_str = '(and '
        for pre, timesteps in self.pre:
            cond_str += pre + ' '
            time_str += timesteps + ' '
        cond_str += ')'

        cond_str += '(and '
        for eff, timesteps in self.eff:
            cond_str += eff + ' '
            time_str += timesteps + ' '
        cond_str += ')'

        return "Action " + self.name + ' ' + str(self.timesteps) + ': ' + self.args + ' ' + cond_str + ' ' + time_str

class MoveTo(Action):
    def __init__(self):
        self.name = 'moveto'
        self.timesteps = 25
        et = self.timesteps - 1
        self.args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?gp - RobotPose ?end - Target)' 
        self.pre = [\
                ('(At ?can ?target)', '0:0'),
                ('(forall (?gr - Grasp) (forall (?obj - Can) (not (NearGraspAngle ?robot ?obj))))', '0:0'),
                # ('(forall (?w - Obstacle) (not (CanGraspCollides ?can ?w ?g)))', '0:0'),
                ('(not (GripperClosed ?robot))', '1:{0}'.format(et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '0:{0}'.format(et-1)),
                ('(forall (?obj - Door) (StationaryDoor ?obj))', '0:{0}'.format(et-1)),
                ('(forall (?obj - Door) (StationaryDoorPos ?obj))', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (StationaryW ?w))', '0:{0}'.format(et-1)),
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '1:{0}'.format(et-1)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?sp ?gp ?obj)))', '0:0'),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?sp ?gp ?obj)))', '0:-1'),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?sp ?gp ?obj)))', '1:{0}'.format(et-4)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?sp ?gp ?obj)))', '{0}:{1}'.format(et-3, et-3)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?sp ?gp ?obj ?can)))', '{0}:{1}'.format(et-2, et-1)),
                ('(forall (?obj - Door) (not (DoorObstructs ?robot ?sp ?gp ?obj)))', '{0}:{1}'.format(1, et-1)),
                ('(ThetaDirValid ?robot)', '{0}:{1}'.format(1, et-3)),
                ('(ForThetaDirValid ?robot)', '{0}:{1}'.format(et-3, et-1)),
                # ('(RobotStationary ?robot)', '{0}:{1}'.format(0,0)),
                # ('(LinearApproach ?robot)', '17:17'),
                ('(At ?can ?target)', '{0}:{0}'.format(et-1)),
        ]
        self.eff = [\
                ('(NearGraspAngle ?robot ?can)', '{0}:{0}'.format(et)),
                # ('(InGraspAngle ?robot ?can)', '{0}:{0}'.format(et)),
                ('(forall (?obj - Can / ?can) (forall (?gr - Grasp) (not (NearGraspAngle ?robot ?obj))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
                ('(StationaryRot ?robot)', '{0}:{1}'.format(et-3, et-1)),
                ('(RobotStationary ?robot)', '{0}:{0}'.format(et-1)),
        ]

class PutInCloset(Action):
    def __init__(self):
        self.name = 'put_in_closet'
        self.timesteps = 20
        et = self.timesteps - 1
        enter_closet = et - 4
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can ?door - Door ?init - Target)'
        self.pre = [\
                ('(At ?c ?init)', '0:0'),
                ('(AtCloset ?robot)', '{0}:{0}'.format(enter_closet, enter_closet)),
                ('(not (DoorNearClosed ?door))', '0:0'),
                ('(not (DoorClosed ?door))', '0:0'),
                ('(not (DoorClosed ?door))', '1:{0}'.format(et-1)),
                ('(RobotStationary ?robot)', '0:0'),
                ('(forall (?obj - Door) (StationaryDoor ?obj))', '0:{0}'.format(et-1)),
                ('(forall (?obj - Door) (StationaryDoorPos ?obj))', '0:{0}'.format(et-1)),
                ('(NearGraspAngle ?robot ?c)', '0:0'),
                ('(GripperClosed ?robot)', '1:{0}'.format(et-1)),
                ('(NearGraspAngle ?robot ?c)', '{0}:{0}'.format(et)),
                ('(NearGraspAngle ?robot ?c)', '{0}:{0}'.format(enter_closet, enter_closet)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?start ?end ?obj ?c)))', '0:{0}'.format(0)),
                ('(forall (?obj - Can) (not (WideObstructsHolding ?robot ?start ?end ?obj ?c)))', '0:{0}'.format(-1)),
                ('(forall (?obj - Can) (not (WideObstructsHolding ?robot ?start ?end ?obj ?c)))', '1:{0}'.format(enter_closet)),
                ('(forall (?obj - Door) (not (DoorObstructs ?robot ?start ?end ?obj)))', '{0}:{1}'.format(1, enter_closet-1)),
                ('(forall (?obj - Can) (StationaryNEq ?obj ?c))', '0:{0}'.format(et-1)), 
                ('(forall (?w - Obstacle) (StationaryW ?w))', '0:{0}'.format(et-1)), 
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '1:{0}'.format(enter_closet-1)),
                ('(RobotStationary ?robot)', '{0}:{0}'.format(et-1)),
                ('(StationaryRot ?robot)', '{0}:{1}'.format(enter_closet, et-1)),
                ('(not (GripperClosed ?robot))', '{0}:{1}'.format(et, et-1)),
                ('(ThetaDirValid ?robot)', '{0}:{1}'.format(1, et-1)),
                ]
        self.eff = [\
                ('(InCloset ?c)', '{0}:{0}'.format(et)),
                ('(not (Near ?c ?init))', '{0}:{1}'.format(et, et-1)),
                ('(not (At ?c ?init))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?start ?end ?c)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?start ?end ?c)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (forall (?targ - Target) (not (WideObstructsHolding ?robot ?start ?end ?c ?obj))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (forall (?targ - Target) (not (ObstructsHolding ?robot ?start ?end ?c ?obj))))', '{0}:{1}'.format(et, et-1)),
        ]

class LeaveCloset(Action):
    def __init__(self):
        self.name = 'leave_closet'
        self.timesteps = 7
        et = self.timesteps - 1
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can ?door - Door ?init - Target)'
        self.pre = [\
                ('(InCloset ?c)', '0:0'),
                ('(NearGraspAngle ?robot ?c)', '0:0'),
                ('(not (GripperClosed ?robot))', '1:{0}'.format(et-1)),
                ('(forall (?obj - Door) (StationaryDoor ?obj))', '0:{0}'.format(et-1)),
                ('(forall (?obj - Door) (StationaryDoorPos ?obj))', '0:{0}'.format(et-1)),
                ('(forall (?obj - Can ) (not (Obstructs ?robot ?start ?end ?obj)))', '{0}:{0}'.format(et-1)),
                ('(forall (?obj - Door) (not (DoorObstructs ?robot ?start ?end ?obj)))', '{0}:{1}'.format(1, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '0:{0}'.format(et-1)), 
                ('(forall (?w - Obstacle) (StationaryW ?w))', '0:{0}'.format(et-1)), 
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                # ('(LinearRetreat ?robot)', '0:{0}'.format(et-1)),
                ('(StationaryRot ?robot)', '0:{0}'.format(et-1)),
                # ('(RevThetaDirValid ?robot)', '0:{0}'.format(et-1)),
                ('(RobotStationary ?robot)', '{0}:{0}'.format(0)),
                ('(RobotStationary ?robot)', '{0}:{0}'.format(et-1)),
                ]
        self.eff = [\
                ('(not (Obstructs ?robot ?start ?end ?c))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?gr - Grasp) (forall (?obj - Can) (not (NearGraspAngle ?robot ?obj))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
        ]


class OpenDoor(Action):
    def __init__(self):
        self.name = 'open_door'
        self.timesteps = 25
        et = self.timesteps - 1
        grasp_time = et - 7
        release_time = et - 3
        self.args = '(?robot - Robot ?door - Door ?sp - RobotPose ?gp - RobotPose)' 
        self.pre = [\
                ('(DoorNearClosed ?door)', '0:0'),
                ('(DoorClosed ?door)', '1:{0}'.format(grasp_time)),
                ('(OpenDoorReady ?robot)', '{0}:{0}'.format(grasp_time-1, grasp_time)),
                ('(OpenDoorApproach ?robot)', '{0}:{0}'.format(grasp_time-2, grasp_time-2)),
                ('(CloseDoorReady ?robot)', '{0}:{0}'.format(release_time, release_time)),
                ('(forall (?gr - Grasp) (forall (?obj - Can) (not (NearGraspAngle ?robot ?obj))))', '0:0'),
                ('(not (GripperClosed ?robot))', '1:{0}'.format(grasp_time-1)),
                ('(GripperClosed ?robot)', '{0}:{1}'.format(grasp_time, release_time)),
                ('(not (GripperClosed ?robot))', '{0}:{1}'.format(release_time+1, et-1)),
                ('(InDoorAngle ?robot ?door)', '{0}:{1}'.format(grasp_time-2, release_time)),
                ('(DoorInGrasp ?robot ?door)', '{0}:{1}'.format(grasp_time, release_time)),
                ('(StationaryDoorPos ?door)', '0:{0}'.format(et-1)),
                #('(StationaryRot ?robot)', '{0}:{1}'.format(grasp_time, et-1)),
                #('(RobotStationary ?robot)', '{0}:{0}'.format(release_time)),
                ('(forall (?obj - Can) (Stationary ?obj))', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (StationaryW ?w))', '0:{0}'.format(et-1)),
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                ('(DoorIsMP ?door)', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '1:{0}'.format(grasp_time-2)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?sp ?gp ?obj)))', '0:0'),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?sp ?gp ?obj)))', '0:-1'),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?sp ?gp ?obj)))', '1:{0}'.format(et-1)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?sp ?gp ?obj)))', '{0}:{1}'.format(release_time, release_time)),
                ('(forall (?obj - Door) (not (DoorObstructs ?robot ?sp ?gp ?obj)))', '{0}:{1}'.format(1, grasp_time-3)),
                ('(ThetaDirValid ?robot)', '{0}:{1}'.format(1, grasp_time-1)),
                #('(ForThetaDirValid ?robot)', '{0}:{1}'.format(et-8, et-7)),
                #('(ThetaDirValid ?robot)', '{0}:{1}'.format(et-7, et-1)),
        ]
        self.eff = [\
                ('(not (DoorClosed ?door))', '{0}:{1}'.format(release_time, et)),
                ('(not (DoorNearClosed ?door))', '{0}:{1}'.format(et-1, et)),
                # ('(InGraspAngle ?robot ?can)', '{0}:{0}'.format(et)),
                ('(forall (?obj - Can) (forall (?gr - Grasp) (not (NearGraspAngle ?robot ?obj))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
                #('(RobotStationary ?robot)', '{0}:{0}'.format(et-1)),
        ]



class CloseDoor(Action):
    def __init__(self):
        self.name = 'close_door'
        self.timesteps = 25
        et = self.timesteps - 1
        grasp_time = et - 7
        release_time = et - 3
        self.args = '(?robot - Robot ?door - Door ?sp - RobotPose ?gp - RobotPose)' 
        self.pre = [\
                ('(not (DoorNearClosed ?door))', '0:0'),
                ('(not (DoorClosed ?door))', '1:{0}'.format(grasp_time)),
                ('(CloseDoorReady ?robot)', '{0}:{0}'.format(grasp_time, grasp_time)),
                ('(CloseDoorApproach ?robot)', '{0}:{0}'.format(grasp_time-2, grasp_time-2)),
                ('(forall (?gr - Grasp) (forall (?obj - Can) (not (NearGraspAngle ?robot ?obj))))', '0:0'),
                ('(not (GripperClosed ?robot))', '1:{0}'.format(grasp_time-1)),
                ('(GripperClosed ?robot)', '{0}:{1}'.format(grasp_time, release_time)),
                ('(not (GripperClosed ?robot))', '{0}:{1}'.format(release_time+1, et-1)),
                ('(InDoorAngle ?robot ?door)', '{0}:{1}'.format(grasp_time-2, release_time)),
                ('(DoorInGrasp ?robot ?door)', '{0}:{1}'.format(grasp_time, release_time)),
                ('(StationaryDoorPos ?door)', '0:{0}'.format(et-1)),
                #('(StationaryRot ?robot)', '{0}:{1}'.format(grasp_time, et-1)),
                #('(RobotStationary ?robot)', '{0}:{0}'.format(release_time)),
                ('(forall (?obj - Can) (Stationary ?obj))', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (StationaryW ?w))', '0:{0}'.format(et-1)),
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                ('(DoorIsMP ?door)', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '1:{0}'.format(grasp_time-2)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?sp ?gp ?obj)))', '0:0'),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?sp ?gp ?obj)))', '0:-1'),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?sp ?gp ?obj)))', '1:{0}'.format(et-1)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?sp ?gp ?obj)))', '{0}:{1}'.format(release_time, release_time)),
                ('(forall (?obj - Door) (not (DoorObstructs ?robot ?sp ?gp ?obj)))', '{0}:{1}'.format(1, grasp_time-3)),
                ('(ThetaDirValid ?robot)', '{0}:{1}'.format(1, grasp_time-1)),
                #('(ForThetaDirValid ?robot)', '{0}:{1}'.format(et-8, et-7)),
                #('(ThetaDirValid ?robot)', '{0}:{1}'.format(et-7, et-1)),
        ]
        self.eff = [\
                ('(DoorClosed ?door)', '{0}:{1}'.format(release_time, et)),
                ('(DoorNearClosed ?door)', '{0}:{1}'.format(et-1, et)),
                # ('(InGraspAngle ?robot ?can)', '{0}:{0}'.format(et)),
                ('(forall (?obj - Can) (forall (?gr - Grasp) (not (NearGraspAngle ?robot ?obj))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
                #('(RobotStationary ?robot)', '{0}:{0}'.format(et-1)),
        ]



actions = [MoveTo(), PutInCloset(), LeaveCloset(), OpenDoor(), CloseDoor()]
for action in actions:
    dom_str += '\n\n'
    dom_str += action.to_str()

# removes all the extra spaces
dom_str = dom_str.replace('            ', '')
dom_str = dom_str.replace('    ', '')
dom_str = dom_str.replace('    ', '')

print(dom_str)
f = open('namo_current_door.domain', 'w')
f.write(dom_str)
