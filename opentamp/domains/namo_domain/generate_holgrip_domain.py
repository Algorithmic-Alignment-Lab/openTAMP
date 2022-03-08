dom_str = """
# AUTOGENERATED. DO NOT EDIT.
# Configuration file for CAN domain. Blank lines and lines beginning with # are filtered out.

# implicity, all types require a name
Types: Can, Target, RobotPose, Robot, Grasp, Obstacle, Rotation

# Define the class location of each non-standard attribute type used in the above parameter type descriptions.

Attribute Import Paths: RedCircle core.util_classes.items, BlueCircle core.util_classes.items, GreenCircle core.util_classes.items, Vector1d core.util_classes.matrix, Vector2d core.util_classes.matrix, Wall core.util_classes.items, NAMO core.util_classes.robots

Predicates Import Path: core.util_classes.namo_grip_predicates

"""

prim_pred_str = 'Primitive Predicates: geom, Can, RedCircle; pose, Can, Vector2d; geom, Target, BlueCircle; value, Target, Vector2d; value, RobotPose, Vector2d; gripper, RobotPose, Vector1d; geom, RobotPose, BlueCircle; geom, Robot, NAMO; pose, Robot, Vector2d; gripper, Robot, Vector1d; value, Grasp, Vector2d; geom, Obstacle, Wall; pose, Obstacle, Vector2d; value, Rotation, Vector1d; theta, Robot, Vector1d; theta, RobotPose, Vector1d; vel, RobotPose, Vector1d; acc, RobotPose, Vector1d; vel, Robot, Vector1d; acc, Robot, Vector1d'
dom_str += prim_pred_str + '\n\n'

der_pred_str = 'Derived Predicates: At, Can, Target; AtInit, Can, Target; RobotAt, Robot, RobotPose; InGripper, Robot, Can, Grasp; Obstructs, Robot, Can, Can, Can; ObstructsHolding, Robot, Target, Target, Can, Can; WideObstructsHolding, Robot, Target, Target, Can, Can; StationaryRot, Robot; Stationary, Can; RobotStationary, Robot; StationaryNEq, Can, Can; IsMP, Robot; StationaryW, Obstacle; Collides, Can, Obstacle; RCollides, Robot, Obstacle; GripperClosed, Robot; Near, Can, Target;  RobotAtGrasp, Robot, Can, Grasp; RobotWithinReach, Robot, Target; RobotNearGrasp, Robot, Can, Grasp; RobotWithinBounds, Robot; WideObstructs, Robot, Can, Can, Can; AtNEq, Can, Can, Target; PoseCollides, RobotPose, Obstacle; TargetCollides, Target, Obstacle; TargetGraspCollides, Target, Obstacle, Grasp; TargetCanGraspCollides, Target, Can, Grasp; CanGraspCollides, Can, Obstacle, Grasp; HLPoseUsed, RobotPose; HLAtGrasp, Robot, Can, Grasp; RobotPoseAtGrasp, RobotPose, Target, Grasp; HLPoseAtGrasp, RobotPose, Target, Grasp; RobotRetreat, Robot, Grasp; RobotApproach, Robot, Grasp; LinearRetreat, Robot; LinearApproach, Robot; InGraspAngle, Robot, Can; NearGraspAngle, Robot, Can; TargNearGraspAngle, Robot, Target; AroundGraspAngle, Robot, Can; ThetaDirValid, Robot; ForThetaDirValid, Robot; RevThetaDirValid, Robot; ScalarVelValid, Robot; HLGraspFailed, Can; HLTransferFailed, Can, Target; HLPlaceFailed, Target'
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
        self.timesteps = 20
        et = self.timesteps - 1
        self.args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?gp - RobotPose ?g - Grasp ?end - Target)' 
        self.pre = [\
                ('(Near ?can ?target)', '0:-1'),
                #('(At ?can ?target)', '{}:{}'.format(1, et)),
                # ('(forall (?w - Obstacle) (not (Collides ?can ?w)))', '0:0'),
                ('(forall (?obj - Can) (not (NearGraspAngle ?robot ?obj)))', '0:0'),
                # ('(forall (?w - Obstacle) (not (CanGraspCollides ?can ?w ?g)))', '0:0'),
                ('(not (GripperClosed ?robot))', '1:{0}'.format(et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (StationaryW ?w))', '0:{0}'.format(et-1)),
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '1:{0}'.format(et-2)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?can ?can ?obj)))', '0:0'),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?can ?can ?obj)))', '1:{0}'.format(et-4)),
                #('(forall (?obj - Can) (not (Obstructs ?robot ?can ?can ?obj)))', '{0}:{1}'.format(et-3, et-2)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?target ?target ?obj ?can)))', '{0}:{1}'.format(et-4, et-1)),
        ]
        self.eff = [\
                ('(NearGraspAngle ?robot ?can)', '{0}:{1}'.format(et, et)),
                ('(InGraspAngle ?robot ?can)', '{0}:{1}'.format(et, et)),
                ('(forall (?obj - Can / ?can) (not (NearGraspAngle ?robot ?obj)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
                ('(StationaryRot ?robot)', '{0}:{1}'.format(et-4, et-1)),
                ('(ForThetaDirValid ?robot)', '{0}:{1}'.format(et-4, et-1)),
        ]

class Transfer(Action):
    def __init__(self):
        self.name = 'transfer'
        self.timesteps = 20
        et = self.timesteps - 1
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can ?t - Target ?g - Grasp ?init - Target)'
        self.pre = [\
                ('(Near ?c ?init)', '0:-1'),
                ('(forall (?obj - Can) (not (AtInit ?obj ?t)))', '0:-1'),
                ('(forall (?obj - Can) (not (At ?obj ?t)))', '0:0'),
                ('(forall (?obj - Can) (not (Near ?obj ?t)))', '0:0'),
                #('(forall (?w - Obstacle) (not (Collides ?c ?w)))', '0:0'),
                #('(At ?c ?init)', '1:1'),
                #('(StationaryRot ?robot)', '0:0'),
                #('(RobotStationary ?robot)', '0:0'),
                #('(forall (?obj - Can) (not (TargetCanGraspCollides ?t ?obj ?g)))', '0:0'),
                #('(forall (?w - Obstacle) (not (TargetGraspCollides ?t ?w ?g)))', '0:0'),
                ('(NearGraspAngle ?robot ?c)', '0:0'),
                #('(NearGraspAngle ?robot ?c)', '1:{}'.format(et-1)),
                ('(NearGraspAngle ?robot ?c)', '{}:{}'.format(et, et)),
                #('(InGraspAngle ?robot ?c)', '0:1'),
                # ('(not (GripperClosed ?robot))', '0:0'),
                ('(GripperClosed ?robot)', '1:{0}'.format(et-1)),
                # ('(InGraspAngle ?robot ?c)', '1:{0}'.format(et-1)),
                #('(InGraspAngle ?robot ?c)', '1:{0}'.format(1)),
                ('(InGraspAngle ?robot ?c)', '{0}:{0}'.format(et)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?t ?t ?obj ?c)))', '0:{0}'.format(0)),
                ('(forall (?obj - Can) (not (WideObstructsHolding ?robot ?t ?t ?obj ?c)))', '2:{0}'.format(et-2)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?t ?t ?obj ?c)))', '1:{0}'.format(et-2)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?t ?t ?obj ?c)))', '{0}:{0}'.format(et-1)),
                # ('(forall (?obj - Can ) (not (Obstructs ?robot ?init ?t ?obj)))', '0:0'),
                ('(forall (?obj - Can) (StationaryNEq ?obj ?c))', '0:{0}'.format(et-1)), 
                ('(forall (?w - Obstacle) (StationaryW ?w))', '0:{0}'.format(et-1)), 
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '2:{0}'.format(et-1)),
                ('(StationaryRot ?robot)', '{0}:{1}'.format(et-3, et-1)),
                ('(ForThetaDirValid ?robot)', '{0}:{1}'.format(et-3, et-1)),
                ('(not (GripperClosed ?robot))', '{0}:{1}'.format(et, et-1)),
               ]

        self.eff = [\
                ('(At ?c ?t)', '{0}:{1}'.format(et, et)),
                ('(forall (?obj - Can / ?c) (not (NearGraspAngle ?robot ?obj)))', '{0}:{1}'.format(et, et-1)),
                #('(NearGraspAngle ?robot ?c)', '{0}:{0}'.format(et)),
                ('(Near ?c ?t)', '{0}:{0}'.format(et)),
                ('(forall (?targ - Target / ?t) (not (Near ?c ?targ)))', '{}:{}'.format(et,et-1)),
                ('(forall (?targ - Target / ?t) (not (At ?c ?targ)))', '{}:{}'.format(et,et-1)),
                ('(not (Near ?c ?init))', '{0}:{1}'.format(et, et-1)),
                ('(not (At ?c ?init))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?obj ?obj ?c)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?obj ?obj ?c)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (forall (?targ - Target) (not (WideObstructsHolding ?robot ?targ ?targ ?c ?obj))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (forall (?targ - Target) (not (ObstructsHolding ?robot ?targ ?targ ?c ?obj))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
                ('(TargNearGraspAngle ?robot ?t)', '{}:{}'.format(et, et)),
        ]

class Place(Action):
    def __init__(self):
        self.name = 'place'
        self.timesteps = 7
        et = self.timesteps - 1
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can ?t - Target ?g - Grasp ?init - Target)'
        self.pre = [\
                #('(At ?c ?t)', '0:0'),
                #('(Near ?c ?t)', '0:0'),
                #('(Near ?c ?t)', '{}:{}'.format(et, et)),
                #('(At ?c ?t)', '1:{}'.format(et)),
                # ('(forall (?obj - Can) (not (TargetCanGraspCollides ?t ?obj ?g)))', '0:0'),
                # ('(forall (?w - Obstacle) (not (TargetGraspCollides ?t ?w ?g)))', '0:0'),
                ('(not (GripperClosed ?robot))', '1:{0}'.format(et-1)),
                ('(TargNearGraspAngle ?robot ?t)', '0:0'),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?t ?t ?obj ?c)))', '0:{0}'.format(et-3)),
                #('(forall (?obj - Can) (not (ObstructsHolding ?robot ?t ?t ?obj ?c)))', '0:{0}'.format(0)),
                ('(forall (?obj - Can ) (not (Obstructs ?robot ?c ?c ?obj)))', '4:{0}'.format(et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '0:{0}'.format(et-1)), 
                ('(forall (?w - Obstacle) (StationaryW ?w))', '0:{0}'.format(et-1)), 
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '1:{0}'.format(et-1)),
                #('(forall (?obj - Can) (not (HLGraspFailed ?obj )))', '{0}:{0}'.format(et,et-1)),
                #('(forall (?obj - Can) (forall (?targ - Target) (not (HLTransferFailed ?obj ?targ))))', '{0}:{0}'.format(et,et-1)),
                #('(forall (?targ - Target) (not (HLPlaceFailed ?targ)))', '{0}:{0}'.format(et,et-1)),
                # ('(LinearRetreat ?robot)', '0:{0}'.format(et-1)),
                # ('(StationaryRot ?robot)', '0:{0}'.format(et-2)),
                ('(StationaryRot ?robot)', '0:{0}'.format(1)),
                #('(RevThetaDirValid ?robot)', '0:{0}'.format(et-2)),
                #('(RevThetaDirValid ?robot)', '0:{0}'.format(1)),
                #('(RobotStationary ?robot)', '{0}:{0}'.format(0)),
                #('(RobotStationary ?robot)', '{0}:{0}'.format(et-1)),
                ]
        self.eff = [\
                ('(forall (?obj - Can) (not (NearGraspAngle ?robot ?obj)))', '{0}:{1}'.format(et, et)),
                ('(forall (?targ - Target) (not (TargNearGraspAngle ?robot ?targ)))', '{}:{}'.format(et, et)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
        ]


actions = [MoveTo(), Transfer(), Place()]
for action in actions:
    dom_str += '\n\n'
    dom_str += action.to_str()

# removes all the extra spaces
dom_str = dom_str.replace('            ', '')
dom_str = dom_str.replace('    ', '')
dom_str = dom_str.replace('    ', '')

print(dom_str)
f = open('namo_current_holgrip.domain', 'w')
f.write(dom_str)