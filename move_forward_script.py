from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.robot_state import RobotStateClient
import bosdyn.client.util
import time 


# Initializing robot
#sdk = create_standard_sdk('_bot')
#robot = sdk.create_robot("192.168.80.3")

sdk = bosdyn.client.create_standard_sdk('cast-single-ray')
robot = sdk.create_robot("192.168.80.3")
bosdyn.client.util.authenticate(robot)


robot.start_time_sync(1.0)
robot.time_sync.wait_for_sync()
# Lease logic
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease = lease_client.take()
lease_keep_alive = LeaseKeepAlive(lease_client)
robot.power_on(timeout_sec=20)

# ESTOP logic
estop_client = robot.ensure_client(EstopClient.default_service_name)
estop_endpoint = EstopEndpoint(estop_client, 'GNClient', 9.0)

# robot clients
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
power_client = robot.ensure_client(PowerClient.default_service_name)

# command building
#cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0.5, v_y=0.0, v_rot=0.0)
#robot_command_client.robot_command(command=cmd,end_time_secs=time.time() + 4.0)

# from bosdyn.client.image import ImageClient
# image_client = robot.ensure_client(ImageClient.default_service_name)
# sources = image_client.list_image_sources()
# print([source.name for source in sources])
# image_response = image_client.get_image_from_sources(
# [
# 'back_depth', 
# 'frontleft_depth', 
# 'frontright_depth', 
# 'left_depth', 
#  'right_depth'
# ])


# from bosdyn.client.local_grid import LocalGridClient

# #local_grid_client = robot.ensure_client(LocalGridClient.default_service_name)
# #print(local_grid_client.get_local_grid_types())
# #grid_response = local_grid_client.get_local_grids(['terrain', 'obstacle_distance'])
# #print(grid_response[0].local_grid.data)

# import pickle

# with open('depth_image.obj', 'wb') as f:
#     pickle.dump(list(image_response), f)

# with open('local_grid.obj', 'wb') as f:
#     pickle.dump(grid_response, f)

#ray casting 
import argparse

import bosdyn.client
import numpy as np
from bosdyn.api import ray_cast_pb2
from bosdyn.client.math_helpers import Vec3
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.util import add_base_arguments, setup_logging

def ray_intersection_type_strings():
    names = ray_cast_pb2.RayIntersection.Type.keys()
    return names[1:]


def ray_intersection_type_strings_to_enum(strings):
    retval = []
    type_dict = dict(ray_cast_pb2.RayIntersection.Type.items())
    print(type_dict)
    for enum_string in strings:
        retval.append(type_dict[enum_string])
    return retval

lidar_info = {
    "body_1": [(0.3, 0, 0), (1,0,0)], 
    "body_2": [(0.276, 0.195, -0.3), (0.924, 0.0585, 0)], 
    "body_3" : [(0.212,0.212,-0.3), (0.707, 0.707, 0)],
    "body_4": [(0.195, 0.276, -0.3), (0.0585, 0.924, 0)],
    "body_5" : [(0, 0.3, -0.3), (0, 1, 0)],
    "body_6" : [(-0.195, 0.276, -0.30), (-0.0585, 0.924, 0)], 
    "body_7" : [(-0.212, 0.212, -0.30), (-0.707, 0.707, 0)],
    "body_8" : [(-0.276, 0.195, -0.30), (-0.924, 0.0585, 0)],

    "body_9" : [(-0.3, 0, -0.3), (-1, 0, 0)], 
    "body_10" : [(-0.276, -0.195, -0.30), (-0.924,-0.0585, 0)],
    "body_11" : [(-0.212, -0.212, -0.30), (-0.707, -0.707, 0)],
    "body_12" : [(-0.195, -0.276, -0.30), (-0.0585, -0.924 0)],

    "body_13" : [(0, -0.3, -0.3), (0, -1, 0)],
    "body_14" : [(0.195, -0.276, -0.3), (0.0585, -0.924, 0)],
    "body_15" : [(0.212, -0.212, -0.3), (0.707, -0.707, 0)],
    "body_16" : [(0.276, -0.195, -0.3), (0.924, -0.0585, 0)],
    "body_17" : [(0,1.1, -0.3), (0, 1, 0)],

        "rf_1" :[(0., 0.35, -0.3), (0, 1, 0)],
        "rf_2" :[(-0.05, 0.35, -0.3), (-1, 1, 0)],
        "rf_3" :[(-0.055, 0.15, -0.3), (-1, 0, 0)],
        "rf_4" :[(-0.055, -0.15, -0.3), (-1, 0, 0)],
        "rf_5" :[(0.05, 0.35, -0.3), (1, 1, 0)],
        "rf_6" :[(-0.055, 0, -0.3), (-1, 0, 0)],
        "rf_7" :[(0, 0.35, -0.3) , (1, 2, 0)],
        "rf_8" :[(-0.055, 0.1, -0.3) , (-1, 0.5, 0)],
        "rf_9" :[(-0.05, 0.3, -0.3) , (-1, 0.5, 0)],
        "rf_10" :[(-0.055, -0.3, -0.3) , (-1, 0, 0)],
        "rf_11" :[(-0.055,  0.3, -0.3) , (-1, 0, 0)],
        "rf_12" :[(-0.055, -0.1, -0.3) , (-1, 2, 0)],
        "rf_13" :[(-0.055, 0.2, -0.3), (-1, 2, 0)],
        "rf_14" :[(-0.055, 0.35, -0.3), (-3, 1, 0)],
        "rf_15" :[(-0.055, 0.35, -0.3), (-1, 3, 0)],
        "rf_16" :[(-0.055, 0.25, -0.3), (-2, 1, 0)],
        "rf_17" :[(-0.055, 0.2, -0.3), (-2, 1, 0)],
        "rf_18" :[(-0.055, 0.15, -0.3), (-2, 1, 0)],
        "rf_19" :[(-0.055, -0.05, -0.3), (-2, 1, 0)],
        "rf_20" :[(-0.055, -0.15, -0.3), (-2, 1, 0)],
        "rf_21" :[(-0.055, 0.15, -0.3), (-1, 1, 0)],
        "rf_22" :[(-0.055, 0.35, -0.3), (-2, 1, 0)],
        "rf_23" :[(0., 0.35, -0.3), (1, 2, 0)],
        "rf_24" :[(0., 0.35, -0.3), (-1, 2, 0)],
        "rf_25" :[(0., 0.35, -0.3), (2, 1, 0)],
        "rf_26" :[(0., 0.35, -0.3), (-2,1, 0)],
        "lf_1" :[(0., 0.35, -0.3), (0, 1, 0)],
        "lf_2" :[(0.05, 0.35, -0.3), (1, 1, 0)],
        "lf_3" :[(0.055, 0.15, -0.3), (1, 0, 0)],
        "lf_4" :[(0.055, -0.15, -0.3), (1, 0, 0)],
        "lf_5" :[(-0.05, 0.35, -0.3), (-1, 1, 0)],
        "lf_6" :[(0.055, 0, -0.3), (1, 0, 0)],
        "lf_7" :[(0, 0.35, -0.3) , (-1, 2, 0)],
        "lf_8" :[(0.055, 0.1, -0.3) , (1, 0.5, 0)],
        "lf_9" :[(0.05, 0.3, -0.3) , (1, 0.5, 0)],
        "lf_10" :[(0.055, -0.3, -0.3) , (1, 0, 0)],
        "lf_11" :[(0.055,  0.3, -0.3) , (1, 0, 0)],
        "lf_12" :[(0.055, -0.1, -0.3) , (1, 2, 0)],
        "lf_13" :[(0.055, 0.2, -0.3), (1, 2, 0)],
        "lf_14" :[(0.055, 0.35, -0.3), (3, 1, 0)],
        "lf_15" :[(0.055, 0.35, -0.3), (1, 3, 0)],
        "lf_16" :[(0.055, 0.25, -0.3), (2, 1, 0)],
        "lf_17" :[(0.055, 0.2, -0.3), (2, 1, 0)],
        "lf_18" :[(0.055, 0.15, -0.3), (2, 1, 0)],
        "lf_19" :[(0.055, -0.05, -0.3), (2, 1, 0)],
        "lf_20" :[(0.055, -0.15, -0.3), (2, 1, 0)],
        "lf_21" :[(0.055, 0.15, -0.3), (1, 1, 0)],
        "lf_22" :[(0.055, 0.35, -0.3), (2, 1, 0)],
        "lf_23" :[(0., 0.35, -0.3), (1, 2, 0)],
        "lf_24" :[(0., 0.35, -0.3), (1, 2, 0)],
        "lf_25" :[(0., 0.35, -0.3), (-2, 1, 0)],
        "lf_26" :[(0., 0.35, -0.3), (2,1, 0)]}

rc_client = robot.ensure_client(RayCastClient.default_service_name)
raycast_types = ray_intersection_type_strings_to_enum(['TYPE_GROUND_PLANE'])
for body in lidar_info:
    ray_origin = Vec3(lidar_info[body][0][0], lidar_info[body][0][1], lidar_info[body][0][2])
    ray_direction = Vec3(lidar_info[body][1][0], lidar_info[body][1][1], lidar_info[body][1][2])
    ray_frame_name = "body"
    min_distance = 2.5

    print("Raycasting from position: {}".format(ray_origin))
    print("Raycasting in direction: {}".format(ray_direction))

    response = rc_client.raycast(ray_origin, ray_direction, raycast_types,
                                min_distance=min_distance, frame_name=ray_frame_name)
    
for idx, hit in enumerate(response.hits):
        print('Hit {}:'.format(idx))
        hit_position = Vec3.from_proto(hit.hit_position_in_hit_frame)
        print('\tPosition: {}'.format(hit_position))
        hit_type_str = ray_cast_pb2.RayIntersection.Type.keys()[hit.type]
        print('\tType: {}'.format(hit_type_str))
        print('\tDistance: {}'.format(hit.distance_meters))


    
cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0.5, v_y=0.0, v_rot=0.0)
robot_command_client.robot_command(command=cmd,end_time_secs=time.time() + 4.0)


from PIL import Image
#import io
#image = Image.open(io.BytesIO(image_response.shot.image.data))
#image.show()

