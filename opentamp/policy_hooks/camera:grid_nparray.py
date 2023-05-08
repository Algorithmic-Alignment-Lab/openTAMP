from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.image import ImageClient
import bosdyn.client.util
import time 
import pickle
import numpy as np
import cv2
import bosdyn
import bosdyn.client
import bosdyn.client
import numpy as np
from bosdyn.api import ray_cast_pb2
from bosdyn.client.math_helpers import Vec3
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.util import add_base_arguments, setup_logging
import argparse


# Initializing robot
sdk = create_standard_sdk('move_forward_bot')
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

timeout = time.time() + 10
while time.time() < timeout:


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


    rc_client = robot.ensure_client(RayCastClient.default_service_name)
    raycast_types = ray_intersection_type_strings_to_enum(['TYPE_GROUND_PLANE'])
    ray_origin = Vec3(0,0.35,-0.3)
    ray_direction = Vec3(1,2,0)
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

    # capture depth image
    # image_client = robot.ensure_client(ImageClient.default_service_name)
    # sources = image_client.list_image_sources()
    # print([source.name for source in sources])
    # image_response = image_client.get_image_from_sources(
    # [
    # 'back_depth', 
    # 'frontleft_depth', 
    # 'frontright_depth', 
    # 'left_depth', 
    # 'right_depth'
    # ])


    # # capture local grid
    # local_grid_client = robot.ensure_client(LocalGridClient.default_service_name)
    # grid_response = local_grid_client.get_local_grids(['terrain'])

    # # process depth image
    # #save first
    # with open('depth_image.obj', 'wb') as f:
    #     pickle.dump(list(image_response), f)

    # b = np.frombuffer(image_response.shot.image.data, dtype=np.uint16)
    # cv_depth = b.reshape(image_response.shot.image.rows,
    #                                 image_response.shot.image.cols)

    # # process local grid

    # with open('local_grid.obj', 'wb') as f:
    #     pickle.dump(grid_response, f)

    # rle_counts = []
    # new_data = []
    # for i in grid_response:
    #     new_data[i] = grid_response[i]*rle_counts[i]
    # b = np.frombuffer(new_data, dtype=np.int16)

  
    # print("camera", cv_depth)
    # print("grid response", b)
    time.sleep(1000)