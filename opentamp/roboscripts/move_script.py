from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.robot_state import RobotStateClient
import bosdyn.client.util
import time
#from bosdyn.client.image import ImageClient
#from PIL import Image
#import io

# Initializing robot
sdk = create_standard_sdk('move_forward_bot')
robot = sdk.create_robot('192.168.80.3')

bosdyn.client.util.authenticate(robot)
robot.start_time_sync(1.0)
robot.time_sync.wait_for_sync()

# Lease logic
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease_client.take()

# ESTOP logic
estop_client = robot.ensure_client(EstopClient.default_service_name)
estop_endpoint = EstopEndpoint(estop_client, 'GNClient', 9.0)

# robot clients
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
power_client = robot.ensure_client(PowerClient.default_service_name)

# command building
cmd = RobotCommandBuilder.synchro_velocity_command(v_x=1.0, v_y=0.0, v_rot=0.0)
robot_command_client.robot_command(command=cmd, end_time_secs=time.time() + 1.0)

""" #camera
image_client = robot.ensure_client(ImageClient.default_service_name)
sources = image_client.list_image_sources()
[source.name for source in sources]
#['back_depth', 'back_depth_in_visual_frame', 'back_fisheye_image', 'frontleft_depth', 'frontleft_depth_in_visual_frame', 'frontleft_fisheye_image', 'frontright_depth', 'frontright_depth_in_visual_frame', 'frontright_fisheye_image', 'left_depth', 'left_depth_in_visual_frame', 'left_fisheye_image', 'right_depth', 'right_depth_in_visual_frame', 'right_fisheye_image']

image_response = image_client.get_image_from_sources(["left_fisheye_image"])[0]
image = Image.open(io.BytesIO(image_response.shot.image.data))
image.show() """