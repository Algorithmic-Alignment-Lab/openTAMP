import pickle
import numpy as np
import cv2
import bosdyn
import bosdyn.client
file = open('../Downloads/depth_image.obj', 'rb')
data = pickle.load(file)
file.close()

#print(data)
#frame_tree_snapshot = data.shot.transforms_snapshot
#print(frame_tree_snapshot)
#print(bosdyn.client.frame_helpers.get_a_tform_b(frame_tree_snapshot, 'body', 'back', validate=True))


#['back_depth', 'back_depth_in_visual_frame', 'back_fisheye_image', 'frontleft_depth', 'frontleft_depth_in_visual_frame', 
 #'frontleft_fisheye_image', 'frontright_depth', 'frontright_depth_in_visual_frame', 'frontright_fisheye_image', 'hand_color_image',
 #  'hand_color_in_hand_depth_frame', 'hand_depth', 'hand_depth_in_hand_color_frame', 'hand_image', 'left_depth', 'left_depth_in_visual_frame',
 #    'left_fisheye_image', 'right_depth', 'right_depth_in_visual_frame', 'right_fisheye_image']



cameras = [
'back', 
'frontleft', 
'frontright', 
'left', 
 'right'
]
for camera_index in range(len(cameras)):
    frame_tree_snapshot = data[camera_index].shot.transforms_snapshot
   # frame_tree_snapshot = data[]
    bosdyn.client.frame_helpers.get_a_tform_b(frame_tree_snapshot, 'body', cameras[camera_index], validate=True)
    print(bosdyn.client.frame_helpers.get_a_tform_b(frame_tree_snapshot, 'body', cameras[camera_index], validate=True))