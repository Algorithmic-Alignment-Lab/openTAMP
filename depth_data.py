import pickle
import numpy as np
import cv2
file = open('depth_image.obj', 'rb')
data = pickle.load(file)
file.close()

b = np.frombuffer(data.shot.image.data, dtype=np.uint16)
cv_depth = b.reshape(data.shot.image.rows,
                                data.shot.image.cols)



# Visual is a JPEG
cv_visual = cv2.imdecode(np.frombuffer(data.shot.image.data, dtype=np.uint8), -1)

 # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling
min_val = np.min(cv_depth)
max_val = np.max(cv_depth)
depth_range = max_val - min_val
depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)

# Write the image out.
filename = "depth_map.jpg"
cv2.imwrite(filename, depth_color)
print(filename)