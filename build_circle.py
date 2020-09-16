'''
A demonstration of the method
'''

import numpy as np
import cv2

# Play with a sample poligon
img_path = './val2017/000000042296.jpg'
obj_polygon = np.array([256.37, 218.07, 268.01, 221.39, 275.08, 226.38, 281.31, 232.62, 289.21, 238.85, 300.85, 252.57, 304.59, 270.86, 307.09, 287.91, 303.76, 304.12, 298.77, 310.77, 274.66, 321.58, 254.29, 322.0, 239.74, 322.0, 224.78, 322.41, 215.63, 320.75, 206.49, 314.51, 193.6, 298.72, 188.61, 290.4, 188.19, 272.94, 194.01, 255.48, 199.0, 243.01, 212.31, 231.79, 218.96, 227.21, 226.02, 223.06, 234.75, 220.56, 246.39, 218.9, 255.12, 218.9, 257.62, 218.9])

r_bbx = np.array([188.19, 218.07, 118.9, 104.34], dtype = 'int')
r_bbx = r_bbx.astype('int')

# Load image
img = cv2.imread(img_path)

# Fit minimum circle for the polygon
obj_polygon_contours = np.array(obj_polygon).reshape((-1,1,2)).astype(np.int32)
obj_fitted_circle = cv2.minEnclosingCircle(obj_polygon_contours)

# Generate updated r_bbx
r_bbx_new = np.array([obj_fitted_circle[0][0] - obj_fitted_circle[1], 
                      obj_fitted_circle[0][1] - obj_fitted_circle[1],
                      obj_fitted_circle[1] * 2, 
                      obj_fitted_circle[1] * 2], dtype = 'int')

# Draw 
cv2.rectangle(img, (r_bbx[0], r_bbx[1]), 
              (r_bbx[0] + r_bbx[2], r_bbx[1] + r_bbx[3]), 
              (255, 0, 0), 2)
cv2.circle(img, (int(obj_fitted_circle[0][0]), int(obj_fitted_circle[0][1])),
           int(obj_fitted_circle[1]), 
           (0, 0, 255), 2)
cv2.rectangle(img, (r_bbx_new[0], r_bbx_new[1]), 
              (r_bbx_new[0] + r_bbx_new[2], r_bbx_new[1] + r_bbx_new[3]), 
              (0, 0, 255), 2)

cv2.drawContours(img,[obj_polygon_contours], 0, (255,255,255), 2)


# Display image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

