# COCO_YOLOv4c
MS COCO ground truth preparation for [YOLOv4c](https://github.com/fastovetsilya/darknet) training and validation. 

# The problem
The original MS COCO dataset annotations include coordinates of rectangular bounding boxes 
as well as polygons (masks) for instance segmentation task for every object. 
For training YOLOv4c we need to enclose the instances of the objects into ellipses instead of the rectangles. The ellipses are parameterized in terms of the rectangles that enclose them. 
Thus, the original bounding boxes is not a correct way of training and testing the model, and in the current project we attempt to transform masks into more suitable bounding boxes. 
# The solution
To simplify the problem, we decided to generate square bounding boxes from the masks. To do this, we fitted a minimum enclosing circle for masks of each instance of the objects. In some cases, the objects were occluded, and the instances contained more than one mask. For this case, a [Convex Hull](https://en.wikipedia.org/wiki/Convex_hull) algorithm was applied to the masks of that instance, and the circle was fitter to the resulting polygon.

The idea of the proposed solution is shown in the image below. The blue bounding box is the original rectangular bounding box from the COCO dataset. The white countour is the mask for that instance of the object. The red is the generated bounding box corresponding to the circular bounding box that encloses the object. 

<img src="https://github.com/fastovetsilya/COCO_YOLOv4c/blob/master/examples/example_1.png" width="640" height="425">

# The results
The comparison between the original bounding boxes and the bounding boxes after the transformation is shown in the images below. In most of the cases, these bounding boxes are more suitable for training and validation of YOLOv4c compared to the original ones. However, better results could be achieved if the data was labeled with ellipses or circles by hand. 

<img src="https://github.com/fastovetsilya/COCO_YOLOv4c/blob/master/examples/example_2.png" width="1132" height="539">
<img src="https://github.com/fastovetsilya/COCO_YOLOv4c/blob/master/examples/example_3.png" width="1132" height="539">





