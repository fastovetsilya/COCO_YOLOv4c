'''
Functions for generate_yolo.py
'''
from shutil import copyfile
import numpy as np
import cv2


def save_yolo_data(merged_annotations, obj_categories, image_id, images_path, output_path):
    '''
    Function to create and save YOLO files from merged annotations table. 
    Uses cbbx_from_poly() function. 
    '''

    # Compute output in YOLO format
    image_dims = [merged_annotations[merged_annotations['image_id'] == image_id]['height'].iloc[0],
                  merged_annotations[merged_annotations['image_id'] == image_id]['width'].iloc[0]]

    image_filename = merged_annotations[merged_annotations['image_id']
                                        == image_id]['file_name'].iloc[0].replace('.jpg', '.txt')
    img_obj_categ = np.array(
        merged_annotations[merged_annotations['image_id'] == image_id]['category_id'])
    img_obj_categ = np.array([obj_categories[1, np.where(
        obj_categories == img_obj_categ[i])[1][0]] for i in range(len(img_obj_categ))])

    # For original rbbx
    # image_bbx = list(merged_annotations[merged_annotations['image_id'] == image_id]['bbox'])

    # For cbbx
    image_polygon = np.array(
        merged_annotations[merged_annotations['image_id'] == image_id]['segmentation'])
    image_bbx = [cbbx_from_poly(el)
                  for el in image_polygon if type(el) is list]

    # Transform bbx to yolo format
    image_bbx_yolo = [[(bbx[0] / image_dims[1]) + (bbx[2] / image_dims[1]) / 2,
                       (bbx[1] / image_dims[0]) + (bbx[3] / image_dims[0]) / 2,
                       bbx[2] / image_dims[1],
                       bbx[3] / image_dims[0]
                       ] for bbx in image_bbx]

    # Iterate through objects
    for yolo_object in range(len(image_bbx_yolo)):
        # Save output in text format
        image_bbx_yolo_object = [
            img_obj_categ[yolo_object]] + image_bbx_yolo[yolo_object]
        image_bbx_yolo_object = str(image_bbx_yolo_object)
        image_bbx_yolo_object = image_bbx_yolo_object.replace(
            ',', '').replace('[', '').replace(']', '')

        if yolo_object == 0:
            file = open(output_path + image_filename, 'w')
            file.write(image_bbx_yolo_object)

        if yolo_object > 0:
            file = open(output_path + image_filename, 'a')
            file.write('\n')
            file.write(image_bbx_yolo_object)

        file.close()

    # Copy image to the output folder
    copyfile(images_path + image_filename.replace('.txt', '.jpg'),
             output_path + image_filename.replace('.txt', '.jpg'))


def cbbx_from_poly(obj_polygons):
    '''
    The function transforms polygons (masks) of the object instances 
    to the ellipse bounding boxes parameterized in terms of 
    the rectangles. 
    '''
    # Transfrom polygons to countours
    merged_contours = np.array(obj_polygons[0]).reshape(
        (-1, 1, 2)).astype(np.int32)
    
    # If more than 2 polygons for one instance, concatenate the contours
    if len(obj_polygons) > 1:
        for i in range(1, len(obj_polygons)):
            obj_polygon_contours = np.array(
                obj_polygons[i]).reshape((-1, 1, 2)).astype(np.int32)
            merged_contours = np.concatenate(
                (merged_contours, obj_polygon_contours), 0)
        
        # Perform convexHull on concatenated contours 
        merged_contours = cv2.convexHull(merged_contours)
    
    # Fit circle to the contours
    obj_fitted_circle = cv2.minEnclosingCircle(merged_contours)
    c_bbx = np.array([obj_fitted_circle[0][0] - obj_fitted_circle[1],
                      obj_fitted_circle[0][1] - obj_fitted_circle[1],
                      obj_fitted_circle[1] * 2,
                      obj_fitted_circle[1] * 2])
    
    return(c_bbx)
