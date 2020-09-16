import os
import glob
import pandas as pd
import numpy as np
from shutil import rmtree
import json
from generate_yolo_functions import save_yolo_data, cbbx_from_poly

# Initialize variables here
#############################
annotations_path_train = './annotations/instances_train2017.json'
annotations_path_val = './annotations/instances_val2017.json'
images_path_train = './train2017/'
images_path_val = './val2017/'
obj_categories = [53] #37 - sports ball; 53 - apple; 55 - orange
yolo_obj_folder_name = 'ball_appl_org'

#############################

# Clean the old and create new directories to save output
try:
    rmtree('./yolo_data/')
except FileNotFoundError:
    print('Directory already cleaned')

try:
    os.mkdir('./yolo_data')
    os.mkdir('./yolo_data/train')
    os.mkdir('./yolo_data/val')
except FileExistsError:
    print('Directories already created')

# Read annotations
# For train set
with open(annotations_path_train, 'r') as f:
    annotations_train = f.read()
# For val set
with open(annotations_path_val, 'r') as f:
    annotations_val = f.read()

# Convert annotations to dictionaries
annotations_train = json.loads(annotations_train)
annotations_val = json.loads(annotations_val)

# Create lists of annotations with sample objects
annotations_train_merged = pd.merge(pd.DataFrame(annotations_train['images']),
                                    pd.DataFrame(
                                        annotations_train['annotations']),
                                    left_on='id', right_on='image_id')
annotations_val_merged = pd.merge(pd.DataFrame(annotations_val['images']),
                                  pd.DataFrame(annotations_val['annotations']),
                                  left_on='id', right_on='image_id')

# Filter out category of objects
obj_categories = np.array([obj_categories,
                           [i for i in range(len(obj_categories))]])

annotations_train_merged = annotations_train_merged[annotations_train_merged['category_id'].isin(
    obj_categories[0, :])]
annotations_val_merged = annotations_val_merged[annotations_val_merged['category_id'].isin(
    obj_categories[0, :])]

# Save YOLO images and annotations for train data
for image_id in list(annotations_train_merged['image_id']):
    save_yolo_data(merged_annotations=annotations_train_merged, obj_categories=obj_categories,
                   image_id=image_id, images_path=images_path_train, output_path='./yolo_data/train/')

# Save YOLO images and annotations for validation data
for image_id in list(annotations_val_merged['image_id']):
    save_yolo_data(merged_annotations=annotations_val_merged, obj_categories=obj_categories,
                   image_id=image_id, images_path=images_path_val, output_path='./yolo_data/val/')

# Create lists of images in specified format
# Train data
train_images_list = glob.glob('./yolo_data/train/*.txt')
file = open('./yolo_data/train_' + yolo_obj_folder_name + '.txt', 'w')
file = open('./yolo_data/train_' + yolo_obj_folder_name + '.txt', 'a')
for s in range(len(train_images_list)):
    train_images_list[s] = train_images_list[s].replace('.txt', '.jpg').replace(
        './yolo_data/train/', 'data/obj_' + yolo_obj_folder_name + '/')
    file.write(train_images_list[s])
    file.write('\n')
file.close()

# Val data
val_images_list = glob.glob('./yolo_data/val/*.txt')
file = open('./yolo_data/test_' + yolo_obj_folder_name + '.txt', 'w')
file = open('./yolo_data/test_' + yolo_obj_folder_name + '.txt', 'a')
for s in range(len(val_images_list)):
    val_images_list[s] = val_images_list[s].replace('.txt', '.jpg').replace(
        './yolo_data/val/', 'data/obj_' + yolo_obj_folder_name + '/')
    file.write(val_images_list[s])
    file.write('\n')
file.close()
