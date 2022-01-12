import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import skimage
import cv2
# Root directory of the project
#ROOT_DIR = os.path.abspath("/")

# Import Mask RCNN
ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "defect_type_model")
# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
WEIGHTS_PATH = "defect_type_model/modelv1/object20211004T1938/mask_rcnn_object_0007.h5"  


CUSTOM_DIR = os.path.join(ROOT_DIR, "dataset/")

DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
# Inspect the model in training or inference modes values: 'inference' or 'training'
TEST_MODE = "inference"

CUSTOM_DIR = "dataset"

class CustomConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"
    
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 5 
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.60
config = CustomConfig()

with tf.device(DEVICE):
  model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

weights_path = WEIGHTS_PATH
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)  

class_names = ['BG', 'apple','hailed','rust','sun','punctured']

path = random.choice(os.listdir("dataset/testing"))
p = "dataset/testing/" + path
p = "dataset/testing/test17.jpg"
image = skimage.io.imread(p)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r1 = results[0]
img = visualize.display_instances(image, r1['rois'],r1['masks'], r1['class_ids'],class_names, r1['scores'], title="Predictions1")
print(type(results))
classes= r1['class_ids']
apple=0
hailed=0
rust=0
sun=0
punctured=0

for i in range(len(classes)):
    if (class_names[classes[i]])=='apple':
      apple+=1
    if (class_names[classes[i]])=='hailed':
      hailed+=1  
    if (class_names[classes[i]])=='rust':
      rust+=1  
    if (class_names[classes[i]])=='sun':
      sun+=1    
    if (class_names[classes[i]])=='punctured':
      punctured+=1      
print("Total Apples found", apple+hailed+rust+sun+punctured)
print("Total Fresh Apples", apple)
print("Hailed", hailed)
print("Rust", rust)
print("Sun", sun)
print("Punctured", punctured)

# for color change
# if class_id == 1: 
# masked_image = apply_mask(masked_image, mask, [1, 0, 0], alpha=1)
# elif class_id == 2:
# masked_image = apply_mask(masked_image, mask, [0, 1, 0], alpha=1)
# else
# masked_image = apply_mask(masked_image, mask, [0, 0, 1], alpha=1)

