# shape_data_reading.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import skimage
from skimage.draw import circle, polygon
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# %matplotlib inline 
coco_dir="/home/tianyu/mfs/google/lyg/coco/PythonAPI/"
sys.path.append(coco_dir)
sys.path.append("/home/tianyu/mfs/google/lyg/Mask_RCNN_test/Mask_RCNN/samples/coco/")  # To find local version

# os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Data Path
TRAIN_PATH = '/home/tianyu/mfs/google/lyg/seg_data/train/'
TEST_PATH = '/home/tianyu/mfs/google/lyg/seg_data/train'

TRAIN_PATH2 = TRAIN_PATH

# Get train and test IDs
# train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids_ = next(os.walk(TEST_PATH+"image/"))[2]
test_ids = [ test_ids_[i].replace(".jpg","") for i in range(len(test_ids_)) ]

IMAGE_DIR = TEST_PATH

# =================== 
class ShapesConfig(Config):
    """Configuration for training on the nuclei dataset.
    Derives from the base Config class and overrides values specific
    to the nuclei dataset.
    """
    # Give the configuration a recognizable name
    NAME = "whale"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1024

    RPN_ANCHOR_RATIOS = [0.5, 3, 5]
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 1

##
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    TRAIN_ROIS_PER_IMAGE = 512
    RPN_NMS_THRESHOLD = 0.3

    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.5 # may be smaller?
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 512
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 70
    
config = ShapesConfig()
config.display()

# # ## =================
# def get_ax(rows=1, cols=1, size=8):
#     """Return a Matplotlib Axes array to be used in
#     all visualizations in the notebook. Provide a
#     central point to control graph sizes.
    
#     Change the default size attribute to control the size
#     of rendered images
#     """
#     _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
#     return ax

## =================
class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    Extend the Dataset class and add a method to load the shapes dataset, 
    load_shapes(), and override the following methods:
    load_image()
    load_mask()
    image_reference()
    """
    def load_shapes(self,PATH):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # PATH=dir
        train_ids_ = next(os.walk(PATH+"/image/"))[2] 
        print("train_ids_,",train_ids_[0])
        train_ids = [ train_ids_[i].replace(".jpg","") for i in range(len(train_ids_)) ]
        count = len(train_ids)
        # print("count,",count)
        # print("train_ids,",train_ids[0])
        # Add classes
        # self.add_class("nuclei", 1, "nucleu")
        self.add_class("whales", 1, "whale")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). 
        for i in range(count):
            id_ = train_ids[i]
            # path_i = PATH + id_+ '/images/' + id_ + '.png'
            path_i = PATH+"/image/"+ id_+'.jpg'
            # self.add_image("nuclei", image_id=i, path=path_i)
            self.add_image("whales", image_id=i, path=path_i)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # image = imread(self.image_info[image_id]['path'])[:,:,0:3]
        # image = resize(image, (256, 256), mode='constant', preserve_range = True)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "whales":
            return info["whales"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        #image: /home/tianyu/mfs/google/lyg/seg_data/train/image/016791b76.jpg
        #mask : /home/tianyu/mfs/google/lyg/seg_data/train/mask/016791b76.png
        import_path=self.image_info[image_id]['path'].split('/')[8]
        if import_path !="stage1_test":
            # info = self.image_info[image_id]
            path = self.image_info[image_id]['path'].replace("/image/","/mask/")
            path = path.replace(".jpg",".png")
            # self.image_info[image_id]['path'].split('/')[0] +"/" + self.image_info[image_id]['path'].split('/')[1] + "/"
            # # print("load_mask_PATH:",PATH)
            # train_ids = next(os.walk(PATH))[1]
            # id_ = train_ids[image_id]
            # path = PATH + id_
            # count = len(next(os.walk(path + '/masks/'))[2])
            count = 1
            # mask = np.zeros([256, 256, count], dtype=np.uint8)
            # mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            # mask = np.expand_dims((skimage.io.imread(path,as_grey=True),axis=-1))
            mask = skimage.io.imread(path,as_grey=True)
            _,mask = cv2.threshold(img,0.45,1,cv2.THRESH_BINARY)
            # mask = mask.point(lambda x: 255 if x > 0.5 else 0)
            mask = np.expand_dims(mask,axis=-1)
            class_ids = np.array([1 for s in range(count)])
            return mask, class_ids.astype(np.int32)
        else:
            pass

## =================
# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(TRAIN_PATH)
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(TRAIN_PATH2)
dataset_val.prepare()

# # Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

## ================ Create model ================ ##

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


## ================ train model ================ ##
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=20, 
            layers='heads')

# # Train the layers from ResNet stage 4 and up
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=40, 
            layers='4+')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE/10,
            epochs=60, 
            layers="all")

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


## ================ Detection ================ ##
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
