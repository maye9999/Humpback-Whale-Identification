#CUDA_VISIBLE_DEVICES=2,3 nohup python -u newnet_MMLDA.py > train_logs/input64x128_modeliter3_wd1e-5_lr1e-3_lesslessaugment_epoch1dataset.out 2>&1 &
#CUDA_VISIBLE_DEVICES=0,2,3 python newnet_MMLDA.py

import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import keras
from keras import layers
from keras import regularizers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, GlobalAveragePooling2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Lambda, Add
from keras.layers import Concatenate, GlobalMaxPooling2D, Reshape
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras_imagenet_models.resnet50 import ResNet50
from keras_imagenet_models.densenet import DenseNet121
import tensorflow as tf
from scipy.io import loadmat

import keras.backend as K
from keras.models import Sequential

import warnings
from collections import defaultdict

labels = defaultdict(int)
total_num = 0
dataset_per_epoch = 1

with open('../data-raw/train.csv') as fin:
    for line in fin:
        if line.startswith('Image'):
            continue
        l = line.strip().split(',')
        labels[l[1]] += 1
        total_num += 1

warnings.simplefilter("ignore", category=DeprecationWarning)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('mean_var', 10, 'parameter in MMLDA')
tf.app.flags.DEFINE_integer('num_dense', 512, 'num dense in MMLDA')

#input_H = 256
#input_W = 512
input_H = 64
input_W = 128

batch_size = 128
epochs = 1000

train_df = pd.read_csv("../train-new.csv")
val_df = pd.read_csv("../val.csv")

#Drop new_whale and 1-sample labels for training data
indexs = []
for i in range(len(train_df['Id'])):
    if train_df['Id'][i]=='new_whale':
        indexs += [i]
train_df = train_df.drop(indexs)

#Drop new_whale and 1-sample labels for val data
indexs = []
for i in range(len(val_df['Id'])):
    if val_df['Id'][i]=='new_whale':
        indexs += [i]
val_df = val_df.drop(indexs)

num_class = 2931
train_df.index = range(len(train_df['Id']))
val_df.index = range(len(val_df['Id']))


#Load means in MMLDA
kernel_dict = loadmat('kernelpara/dimmeans'+str(FLAGS.num_dense)+'_meanvar'+str(FLAGS.mean_var)+'_class'+str(num_class)+'.mat')
mean_logits = kernel_dict['mean_logits'] #num_class X num_dense
mean_logits = tf.constant(mean_logits,dtype=tf.float32)


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 500:
        lr *= 1e-1
    elif epoch > 800:
        lr *= 1e-2
    print('Learning rate: ', lr)
    return lr

def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 6), activation='relu', **kwargs)(y)  # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    y = Add()([x, y])  # Add the bypass connection
    y = Activation('relu')(y)
    return y


def build_model(l2, activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}
    inp = Input(shape=(input_H, input_W, 3))  # 256x512x1
    x = Conv2D(64, (6, 12), strides=2, activation='relu', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 64x128x64
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 6), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 32x64x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 32x64x128
    for _ in range(3):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 16x32x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 16x32x256
    for _ in range(3):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 8x16x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 8x16x384
    for _ in range(3):
        x = subblock(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 4x8x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 4x8x512
    for _ in range(3):
        x = subblock(x, 128, **kwargs)

    x = GlobalAveragePooling2D()(x)  # 512
    branch_model = Model(inp, x)
    return branch_model
    

def dot_loss(y_true, y_pred):
    return -tf.reduce_sum(y_pred * y_true, axis=-1) #batch_size X 1

#MMLDA prediction function
def MMLDA_layer(x, means=mean_logits):
    #x_shape = batch_size X num_dense
    x_expand = tf.tile(tf.expand_dims(x,axis=1),[1,num_class,1]) #batch_size X num_class X num_dense
    mean_expand = tf.expand_dims(means,axis=0) #1 X num_class X num_dense
    logits = -tf.norm(x_expand - mean_expand, ord='euclidean', axis=-1) #batch_size X num_class

    logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True) #Avoid numerical rounding
    logits = logits - tf.log(tf.reduce_sum(tf.exp(logits), axis=-1, keepdims=True)) #Avoid numerical rounding
    return logits




model_nonlinear = build_model(1e-5)
x = model_nonlinear.output
x = Dense(FLAGS.num_dense, activation=None)(x) #x = ReLU(W1x+b1)
new_layer = Lambda(MMLDA_layer)
predictions = new_layer(x)
base_model = Model(inputs=model_nonlinear.input, outputs=predictions)

model = keras.utils.multi_gpu_model(base_model, gpus=2, cpu_merge=True, cpu_relocation=False)

model.compile(loss=dot_loss, optimizer=keras.optimizers.Adam(lr_schedule(0)), metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])
model.summary()


# Prepare model model saving directory.
filepath_dir = 'trained_models/input64x128_modeliter3_wd1e-5_lr1e-3_lesslessaugment_epoch1dataset'
save_dir = os.path.join(os.getcwd(), filepath_dir)
model_name = 'model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_top_k_categorical_accuracy', verbose=2,
                 save_best_only=True, save_weights_only=True,
                 mode='max', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ParallelModelCheckpoint(base_model, filepath=filepath)

# Prepare callbacks for model saving and for learning rate adjustment.
lr_scheduler = LearningRateScheduler(lr_schedule)


callbacks = [checkpoint,lr_scheduler]


# train_datagen = ImageDataGenerator(
#     # randomly shift images horizontally
#     width_shift_range=0.3,
#     # randomly shift images vertically
#     height_shift_range=0.3,
#     # set range for random shear
#     shear_range=0.3,
#     # set range for random zoom
#     zoom_range=0.3,
#     # set range for random channel shifts
#     channel_shift_range=0.,
#     # set mode for filling points outside the input boundaries
#     rotation_range=45,
#     fill_mode='constant',
#     # value used for fill_mode = "constant"
#     cval=0.5,
#     # randomly flip images
#     horizontal_flip=True,
#     # randomly flip images
#     vertical_flip=True,
#     # set rescaling factor (applied before any other transformation)
#     rescale=1./255)

train_datagen = ImageDataGenerator(
    # randomly shift images horizontally
    width_shift_range=0.2,
    # randomly shift images vertically
    height_shift_range=0.2,
    # set range for random shear
    shear_range=0.2,
    # set range for random zoom
    zoom_range=0.2,
    # set range for random channel shifts
    channel_shift_range=0.,
    # set mode for filling points outside the input boundaries
    rotation_range=10,
    fill_mode='constant',
    # value used for fill_mode = "constant"
    cval=0.5,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=True,
    # set rescaling factor (applied before any other transformation)
    rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='/home/tianyu/data-crop',
        x_col='Image',
        y_col='Id',
        target_size=(input_H, input_W),
        batch_size=batch_size,
        class_mode='categorical',
        interpolation='bilinear',
        shuffle=True)

classes_val = [0] * train_generator.num_classes
for k in train_generator.class_indices.keys():
    classes_val[train_generator.class_indices[k]]=k

validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory='/home/tianyu/data-crop',
        x_col='Image',
        y_col='Id',
        classes=classes_val,
        target_size=(input_H, input_W),
        batch_size=batch_size,
        class_mode='categorical',
        interpolation='bilinear',
        shuffle=False)


model.fit_generator(
    train_generator,
    steps_per_epoch=dataset_per_epoch*train_generator.samples/train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size,
    verbose=2,
    workers=10,
    use_multiprocessing=True,
    callbacks=callbacks)