#CUDA_VISIBLE_DEVICES=1 nohup python -u CNN_baseline.py > train_logs/CNN_baseline.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 python newnet_baseline.py

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
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Lambda, Add
from keras.layers import Concatenate, GlobalMaxPooling2D, Reshape
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras_imagenet_models.resnet50 import ResNet50
from keras_imagenet_models.densenet import DenseNet121
import tensorflow as tf

import keras.backend as K
from keras.models import Sequential

import warnings
from collections import defaultdict

labels = defaultdict(int)
total_num = 0

with open('../data-raw/train.csv') as fin:
    for line in fin:
        if line.startswith('Image'):
            continue
        l = line.strip().split(',')
        labels[l[1]] += 1
        total_num += 1

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('mean_var', 10, 'parameter in MMLDA')
tf.app.flags.DEFINE_integer('num_dense', 512, 'num dense in MMLDA')

warnings.simplefilter("ignore", category=DeprecationWarning)

input_H = 128
input_W = 256
num_class = 5005

batch_size = 16
epochs = 30

train_df = pd.read_csv("../data-raw/train.csv")
print(train_df.head())

def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
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
        x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 32x64x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 32x64x128
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 16x32x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 16x32x256
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 8x16x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 8x16x384
    for _ in range(4):
        x = subblock(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 4x8x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 4x8x512
    for _ in range(4):
        x = subblock(x, 128, **kwargs)

    x = GlobalMaxPooling2D()(x)  # 512
    branch_model = Model(inp, x)
    return branch_model

def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, input_H, input_W, 3))
    count = 0
    
    for fig in data['Image']:
        #https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/utils.py
        img = image.load_img("../data-"+dataset+"/"+fig, target_size=(input_H, input_W, 3), interpolation='bilinear') 
        x = image.img_to_array(img)

        X_train[count] = x
        if (count%1000 == 0):
            print("Processing image: ", count+1)
        count += 1
        if count >= m:
            break
    return X_train

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y_return = onehot_encoded
    count = 0
    for name in y:
        factor = 1. / labels[name]
        y_return[count] = factor * y_return[count]
        count += 1
    return y_return, label_encoder

#X = prepareImages(train_df, train_df.shape[0], "train")
X = prepareImages(train_df, 500, "train")

X /= 255


y, label_encoder = prepare_labels(train_df['Id'])
y=y[:500]

model_nonlinear = build_model(0)
x = model_nonlinear.output
predictions = Dense(num_class, activation='softmax')(x) #x = ReLU(W1x+b1)
model = Model(inputs=model_nonlinear.input, outputs=predictions)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.0001), metrics=['accuracy'])
model.summary()


# Prepare model model saving directory.
filepath_dir = 'trained_models/test_baseline'
save_dir = os.path.join(os.getcwd(), filepath_dir)
model_name = 'model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='acc', mode='max', verbose=2, save_best_only=True, save_weights_only=True)


callbacks = [checkpoint]


datagen = ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # epsilon for ZCA whitening
    zca_epsilon=0,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally
    width_shift_range=0.,
    # randomly shift images vertically
    height_shift_range=0.,
    # set range for random shear
    shear_range=0.,
    # set range for random zoom
    zoom_range=0.,
    # set range for random channel shifts
    channel_shift_range=0.,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # value used for fill_mode = "constant"
    cval=0.,
    # randomly flip images
    horizontal_flip=False,
    # randomly flip images
    vertical_flip=False,
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.)

datagen.fit(X)

model.fit_generator(
    datagen.flow(X, y, batch_size=batch_size),
    epochs=epochs,
    verbose=1,
    workers=4,)
    #callbacks=callbacks)


test = os.listdir("../data-test/")
print(len(test))


col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''

X = prepareImages(test_df, test_df.shape[0], "test")
X /= 255

predictions = model.predict(np.array(X), verbose=1)

for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))

test_df.head(10)
test_df.to_csv('submission.csv', index=False)