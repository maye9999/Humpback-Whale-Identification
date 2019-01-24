#CUDA_VISIBLE_DEVICES=1 nohup python -u CNN_MMLDA.py > train_logs/CNN_baseline.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 python CNN_MMLDA.py

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
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Lambda
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

input_H = 288
input_W = 576
num_class = 5005

batch_size = 8
epochs = 30


#Load means in MMLDA
kernel_dict = loadmat('kernelpara/dimmeans'+str(FLAGS.num_dense)+'_meanvar'+str(FLAGS.mean_var)+'.mat')
mean_logits = kernel_dict['mean_logits'] #num_class X num_dense
mean_logits = tf.constant(mean_logits,dtype=tf.float32)


train_df = pd.read_csv("../data-raw/train.csv")
print(train_df.head())


def dot_loss(y_true, y_pred):
    return -tf.reduce_sum(y_pred * y_true, axis=-1) #batch_size X 1

#MMLDA prediction function
def MMLDA_layer(x, means=mean_logits):
    #x_shape = batch_size X num_dense
    x_expand = tf.tile(tf.expand_dims(x,axis=1),[1,num_class,1]) #batch_size X num_class X num_dense
    mean_expand = tf.expand_dims(means,axis=0) #1 X num_class X num_dense
    logits = -tf.reduce_sum(tf.square(x_expand - mean_expand), axis=-1) #batch_size X num_class
    return logits

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

X = prepareImages(train_df, train_df.shape[0], "train")
#X = prepareImages(train_df, 500, "train")

X /= 255


y, label_encoder = prepare_labels(train_df['Id'])
#y=y[:500]

model_input = Input(shape=(input_H, input_W, 3))
model_nonlinear = DenseNet121(include_top=False, \
	weights=None, \
	input_tensor=model_input, \
    pooling = 'avg', \
	input_shape=None, \
	classes=y.shape[1])
x = model_nonlinear.output
x = Dense(FLAGS.num_dense, activation=None)(x) #x = ReLU(W1x+b1)
new_layer = Lambda(MMLDA_layer)
predictions = new_layer(x)
model = Model(inputs=model_input, outputs=predictions)

model.compile(loss=dot_loss, optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
model.summary()


# Prepare model model saving directory.
filepath_dir = 'trained_models/test_MMLDA'
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
    zca_epsilon=1e-06,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
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
    vertical_flip=True,
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

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