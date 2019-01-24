import os
import random

import keras
from PIL.ImageDraw import Draw
from keras.utils import Sequence
from keras_preprocessing.image import img_to_array
from pandas import read_csv
from tqdm import tqdm

model_name = 'siamese-pretrain'
img_path = '../web/'
img_shape_bbox = (128, 128, 1)
img_shape = (128, 256, 1)  # The image shape used by the model
anisotropy = 2.15  # The horizontal compression ratio
crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy

from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    Lambda, MaxPooling2D, Reshape
from keras.models import Model
import keras.backend as K
import numpy as np


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


def build_model(lr, l2, activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)

    inp = Input(shape=img_shape)  # 384x384x1
    inp2 = Concatenate()([inp, inp, inp])
    # dense_net = keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=inp2, input_shape=(128, 256, 3))
    dense_net = keras.applications.densenet.DenseNet121(include_top=False, input_tensor=inp2, input_shape=(128, 256, 3))
    x = dense_net.output
    x = GlobalMaxPooling2D()(x)  # 512
    branch_model = Model(inp, x)

    ############
    # HEAD MODEL
    ############
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])
    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    # model = keras.utils.multi_gpu_model(Model([img_a, img_b], x), gpus=1)
    orig_model = Model([img_a, img_b], x)
    # model = keras.utils.multi_gpu_model(Model([img_a, img_b], x), gpus=2)
    model = orig_model
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model, orig_model, dense_net


model, branch_model, head_model, orig_model, dense_model = build_model(64e-5, 0)

# Read the bounding box data from the bounding box
import pickle
from os.path import isfile

p2bb_bbox = dict()
with open('bbox.pickle', 'rb') as f:
    p2bb = pickle.load(f)

print("p2bb", len(p2bb), p2bb.items()[:5])


from PIL import Image as pil_image
from scipy.ndimage import affine_transform


def read_raw_image(p):
    if isfile(img_path + p):
        img = pil_image.open(img_path + p)
    else:
        img = pil_image.open('../data-train/' + p)
    return img


def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


def read_cropped_image(p, augment):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    # Read the image, transform to black and white and comvert to numpy array
    img = read_raw_image(p).convert('L')
    size_x, size_y = img.size
    img = img_to_array(img)

    # Determine the region of the original image we want to capture based on the bounding box.
    try:
        x0, y0, x1, y1 = p2bb_bbox[p]
    except:
        x0, y0, x1, y1 = p2bb[p]
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    if x0 < 0:
        x0 = 0
    if x1 > size_x:
        x1 = size_x
    if y0 < 0:
        y0 = 0
    if y1 > size_y:
        y1 = size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy * anisotropy:
        dy = 0.5 * (dx / anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx = 0.5 * (dy * anisotropy - dx)
        x0 -= dx
        x1 += dx

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5 * img_shape[0]], [0, 1, -0.5 * img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0) / img_shape[0], 0, 0], [0, (x1 - x0) / img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
            random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
        ), trans)
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)

    # Apply affine transformation
    matrix = trans[:2, :2]
    offset = trans[:2, 2]
    img = img.reshape(img.shape[:-1])
    img = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img))
    img = img.reshape(img_shape)

    # Normalize to zero mean and unit variance
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img


def read_for_validation(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image(p, False)


def score_reshape(score, x, y=None):
    """
    Transformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size,))
        iy = iy.reshape((iy.size,))
        m[iy, ix] = score.squeeze()
    return m


# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data, batch_size=256, verbose=1):
        super(FeatureGen, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.verbose = verbose
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Features')

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        for i in range(size): a[i, :, :, :] = read_for_validation(self.data[start + i])
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return a

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


# A Keras generator to evaluate on the HEAD MODEL on features already pre-computed.
# It computes only the upper triangular matrix of the cost matrix if y is None.
class ScoreGen(Sequence):
    def __init__(self, x, y=None, batch_size=4096, verbose=1):
        super(ScoreGen, self).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        if y is None:
            self.y = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0], 1)
        else:
            self.iy, self.ix = np.indices((y.shape[0], x.shape[0]))
            self.ix = self.ix.reshape((self.ix.size,))
            self.iy = self.iy.reshape((self.iy.size,))
        self.subbatch = (len(self.x) + self.batch_size - 1) // self.batch_size
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Scores')

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.ix))
        a = self.y[self.iy[start:end], :]
        b = self.x[self.ix[start:end], :]
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return [a, b]

    def __len__(self):
        return (len(self.ix) + self.batch_size - 1) // self.batch_size


tmp = keras.models.load_model(model_name + '-80.model')
orig_model.set_weights(tmp.get_weights())

bbox_model = keras.models.load_model('cropping.model')

known = []
training = dict([(p, w) for _, p, w in read_csv('../train-new.csv').to_records()])
BLACK_LIST = {'363169733.jpg', '9cfc18024.jpg', '431a8ddcc.jpg', '4e824e5c5.jpg', '34e77468d.jpg', '444b09aca.jpg',
              'b67ff037d.jpg', 'b7c4aa1b8.jpg', 'f0cfd99be.jpg', 'd833c83d0.jpg', '27afd7a26.jpg', 'b1f28ee4d.jpg',
              'cda2c9dbf.jpg', 'b370e1339.jpg', 'f9daed87f.jpg', 'a1cddb0d0.jpg', '85a95e7a8.jpg', '63492795c.jpg',
              '233c00eda.jpg', 'c3753fbe1.jpg', '1ef6137fb.jpg'}
for p, w in training.items():
    if w != "new_whale":  # Use only identified whales
        known.append(p)
known = sorted(known)

fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)


# Transform coordinates according to the provided affine transformation
def coord_transform(list, trans):
    result = []
    for x, y in list:
        y, x, _ = trans.dot([y, x, 1]).astype(np.int)
        result.append((x, y))
    return result


# Compute the coordinate transformation required to center the pictures, padding as required.
def center_transform(affine, input_shape):
    hi, wi = float(input_shape[0]), float(input_shape[1])
    ho, wo = float(img_shape_bbox[0]), float(img_shape_bbox[1])
    top, left, bottom, right = 0, 0, hi, wi
    if wi / hi / anisotropy < wo / ho:  # input image too narrow, extend width
        w = hi * wo / ho * anisotropy
        left = (wi - w) / 2
        right = left + w
    else:  # input image too wide, extend height
        h = wi * ho / wo / anisotropy
        top = (hi - h) / 2
        bottom = top + h
    center_matrix = np.array([[1, 0, -ho / 2], [0, 1, -wo / 2], [0, 0, 1]])
    scale_matrix = np.array([[(bottom - top) / ho, 0, 0], [0, (right - left) / wo, 0], [0, 0, 1]])
    decenter_matrix = np.array([[1, 0, hi / 2], [0, 1, wi / 2], [0, 0, 1]])
    return np.dot(np.dot(decenter_matrix, scale_matrix), np.dot(affine, center_matrix))


def read_raw_image(p):
    return pil_image.open(img_path + p)


def read_array(p):
    img = read_raw_image(p).convert('L')
    return img_to_array(img)


# Apply an affine transformation to an image represented as a numpy array.
def transform_img(x, affine):
    matrix = affine[:2, :2]
    offset = affine[:2, 2]
    x = np.moveaxis(x, -1, 0)
    channels = [affine_transform(channel, matrix, offset, output_shape=img_shape_bbox[:-1], order=1,
                                 mode='constant', cval=np.average(channel)) for channel in x]
    return np.moveaxis(np.stack(channels, axis=0), 0, -1)


def read_for_validation_bbox(p):
    x = read_array(p)
    t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = center_transform(t, x.shape)
    x = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x, t


def generate_bbox(p, model):
    img, trans = read_for_validation_bbox(p)
    a = np.expand_dims(img, axis=0)
    x0, y0, x1, y1 = model.predict(a).squeeze()
    (u0, v0), (u1, v1) = coord_transform([(x0, y0), (x1, y1)], trans)
    p2bb[p] = (u0, v0, u1, v1)
    return u0, v0, u1, v1


def preview_bbox(p, x0, y0, x1, y1):
    img = read_raw_image(p).convert('RGB')
    draw = Draw(img)
    draw.line([(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)], fill='yellow', width=6)
    img.save('../web/' + p + '-bbox.jpg')


def infer_one(p_name, threshold=0.99):
    u0, v0, u1, v1 = generate_bbox(p_name, bbox_model)
    preview_bbox(p_name, u0, v0, u1, v1)

    fval = branch_model.predict_generator(FeatureGen([p_name], batch_size=1))
    score = head_model.predict_generator(ScoreGen(fknown, fval, batch_size=8192), max_queue_size=20, workers=10, verbose=0)
    score = score_reshape(score, fknown, fval)
    t = []
    xs = []
    s = set()
    a = score[0, :]
    for j in list(reversed(np.argsort(a))):
        h = known[j]
        if h in BLACK_LIST:
            continue
        # if a[j] < threshold and "new_whale" not in s:
        #     s.add("new_whale")
        #     t.append("new_whale")
        #     xs.append("new_whale")
        #     if len(t) == 10:
        #         break
        w = training[h]
        if w not in s:
            s.add(w)
            t.append(w)
            xs.append(h)
            if len(t) == 9:
                break
    return t, xs


from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory

UPLOAD_FOLDER = '../web'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='./')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'files' not in request.files:
            print("Here")
            flash('No file part')
            return redirect(request.url)
        file = request.files['files']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print("HereHere")
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            labels, t = infer_one(filename, 0)
            # t = ['b758f5366.jpg', '03d0be3ab.jpg', '7e5b193ce.jpg', 'bcb504542.jpg']
            return render_template('index.html', file_name=filename, predictions=zip(t, labels))
    return render_template('index.html')


@app.route('/results/<file_name>')
def results(file_name, predictions):
    return render_template('index.html', file_name=file_name, predictions=predictions)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/imgs/<filename>')
def training_imgs(filename):
    return send_from_directory('../data-train/',
                               filename)


app.run('0.0.0.0', 8888, debug=False, threaded=False)



# print("Doing infer...")
# from time import time
# start = time()
# # ANS should be w_9c506f6
# ret, xs = infer_one('0af805558.jpg')
# end = time()
# print("Took %f ms" % ((end - start) * 1000.0))
#
# print(ret)
# print(xs)