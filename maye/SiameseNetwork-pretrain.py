# Modified by maye from https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563

from copy import copy

import keras
from pandas import read_csv
from tqdm import tqdm
from keras_tqdm import TQDMCallback
from pdb import set_trace
training = dict([(p, w) for _, p, w in read_csv('../train-new.csv').to_records()])
# test = [p for _, p, _ in read_csv('../data-raw/sample_submission.csv').to_records()]

# all = list(training.keys()) + test

# print(len(training), len(test), len(all), list(training.items())[:5], test[:5])
print(len(training))

# Read the bounding box data from the bounding box
import pickle
from os.path import isfile

with open('bbox.pickle', 'rb') as f:
    p2bb = pickle.load(f)

print("p2bb", len(p2bb), list(p2bb.items())[:5])

with open('p2size.pickle', 'rb') as f:
    p2size = pickle.load(f)

print("p2size", len(p2size), list(p2size.items())[:5])

import random
import numpy as np
from keras import backend as K
from keras.preprocessing.image import img_to_array, array_to_img
from scipy.ndimage import affine_transform
from PIL import Image as pil_image

img_shape = (128, 256, 1)  # The image shape used by the model
anisotropy = 2.15  # The horizontal compression ratio
crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy


def expand_path(p):
    if isfile('../data-train/' + p):
        return '../data-train/' + p
    if isfile('../data-test/' + p):
        return '../data-test/' + p
    return p


def read_raw_image(p):
    img = pil_image.open(expand_path(p))
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
    # If an image id was given, convert to filename
    size_x, size_y = p2size[p]

    # Determine the region of the original image we want to capture based on the bounding box.
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

    # Read the image, transform to black and white and comvert to numpy array
    img = read_raw_image(p).convert('L')
    # img = read_raw_image(p).convert('RGB')
    img = img_to_array(img)
    # print(img.shape)
    # Apply affine transformation
    matrix = trans[:2, :2]
    offset = trans[:2, 2]
    img = img.reshape(img.shape[:-1])
    img = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img))
    img = img.reshape(img_shape)
    # print (img)
    # Normalize to zero mean and unit variance
    img /= 255.
    img -= 0.456
    img /= 0.224
    # print(img)
    # set_trace()
    # img -= np.mean(img, keepdims=True)
    # img /= np.std(img, keepdims=True) + K.epsilon()
    return img


def read_for_training(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image(p, True)


def read_for_validation(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image(p, False)


# p = list(training.keys())[444]
p = '0001f9222.jpg'
read_raw_image(p).save('raw.jpg')
array_to_img(read_for_validation(p)).save('val.jpg')
array_to_img(read_for_training(p)).save('train.jpg')

from pdb import set_trace
set_trace()

from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    Lambda, MaxPooling2D, Reshape
from keras.models import Model


def build_model(lr, l2, activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)

    inp = Input(shape=img_shape)  # 384x384x1
    inp2 = Concatenate()([inp, inp, inp])
    dense_net = keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=inp2, input_shape=(128, 256, 3))
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
branch_model.summary()
head_model.summary()

from keras.utils import Sequence

# First try to use lapjv Linear Assignment Problem solver as it is much faster.
# At the time I am writing this, kaggle kernel with custom package fail to commit.
# scipy can be used as a fallback, but it is too slow to run this kernel under the time limit
# As a workaround, use scipy with data partitioning.
# Because algorithm is O(n^3), small partitions are much faster, but not what produced the submitted solution
from lap import lapjv

# Find the list of training images, keep only whales with at least two images.
train = []  # A list of training image ids
y2x = dict()
for p, w in training.items():
    if w == 'new_whale':
        continue
    if w in y2x and p not in y2x[w]:
        y2x[w].append(p)
    else:
        y2x[w] = [p]

y2x2 = copy(y2x)
for k, v in y2x.items():
    if len(v) > 1:
        train += v
    else:
        del y2x2[k]
y2x = y2x2
random.shuffle(train)
train_set = set(train)

image2id = {}  # The position in train of each training image id
for i, t in enumerate(train):
    image2id[t] = i

print("Training size", len(train), len(image2id), len(y2x))


class TrainingData(Sequence):
    def __init__(self, score, steps=1000, batch_size=32):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()
        self.score = -score  # Maximizing the score is the same as minimuzing -score.
        self.steps = steps
        self.batch_size = batch_size
        for ts in y2x.values():
            idxs = [image2id[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    self.score[
                        i, j] = 10000.0  # Set a large value for matching whales -- eliminates this potential pairing
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        b = np.zeros((size,) + img_shape, dtype=K.floatx())
        c = np.zeros((size, 1), dtype=K.floatx())
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = read_for_training(self.match[j][0])
            b[i, :, :, :] = read_for_training(self.match[j][1])
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = read_for_training(self.unmatch[j][0])
            b[i + 1, :, :, :] = read_for_training(self.unmatch[j][1])
            c[i + 1, 0] = 0  # Different whales
            j += 1
        return [a, b], c

    def on_epoch_end(self):
        if self.steps <= 0:
            return  # Skip this on the last epoch.
        self.steps -= 1
        self.match = []
        self.unmatch = []
        _, _, x = lapjv(self.score)  # Solve the linear assignment problem
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in y2x.values():
            d = copy(ts)
            while True:
                random.shuffle(d)
                if not np.any(ts == d):
                    break
            for ab in zip(ts, d):
                self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((train[i], train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(train) and len(self.unmatch) == len(train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size


# Test on a batch of 32 with random costs.
score = np.random.random_sample(size=(len(train), len(train)))
data = TrainingData(score)
(a, b), c = data[0]
print(a.shape, b.shape, c.shape)


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
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
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


def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
    return K.get_value(model.optimizer.lr)


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


def compute_score(verbose=1):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    features = branch_model.predict_generator(FeatureGen(train, verbose=verbose), max_queue_size=12, workers=12,
                                              verbose=0, use_multiprocessing=True)
    score = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=12, verbose=0)
    score = score_reshape(score, features)
    return features, score


def make_steps(step, ampl):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    global image2id, steps, features, score, histories

    # shuffle the training pictures
    random.shuffle(train)

    # Map training picture hash value to index in 'train' array
    image2id = {}  # The position in train of each training image id
    for i, t in enumerate(train):
        image2id[t] = i

    # Compute the match score for each picture pair
    features, score = compute_score()

    # Train the model for 'step' epochs
    history = model.fit_generator(
        TrainingData(score + ampl * np.random.random_sample(size=score.shape), steps=step, batch_size=16),
        initial_epoch=steps, epochs=steps + step, max_queue_size=12, workers=6, verbose=0, use_multiprocessing=True,
        callbacks=[
            TQDMCallback(leave_inner=True, metric_format='{value:0.3f}')
        ]).history
    steps += step

    # Collect history data
    history['epochs'] = steps
    history['ms'] = np.mean(score)
    history['lr'] = get_lr(model)
    print(history['epochs'], history['lr'], history['ms'])
    histories.append(history)


model_name = 'siamese-pretrain'
histories = []
steps = 0


def prepare_submission(threshold, filename):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    vtop  = 0
    vhigh = 0
    pos   = [0,0,0,0,0,0]
    prediction = []
    with open(filename, 'w') as f:
        f.write('Image,Id\n')
        for i,p in enumerate(tqdm(val)):
            t = []
            s = set()
            a = score[i,:]
            for j in list(reversed(np.argsort(a))):
                h = known[j]
                if a[j] < threshold and "new_whale" not in s:
                    pos[len(t)] += 1
                    s.add("new_whale")
                    t.append("new_whale")
                    if len(t) == 5:
                        break
                w = training[h]
                assert w != "new_whale"
                if w not in s:
                    if a[j] > 1.0:
                        vtop += 1
                    elif a[j] >= threshold:
                        vhigh += 1
                    s.add(w)
                    t.append(w)
                    if len(t) == 5:
                        break
                if len(t) == 5: break
            if "new_whale" not in s: pos[5] += 1
            assert len(t) == 5 and len(s) == 5
            f.write(p + ',' + ' '.join(t[:5]) + '\n')
            prediction.append(t[:5])
    return vtop, vhigh, pos, prediction


def compute_acc(val, prediction, golden, top_k=5):
    """
    Compute ACC score
    :param val: in shape [n] of picture name
    :param prediction: in shape [n, top_k] of whale ids
    :param golden: in shape [n] of golden
    """
    assert len(val) == len(prediction)
    assert len(val) == len(golden)
    total_counts = len(val)
    true_counts = 0
    for i in range(total_counts):
        if golden[i] in prediction[i][:top_k]:
            true_counts += 1
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("90 ACC TOP %d:" % top_k, true_counts / float(total_counts))
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")


if isfile(model_name + '-90.model'):
# if False:
    tmp = keras.models.load_model(model_name + '-90.model')
    orig_model.set_weights(tmp.get_weights())

    known = []
    for p, w in training.items():
        if w != "new_whale":  # Use only identified whales
            known.append(p)
    known = sorted(known)

    val = [p for _, p, _ in read_csv('../val.csv').to_records()]
    ans = [p for _, _, p in read_csv('../val.csv').to_records()]

    fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0, use_multiprocessing=True)
    fval = branch_model.predict_generator(FeatureGen(val), max_queue_size=20, workers=10, verbose=0, use_multiprocessing=True)
    score = head_model.predict_generator(ScoreGen(fknown, fval, batch_size=8192), max_queue_size=20, workers=10, verbose=0)
    score = score_reshape(score, fknown, fval)
    # print("Here")
    vtop, vhigh, pos, prediction = prepare_submission(0.9999, 'mpiotte-pretrain-90.csv')
    print(vtop, vhigh, pos)
    from pdb import set_trace
    # set_trace()
    compute_acc(val, prediction, ans, 5)
    # set_trace()
else:
    # epoch -> 10
    make_steps(10, 1000)
    orig_model.save(model_name + '-10.model')
    ampl = 100.0
    for _ in range(10):
        print('noise ampl.  = ', ampl)
        make_steps(5, ampl)
        ampl = max(1.0, 100 ** -0.1 * ampl)
        orig_model.save(model_name + '-%d.model' % steps)
    # epoch -> 150
    for _ in range(9):
        make_steps(10, 1.0)
        orig_model.save(model_name + '-%d.model' % steps)
    # # epoch -> 200
    # set_lr(model, 16e-5)
    # for _ in range(10):
    #     make_steps(5, 0.5)
    # # epoch -> 240
    # set_lr(model, 4e-5)
    # for _ in range(8):
    #     make_steps(5, 0.25)
    # # epoch -> 250
    # set_lr(model, 1e-5)
    # for _ in range(2):
    #     make_steps(5, 0.25)
    # # epoch -> 300
    # weights = model.get_weights()
    # model, branch_model, head_model = build_model(64e-5, 0.0002)
    # model.set_weights(weights)
    # for _ in range(10):
    #     make_steps(5, 1.0)
    # # epoch -> 350
    # set_lr(model, 16e-5)
    # for _ in range(10):
    #     make_steps(5, 0.5)
    # # epoch -> 390
    # set_lr(model, 4e-5)
    # for _ in range(8):
    #     make_steps(5, 0.25)
    # # epoch -> 400
    # set_lr(model, 1e-5)
    # for _ in range(2):
    #     make_steps(5, 0.25)
    orig_model.save(model_name + '.model')
