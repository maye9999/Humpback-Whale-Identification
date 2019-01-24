import numpy as np
from scipy.ndimage import affine_transform
from keras.preprocessing.image import img_to_array
from keras import backend as K
from keras.utils import Sequence
from keras.models import Model, load_model
from pandas import read_csv
from PIL.ImageDraw import Draw
from PIL import Image as pil_image
from os.path import isfile
import pickle
from tqdm import tqdm

img_shape = (128, 128, 1)
anisotropy = 2.15


def expand_path(p):
    if isfile('../data-train/' + p):
        return '../data-train/' + p
    if isfile('../data-test/' + p):
        return '../data-test/' + p
    return p


# Transform coordinates according to the provided affine transformation
def coord_transform(list, trans):
    result = []
    for x, y in list:
        y, x, _ = trans.dot([y, x, 1]).astype(np.int)
        result.append((x, y))
    return result


def read_raw_image(p):
    return pil_image.open(expand_path(p))


def read_array(p):
    img = read_raw_image(p).convert('L')
    return img_to_array(img)


# Apply an affine transformation to an image represented as a numpy array.
def transform_img(x, affine):
    matrix = affine[:2, :2]
    offset = affine[:2, 2]
    x = np.moveaxis(x, -1, 0)
    channels = [affine_transform(channel, matrix, offset, output_shape=img_shape[:-1], order=1,
                                 mode='constant', cval=np.average(channel)) for channel in x]
    return np.moveaxis(np.stack(channels, axis=0), 0, -1)


# Compute the coordinate transformation required to center the pictures, padding as required.
def center_transform(affine, input_shape):
    hi, wi = float(input_shape[0]), float(input_shape[1])
    ho, wo = float(img_shape[0]), float(img_shape[1])
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


def read_for_validation(p):
    x = read_array(p)
    t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = center_transform(t, x.shape)
    x = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x, t


def generate_bbox(to_do, model):
    print(len(to_do))
    ret = {}
    for p in tqdm(to_do):
        img, trans = read_for_validation(p)
        a = np.expand_dims(img, axis=0)
        x0, y0, x1, y1 = model.predict(a).squeeze()
        (u0, v0), (u1, v1) = coord_transform([(x0, y0), (x1, y1)], trans)
        ret[p] = (u0, v0, u1, v1)
    return ret


def preview(to_do, dic):
    for p in to_do:
        img = read_raw_image(p).convert('RGB')
        draw = Draw(img)
        x0, y0, x1, y1 = dic[p]
        draw.line([(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)], fill='yellow', width=6)
        img.save(p)


if __name__ == '__main__':
    model = load_model('cropping.model')
    model.summary()
    to_do = [p for _, p, _ in read_csv('../data-raw/train.csv').to_records()]
    to_do += [p for _, p, _ in read_csv('../data-raw/sample_submission.csv').to_records()]
    dic = generate_bbox(to_do, model)
    with open('bbox.pickle', 'wb') as fout:
        pickle.dump(dic, fout)
    # preview(to_do[:25], dic)
    # print(dic)
