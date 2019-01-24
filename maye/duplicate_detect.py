from pandas import read_csv

training = dict([(p, w) for _, p, w in read_csv('../data-raw/train.csv').to_records()])
test = [p for _, p, _ in read_csv('../data-raw/sample_submission.csv').to_records()]

all = list(training.keys()) + test

print(len(training), len(test), len(all), list(training.items())[:5], test[:5])

# Determise the size of each image
from os.path import isfile
from PIL import Image as pil_image
from tqdm import tqdm
import pickle
import numpy as np
from imagehash import phash
from math import sqrt


def expand_path(p):
    if isfile('../data-train/' + p): return '../data-train/' + p
    if isfile('../data-test/' + p): return '../data-test/' + p
    return p


if isfile('p2size.pickle'):
    p2size = pickle.load(open('p2size.pickle', 'rb'))
else:
    p2size = {}
    for p in tqdm(all):
        size = pil_image.open(expand_path(p)).size
        p2size[p] = size

    print(len(p2size), list(p2size.items())[:5])

    pickle.dump(p2size, open('p2size.pickle', 'wb'))

# Read or generate p2h, a dictionary of image name to image id (picture to hash)
# Compute phash for each image in the training and test set.
if isfile('p2h.pickle'):
    p2h = pickle.load(open('p2h.pickle', 'rb'))
else:
    from multiprocessing import Pool, Manager
    manager = Manager()
    p2h = manager.dict()

    def process(p):
        img = pil_image.open(expand_path(p))
        h = phash(img)
        p2h[p] = h


    pool = Pool(40)
    for p in all:
        pool.apply_async(process, (p,))
    pool.close()
    print("Joining Pool...")
    pool.join()
    print("Joined Pool")
    print("p2h", len(p2h), list(p2h.items())[:5])

    # Find all images associated with a given phash value.
    h2ps = {}
    for p, h in p2h.items():
        if h not in h2ps:
            h2ps[h] = []
        if p not in h2ps[h]:
            h2ps[h].append(p)

    # Find all distinct phash values
    hs = list(h2ps.keys())

print("p2h", len(p2h), list(p2h.items())[:5])

# For each image id, determine the list of pictures
h2ps = {}
for p,h in p2h.items():
    if h not in h2ps:
        h2ps[h] = []
    if p not in h2ps[h]:
        h2ps[h].append(p)
print(len(h2ps), list(h2ps.items())[:1])


# For each images id, select the preferred image
def prefer(ps):
    if len(ps) == 1:
        return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        if s[0]*s[1] > best_s[0]*best_s[1]:  # Select the image with highest resolution
            best_p = p
            best_s = s
    return best_p


p2p = {}
for h, ps in h2ps.items():
    if len(ps) == 1:
        continue
    best_p = prefer(ps)
    for pp in ps:
        p2p[pp] = best_p
print(len(p2p), list(p2p.items())[:5])

pickle.dump(p2p, open('p2p.pickle', 'wb'))
# total 4
