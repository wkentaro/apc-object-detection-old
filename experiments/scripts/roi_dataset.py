#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import argparse
import cPickle as pickle
import os
import sys

import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split

from _init_path import DATA_DIR, LOGS_DIR


parser = argparse.ArgumentParser()
parser.add_argument('which_set')
args = parser.parse_args()

which_set = args.which_set

if which_set not in ['train', 'test']:
    sys.exit(1)


data_dir = os.path.join(DATA_DIR, 'dapc15{0}'.format(which_set))
files = load_files(data_dir, load_content=False)


filenames_for_X = []
filenames_for_y = []
for fname in files.filenames:
    fname = os.path.abspath(fname)
    if fname.endswith('_mask.jpg'):
        filenames_for_y.append(fname)
    else:
        filenames_for_X.append(fname)


def bounding_rect_of_mask(mask):
    where = np.argwhere(mask)
    (y_start, x_start), (y_end, x_end) = where.min(0), where.max(0) + 1
    return y_start, y_end, x_start, x_end


y = []
for fname in filenames_for_y:
    mask = io.imread(fname, as_grey=True)
    y_start, y_end, x_start, x_end = bounding_rect_of_mask(mask)
    y_center, x_center = mask.shape[0] / 2, mask.shape[1] / 2
    y_start /= y_center
    y_end /= y_center
    x_start /= x_center
    x_end /= x_center
    y.append([y_start, y_end, x_start, x_end])
y = np.array(y)


pkl_fname = os.path.join(LOGS_DIR, 'roi_y_{0}.pkl'.format(which_set))
with open(pkl_fname, 'wb') as f:
    pickle.dump(y, f)


IMG_SHAPE = (89, 133)
print('Reshaping image to {0}'.format(IMG_SHAPE))


X = []
for fname in filenames_for_X:
    img = io.imread(fname)
    img = resize(img, IMG_SHAPE)
    X.append(img.reshape(-1) / 255.)
X = np.array(X)


pkl_fname = os.path.join(LOGS_DIR, 'roi_X_{0}.pkl'.format(which_set))
with open(pkl_fname, 'wb') as f:
    pickle.dump(X, f)
