#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import os
import shutil

import numpy as np
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split


here = os.path.dirname(os.path.abspath(__file__))
files = load_files(os.path.join(here, '../dapc15'), load_content=False)


train_data_dir = os.path.join(here, '../dapc15train')
if not os.path.exists(train_data_dir):
    os.mkdir(train_data_dir)
test_data_dir = os.path.join(here, '../dapc15test')
if not os.path.exists(test_data_dir):
    os.mkdir(test_data_dir)

target_names = []
filenames = []
for i in xrange(len(files.filenames)):
    if files.filenames[i].endswith('_mask.jpg'):
        continue
    target_name = files.target_names[files.target[i]]
    target_names.append(target_name)
    filenames.append(files.filenames[i])

N = len(filenames)
p = np.random.randint(0, N, int(0.2 * N))

for i in xrange(N):
    fname = filenames[i]
    target_name = target_names[i]
    if i in p:
        # test file
        target_dir = os.path.join(test_data_dir, target_name)
    else:
        # train file
        target_dir = os.path.join(train_data_dir, target_name)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    shutil.copy(
        fname,
        os.path.join(target_dir, os.path.basename(fname)),
    )
    mask_fname = os.path.splitext(fname)[0] + '_mask.jpg'
    shutil.copy(
        mask_fname,
        os.path.join(target_dir, os.path.basename(mask_fname)),
    )
