#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from skimage import io
from skimage.transform import resize
from sklearn.datasets import load_files


files = load_files('./raw_image', load_content=False)

for file_n in files.filenames:
    img = io.imread(file_n)
    resized = resize(img, (178, 267))
    io.imsave(file_n, resized)