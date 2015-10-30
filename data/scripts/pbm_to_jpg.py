#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from skimage import io
from sklearn.datasets import load_files


files = load_files('./mask_image', load_content=False)

for file_n in files.filenames:
    img = io.imread(file_n)
    basename, _ = os.path.splitext(file_n)
    io.imsave('{0}.jpg'.format(basename), img)