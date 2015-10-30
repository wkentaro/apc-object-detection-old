#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from skimage import io
from sklearn.datasets import load_files


files = load_files('./mask_image', load_content=False)

for file_n in files.filenames:
    img = io.imread(file_n, as_grey=True)
    io.imsave(file_n, ~img)