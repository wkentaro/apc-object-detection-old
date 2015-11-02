#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pickle
import six
import time

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import numpy as np

print('using gpu: 0')
# xp = np
xp = cuda.cupy

model_name = 'VGG'
stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename='../logs/{}_{}.txt'.format(model_name, stamp),
    level=logging.DEBUG,
)


def fetch_roidata(which_set):
    with open('../logs/roi_X_{0}.pkl'.format(which_set)) as f:
        X = pickle.load(f)
    with open('../logs/roi_y_{0}.pkl'.format(which_set)) as f:
        y = pickle.load(f)
    return X, y


# Prepare dataset
print('loading ROI dataset')
x_train, _ = fetch_roidata(which_set='train')
x_train = x_train.reshape((len(x_train), 3, 44, 66))
print('x_train.shape:', x_train.shape)
y_train = np.zeros((len(x_train), 4))
N = len(x_train)
x_test, _ = fetch_roidata(which_set='test')
x_test = x_test.reshape((len(x_test), 3, 44, 66))
print('x_test.shape:', x_test.shape)
y_test = np.zeros((len(x_test), 4))
N_test = len(x_test)
print('N:', N)
print('N_test:', N_test)


# params
batchsize = 10
n_epoch = 20


# model
from vgg import VGG
model = VGG()
cuda.get_device(0).use()
model.to_gpu()


# optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)


# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch = xp.asarray(x_train[perm[i:i + batchsize]], dtype=xp.float32)
        y_batch = xp.asarray(y_train[perm[i:i + batchsize]], dtype=xp.float32)

        optimizer.zero_grads()
        loss = model.forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        if epoch == 1 and i == 0:
            with open('../logs/{0}_graph.dot'.format(model_name), 'w') as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open('../logs/{0}_graph.wo_split.dot'.format(model_name), 'w') as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(loss.data) * len(y_batch)

    msg = 'epoch:{:02d}\ttrain mean loss={},'.format(epoch, sum_loss / N)
    logging.info(msg)
    print(msg)

    # evaluation
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = xp.asarray(x_test[i:i + batchsize], dtype=xp.float32)
        y_batch = xp.asarray(y_test[i:i + batchsize], dtype=xp.float32)

        loss, h = model.forward(x_batch, y_batch, train=False)

        sum_loss += float(loss.data) * len(y_batch)

    msg = 'epoch:{:02d}\ttest mean loss={},'.format(epoch, sum_loss / N)
    logging.info(msg)
    print(msg)


print('dumping model')
with open('../logs/trained_{0}_model.pkl'.format(model_name), 'wb') as f:
    pickle.dump(model.to_cpu(), f)
print('done')