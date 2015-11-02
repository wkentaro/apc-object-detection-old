#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import numpy as np


def fetch_roidata(which_set):
    with open('../logs/roi_X_{0}.pkl'.format(which_set)) as f:
        X = pickle.load(f)
    with open('../logs/roi_y_{0}.pkl'.format(which_set)) as f:
        y = pickle.load(f)
    return X, y


# GPU or CPU
parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
    print('using gpu')
xp = cuda.cupy if args.gpu >= 0 else np

# Fixed params
batchsize = 10
n_epoch = 20
input_img_shape = (89, 133, 3)

# Prepare dataset
print('loading ROI dataset')
x_train, y_train = fetch_roidata(which_set='train')
x_test, y_test = fetch_roidata(which_set='test')
N, N_test = len(x_train), len(x_test)

print('N:', N)
print('N_test:', N_test)

# Prepare multi-layer perceptron model
model = chainer.FunctionSet(
    l1=F.Linear(35511, 10000),
    l2=F.Linear(10000, 2000),
    l3=F.Linear(2000, 100),
    l4=F.Linear(100, 20),
    l5=F.Linear(20, 100),
    l6=F.Linear(100, 2000),
    l7=F.Linear(2000, 10000),
    l8=F.Linear(10000, 35511),
)
if args.gpu >= 0:
    print('converting model to gpu')
    cuda.get_device(args.gpu).use()
    model.to_gpu()


def forward(x_data, y_data, train=True):
    # Neural net architecture
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    h3 = F.dropout(F.relu(model.l3(h2)), train=train)
    h4 = F.dropout(F.relu(model.l4(h3)), train=train)
    h5 = F.dropout(F.relu(model.l5(h4)), train=train)
    h6 = F.dropout(F.relu(model.l6(h5)), train=train)
    h7 = F.dropout(F.relu(model.l7(h6)), train=train)
    y = model.l8(h7)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch = xp.asarray(x_train[perm[i:i + batchsize]])
        y_batch = xp.asarray(y_train[perm[i:i + batchsize]])

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        if epoch == 1 and i == 0:
            with open("graph.dot", "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open("graph.wo_split.dot", "w") as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = xp.asarray(x_test[i:i + batchsize])
        y_batch = xp.asarray(y_test[i:i + batchsize])

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
