#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F


class VGG(FunctionSet):

    """
    VGGnet
    """

    def __init__(self):
        super(VGG, self).__init__(
            conv1_1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=F.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=F.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=F.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=F.Convolution2D(128, 64, 3, stride=1, pad=1),
            conv3_2=F.Convolution2D(64, 64, 3, stride=1, pad=1),

            fc5=F.Linear(10240, 2000),
            fc6=F.Linear(2000, 100),
            fc7=F.Linear(100, 4)
        )

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.max_pooling_2d(h, 2, stride=1)

        h = F.dropout(F.relu(self.fc5(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc6(h)), train=train, ratio=0.5)
        h = self.fc7(h)

        if train:
            return F.mean_squared_error(h, t)
        else:
            return F.mean_squared_error(h, t), h