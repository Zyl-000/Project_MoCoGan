import tensorflow as tf
import tensorlayer as tl
import numpy
from tensorlayer.layers import Input, Dense, Conv2d, Conv3d, Flatten
from tensorlayer.layers import DeConv2d, BatchNorm1d, BatchNorm2d, BatchNorm3d
from tensorlayer.models import Model


class discriminator_I(Model):
    
    def __init__(self, nc=3, ndf=64):
        super(discriminator_I,self).__init__()

        self.conv1 = Conv2d(in_channels=nc, n_filter=ndf, filter_size=(4,4), strides=(2,2), padding='SAME', b_init=None, act=tf.nn.leaky_relu)
        self.conv2 = Conv2d(in_channels=ndf, n_filter=ndf * 2, filter_size=(4,4), strides=(2,2), padding='SAME', b_init=None)
        self.batch1 = BatchNorm2d(num_features=ndf * 2, act=tf.nn.leaky_relu)
        self.conv3 = Conv2d(in_channels=ndf * 2, n_filter=ndf * 4, filter_size=(4,4), strides=(2,2), padding='SAME', b_init=None)
        self.batch2 = BatchNorm2d(num_features=ndf * 4, act=tf.nn.leaky_relu)
        self.conv4 = Conv2d(in_channels=ndf * 4, n_filter=ndf * 8, filter_size=(4,4), strides=(2,2), padding='SAME', b_init=None)
        self.batch3 = BatchNorm2d(num_features=ndf * 8, act=tf.nn.leaky_relu)
        self.conv5 = Conv2d(in_channels=ndf * 8, n_filter=1, filter_size=(6,6), strides=(1,1), padding='VALID', b_init=None, act=tf.nn.sigmoid)
    
    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.batch1(z)
        z = self.conv3(z)
        z = self.batch2(z)
        z = self.conv4(z)
        z = self.batch3(z)
        out = self.conv5(z)
        return out


class discriminator_V(Model):

    def __init__(self, nc=3, ndf=64, T=16):
        super(discriminator_V,self).__init__()

        self.conv1 = Conv3d(in_channels=nc, n_filter=ndf, filter_size=(4,4,4), strides=(2,2,2), padding='SAME', b_init=None, act=tf.nn.leaky_relu)
        self.conv2 = Conv3d(in_channels=ndf, n_filter=ndf * 2, filter_size=(4,4,4), strides=(2,2,2), padding='SAME', b_init=None)
        self.batch1 = BatchNorm3d(num_features=ndf * 2, act=tf.nn.leaky_relu)
        self.conv3 = Conv3d(in_channels=ndf * 2, n_filter=ndf * 4, filter_size=(4,4,4), strides=(2,2,2), padding='SAME', b_init=None)
        self.batch2 = BatchNorm3d(num_features=ndf * 4, act=tf.nn.leaky_relu)
        self.conv4 = Conv3d(in_channels=ndf * 4, n_filter=ndf * 8, filter_size=(4,4,4), strides=(2,2,2), padding='SAME', b_init=None)
        self.batch3 = BatchNorm3d(num_features=ndf * 8, act=tf.nn.leaky_relu)
        self.flatten = Flatten()
        self.dense = Dense(in_channels=(int((ndf * 8) * (T / 16) * 36)), n_units=1, act=tf.nn.sigmoid)

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.batch1(z)
        z = self.conv3(z)
        z = self.batch2(z)
        z = self.conv4(z)
        z = self.batch3(z)
        z = self.flatten(z)
        out = self.dense(z)
        return out


def generator_I(shape, nc=3, ngf=64, nz=60):

    ni = Input(shape)
    nn = DeConv2d(in_channels=nz, n_filter=ngf * 8, filter_size=(6,6), strides=(1,1), padding='VALID', b_init=None)(ni)
    nn = BatchNorm2d(num_features=ngf * 8, act=tf.nn.relu)(nn)
    nn = DeConv2d(in_channels=ngf * 8, n_filter=ngf * 4, filter_size=(4,4), strides=(2,2), padding='SAME', b_init=None)(nn)
    nn = BatchNorm2d(num_features=ngf * 4, act=tf.nn.relu)(nn)
    nn = DeConv2d(in_channels=ngf * 4, n_filter=ngf * 2, filter_size=(4,4), strides=(2,2), padding='SAME', b_init=None)(nn)
    nn = BatchNorm2d(num_features=ngf * 2, act=tf.nn.relu)(nn)
    nn = DeConv2d(in_channels=ngf * 2, n_filter=ngf, filter_size=(4,4), strides=(2,2), padding='SAME', b_init=None)(nn)
    nn = BatchNorm2d(num_features=ngf, act=tf.nn.relu)(nn)
    out = DeConv2d(in_channels=ngf, n_filter=nc, filter_size=(4,4), strides=(2,2), padding='SAME', b_init=None, act=tf.nn.tanh)(nn)

    return Model(inputs=ni, outputs=out, name = 'generator_I')


class getGRU(Model):

    def __init__(self, input_size, hidden_size, dropout_value=0):
        super(getGRU,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.grulayer = tl.layers.RNN(
            cell=tf.keras.layers.GRUCell(units=hidden_size,dropout=dropout_value),
            in_channels=input_size,return_last_output=False, return_seq_2d=False, return_last_state=False
        )
        self.dense = Dense(in_channels=self.hidden_size, n_units=self.output_size)
        self.bn = BatchNorm1d(num_features=self.output_size)

    def forward(self, x):
        z = self.grulayer(x)
        z = self.dense(z)
        outputs = self.bn(z)
        return outputs
