import tensorflow as tf
import tensorlayer as tl
import numpy
from tensorlayer.layers import Input, Dense, Conv2d, Conv3d, Flatten
from tensorlayer.layers import DeConv2d, BatchNorm1d, BatchNorm2d, BatchNorm3d
from tensorlayer.models import Model


def get_discriminator_I(shape, nc=3, ndf=64):
    ni = Input(shape)
    nn = Conv2d(in_channels=nc, n_filter=ndf, filter_size=4, strides=2, padding=1, b_init=None, act=tf.nn.leaky_relu)(
        ni)

    nn = Conv2d(in_channels=ndf, n_filter=ndf * 2, filter_size=4, strides=2, padding=1, b_init=None)(nn)
    nn = BatchNorm2d(num_features=ndf * 2, act=tf.nn.leaky_relu)(nn)

    nn = Conv2d(in_channels=ndf * 2, n_filter=ndf * 4, filter_size=4, strides=2, padding=1, b_init=None)(nn)
    nn = BatchNorm2d(num_features=ndf * 4, act=tf.nn.leaky_relu)(nn)

    nn = Conv2d(in_channels=ndf * 4, n_filter=ndf * 8, filter_size=4, strides=2, padding=1, b_init=None)(nn)
    nn = BatchNorm2d(num_features=ndf * 8, act=tf.nn.leaky_relu)(nn)

    nn = Conv2d(in_channels=ndf * 8, n_filter=1, filter_size=6, strides=1, padding=0, b_init=None, act=tf.nn.sigmoid)(
        nn)
    D_I = Model(inputs=ni, outputs=nn, name="discriminator_I")
    return D_I


def get_discriminator_V(shape, nc=3, ndf=64, T=16):
    # input is nc*T*96*96
    ni = Input(shape)
    nn = Conv3d(in_channels=nc, n_filter=ndf, filter_size=4, strides=2, padding=1, b_init=None, act=tf.nn.leaky_relu)(
        ni)

    nn = Conv3d(in_channels=ndf, n_filter=ndf * 2, filter_size=4, strides=2, padding=1, b_init=None)(nn)
    nn = BatchNorm3d(num_features=ndf * 2, act=tf.nn.leaky_relu)(nn)

    nn = Conv3d(in_channels=ndf * 2, n_filter=ndf * 4, filter_size=4, strides=2, padding=1, b_init=None)(nn)
    nn = BatchNorm3d(num_features=ndf * 4, act=tf.nn.leaky_relu)(nn)

    nn = Conv3d(in_channels=ndf * 4, n_filter=ndf * 8, filter_size=4, strides=2, padding=1, b_init=None)(nn)
    nn = BatchNorm3d(num_features=ndf * 8, act=tf.nn.leaky_relu)(nn)

    nn = Flatten()(nn)
    # 510425
    nn = Dense(in_channels=(int((ndf * 8) * (T / 16) * 36)), n_units=1, act=tf.nn.sigmoid)(nn)
    D_V = Model(inputs=ni, outputs=nn, name="discriminator_V")
    return D_V


def get_generator(shape, nc=2, ngf=64, nz=60):
    ni = Input(shape)
    nn = DeConv2d(in_channels=nz, n_filter=ngf * 8, filter_size=6, strides=1, padding=0, b_init=None)(ni)
    nn = BatchNorm2d(num_features=ngf * 8, act=tf.nn.relu)(nn)

    nn = DeConv2d(in_channels=ngf * 8, n_filter=ngf * 4, filter_size=4, strides=2, padding=1, b_init=None)(nn)
    nn = BatchNorm2d(num_features=ngf * 4, act=tf.nn.relu)(nn)

    nn = DeConv2d(in_channels=ngf * 4, n_filter=ngf * 2, filter_size=4, strides=2, padding=1, b_init=None)(nn)
    nn = BatchNorm2d(num_features=ngf * 2, act=tf.nn.relu)(nn)

    nn = DeConv2d(in_channels=ngf * 2, n_filter=ngf, filter_size=4, strides=2, padding=1, b_init=None)(nn)
    nn = BatchNorm2d(num_features=ngf, act=tf.nn.relu)(nn)

    nn = DeConv2d(in_channels=ngf, n_filter=nc, filter_size=4, strides=2, padding=1, b_init=None, act=tf.nn.tanh)(nn)
    G_I = Model(inputs=ni, outputs=nn, name="generator_I")
    return G_I

class getGRU(Model):
    def __init__(self, input_size, hidden_size):
        super(getGRU,self).__init__()
        output_size      = input_size
        self.hidden_size = hidden_size

        self.gru    = tl.layers.GRURNN(in_channels=input_size,units=hidden_size)
        self.dense  = Dense(in_channels=hidden_size, n_units=output_size)
        self.bn     = BatchNorm1d(num_features=output_size)
    
    def forward(self, x, n_frames):
        outputs = []
        for i in range(n_frames):
            self.hidden = self.gru(inputs, self.hidden)
            inputs = self.dense(self.hidden)
            outputs.append(inputs)
        outputs = [ self.bn(elm) for elm in outputs ]
        outputs = tf.stack(outputs)
        return outputs
    
    def initHidden(self, batch_size):
        self.hidden = numpy.array(tf.zeros(batch_size, self.hidden_size))
