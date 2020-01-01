import tensorflow as tf
import tensorlayer as tl
import numpy
from tensorlayer.layers import Input, Dense, Conv2d, Conv3d, Flatten
from tensorlayer.layers import DeConv2d, BatchNorm1d, BatchNorm2d, BatchNorm3d
from tensorlayer.models import Model

class discriminator_I(Model):
    
    def __init__(self, nc=3, ndf=64):
        super(discriminator_I,self).__init__()

        self.conv1 = Conv2d(in_channels=nc, n_filter=ndf, filter_size=4, strides=2, padding=1, b_init=None, act=tf.nn.leaky_relu)
        self.conv2 = Conv2d(in_channels=ndf, n_filter=ndf * 2, filter_size=4, strides=2, padding=1, b_init=None)
        self.batch1 = BatchNorm2d(num_features=ndf * 2, act=tf.nn.leaky_relu)
        self.conv3 = Conv2d(in_channels=ndf * 2, n_filter=ndf * 4, filter_size=4, strides=2, padding=1, b_init=None)
        self.batch2 = BatchNorm2d(num_features=ndf * 4, act=tf.nn.leaky_relu)
        self.conv4 = Conv2d(in_channels=ndf * 4, n_filter=ndf * 8, filter_size=4, strides=2, padding=1, b_init=None)
        self.batch3 = BatchNorm2d(num_features=ndf * 8, act=tf.nn.leaky_relu)
        self.conv5 = Conv2d(in_channels=ndf * 8, n_filter=1, filter_size=6, strides=1, padding=0, b_init=None, act=tf.nn.sigmoid)
    
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

        self.conv1 = Conv3d(in_channels=nc, n_filter=ndf, filter_size=4, strides=2, padding=1, b_init=None, act=tf.nn.leaky_relu)
        self.conv2 = Conv3d(in_channels=ndf, n_filter=ndf * 2, filter_size=4, strides=2, padding=1, b_init=None)
        self.batch1 = BatchNorm3d(num_features=ndf * 2, act=tf.nn.leaky_relu)
        self.conv3 = Conv3d(in_channels=ndf * 2, n_filter=ndf * 4, filter_size=4, strides=2, padding=1, b_init=None)
        self.batch2 = BatchNorm3d(num_features=ndf * 4, act=tf.nn.leaky_relu)
        self.conv4 = Conv3d(in_channels=ndf * 4, n_filter=ndf * 8, filter_size=4, strides=2, padding=1, b_init=None)
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

class generator_I(Model):

    def __init__(self, nc=2, ngf=64, nz=60):     
        super(generator_I,self).__init__()
    
        self.deconv1 = DeConv2d(in_channels=nz, n_filter=ngf * 8, filter_size=6, strides=1, padding=0, b_init=None)
        self.batch1 = BatchNorm2d(num_features=ngf * 8, act=tf.nn.relu)
        self.deconv2 = DeConv2d(in_channels=ngf * 8, n_filter=ngf * 4, filter_size=4, strides=2, padding=1, b_init=None)
        self.batch2 = BatchNorm2d(num_features=ngf * 4, act=tf.nn.relu)
        self.deconv3 = DeConv2d(in_channels=ngf * 4, n_filter=ngf * 2, filter_size=4, strides=2, padding=1, b_init=None)
        self.batch3 = BatchNorm2d(num_features=ngf * 2, act=tf.nn.relu)
        self.deconv4 = DeConv2d(in_channels=ngf * 2, n_filter=ngf, filter_size=4, strides=2, padding=1, b_init=None)
        self.batch4 = BatchNorm2d(num_features=ngf, act=tf.nn.relu)
        self.deconv5 = DeConv2d(in_channels=ngf, n_filter=nc, filter_size=4, strides=2, padding=1, b_init=None, act=tf.nn.tanh)
    
    def forward(self, x):
        z = self.deconv1(x)
        z = self.batch1(z)
        z = self.deconv2(z)
        z = self.batch2(z)
        z = self.deconv3(z)
        z = self.batch3(z)
        z = self.deconv4(z)
        z = self.batch4(z)
        out  = self.deconv5(z)
        return out

class getGRU(Model):
    def __init__(self, input_size, hidden_size):
        super(getGRU,self).__init__()
        output_size = input_size
        self.hidden_size = hidden_size

        self.gru = tl.layers.GRURNN(in_channels=input_size,units=hidden_size)
        self.dense = Dense(in_channels=hidden_size, n_units=output_size)
        self.bn = BatchNorm1d(num_features=output_size)
    
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
        self.hidden = tf.Variable(tf.zeros(batch_size, self.hidden_size))
