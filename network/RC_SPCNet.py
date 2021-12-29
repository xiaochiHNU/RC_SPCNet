# coding:utf-8
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, concatenate, Activation,  ZeroPadding2D, Conv2DTranspose
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, AveragePooling2D
from keras.layers.merge import Concatenate, add
from keras import layers
from keras.models import Model
from keras.utils import plot_model

# Backend
from keras import backend as K

class Subpixel(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='same',
                 data_format=None,
                 strides=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize, a, b, int(c/int(r*r)),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], int(unshifted[3]/(self.r*self.r)))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters']/=self.r*self.r
        config['r'] = self.r
        return config


def residual_conv1(input_tensor, filter, stage, block, strides=(2, 2)):  # strides

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter, (3, 3), strides=strides, padding='same',
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('elu')(x)

    x = Conv2D(filter, (3, 3), padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    if K.int_shape(x)[2]  != K.int_shape(input_tensor)[2]:
        shortcut = Conv2D(filter, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
        x = add([x, shortcut])
    else:
        x = add([x, input_tensor])

    x = Activation('elu')(x)
    return x

def residual_conv2(input_tensor, filter, stage, block, dilation_rate=(1, 1)):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter, (3, 3), padding='same',  dilation_rate=dilation_rate,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('elu')(x)

    x = Conv2D(filter, (3, 3),
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    if K.int_shape(x)[3]  != K.int_shape(input_tensor)[3]:
        shortcut = Conv2D(filter, (1, 1),
                      name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
        x = add([x, shortcut])
    else:
        x = add([x, input_tensor])

    x = Activation('elu')(x)
    return x

def residual_conv3(input_tensor, filters, stage, block, dilation_rate = (1, 1)):

    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), dilation_rate = dilation_rate,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('elu')(x)

    x = Conv2D(filters2, (3, 3), padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    if K.int_shape(x)[3]  != K.int_shape(input_tensor)[3]:
        shortcut = Conv2D(filters2, (1, 1),
                      name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
        x = add([x, shortcut])
    else:
        x = add([x, input_tensor])

    x = Activation('elu')(x)
    return x


def residual_conv4(input_tensor, filters, stage, block, dilation_rate = (1, 1)):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), dilation_rate=dilation_rate,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('elu')(x)

    x = Conv2D(filters2, (3, 3), padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('elu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    if K.int_shape(x)[3]  != K.int_shape(input_tensor)[3]:
        shortcut = Conv2D(filters3, (1, 1),
                      name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
        x = add([x, shortcut])
    else:
        x = add([x, input_tensor])

    x = Activation('elu')(x)
    return x

def RC_SPCNet(input_shape):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    inputs = Input(shape=input_shape)     # inputs 512*512*1

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv')(inputs)          # 512 -> 256
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('elu')(x)
    net1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)                                         # branch_0 256*256*64  第1个skip connections

    # Block 1   3 times
    x = residual_conv1(net1, 128, stage=1, block='a', strides=(2, 2))       # x  256 -> 128
    x = residual_conv2(x, 128, stage=1, block='b')
    net2 = residual_conv2(x, 128, stage=1, block='c')

    # Block 2   3 times
    x = residual_conv1(net2, 256, stage=2, block='a', strides=(2, 2))     # x  128 -> 64
    x = residual_conv2(x, 256, stage=2, block='b')
    x = residual_conv2(x, 256, stage=2, block='c')

    # Block 3   6 times
    x = residual_conv2(x, 512, stage=3, block='a',  dilation_rate=(2, 2))     # x
    x = residual_conv2(x, 512, stage=3, block='b')
    x = residual_conv2(x, 512, stage=3, block='c')
    x = residual_conv2(x, 512, stage=3, block='d')
    x = residual_conv2(x, 512, stage=3, block='e')
    x = residual_conv2(x, 512, stage=3, block='f')

    # Block 4   3 times
    x = residual_conv3(x, [512, 1024], stage=4, block='a', dilation_rate=(2, 2))
    x = residual_conv3(x, [512, 1024], stage=4, block='b')
    x = residual_conv3(x, [512, 1024], stage=4, block='c')

    # Block 5
    x = residual_conv4(x, [512, 1024, 2048], stage=5, block='a', dilation_rate=(2, 2))

    # Block 6
    net3 = residual_conv4(x, [1024, 2048, 4096], stage=6, block='a', dilation_rate=(2, 2))

    # net1 sub_pixel  upsampling factor 2
    net1 = Subpixel(16, (1, 1), 2, activation='elu')(net1)
    net1 = BatchNormalization()(net1)
    net1 = Conv2D(32, (1, 1), activation='elu', padding='same')(net1)
    net1 = BatchNormalization()(net1)

    # net2 sub_pixel upsampling factor 4
    net2 = Subpixel(32, (1, 1), 4, activation='elu')(net2)
    net2 = BatchNormalization()(net2)

    # net3 sub_pixel upsampling factor 8
    net3 = Subpixel(32, (1, 1), 8, activation='elu')(net3)
    net3 = BatchNormalization()(net3)

    # context information
    net = add([net1, net2, net3])

    net = Conv2D(16, (3, 3), activation='elu', padding='same')(net)
    net = BatchNormalization()(net)
    net = Conv2D(8, (3, 3), activation='elu', padding='same')(net)
    net = BatchNormalization()(net)

    # softmax
    predictions = Conv2D(2, (1, 1), activation='elu')(net)
    predictions = BatchNormalization()(predictions)
    predictions = Activation('softmax')(predictions)

    # Create model.
    model = Model(inputs=[inputs], outputs=[predictions])
    #plot_model(model, to_file='model_RC_SPCNet.png', show_shapes=True)   # 绘制RC_SPCNet模型

    return model


if __name__ == '__main__':
    model = RC_SPCNet(input_shape=[512, 512, 1])
    model.summary()
