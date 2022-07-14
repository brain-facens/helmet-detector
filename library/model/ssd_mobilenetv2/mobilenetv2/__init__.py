from keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Add, Input
from keras.models import Model

import tensorflow as tf


def conv_block(_input, _filter:int, _kernel:int, _strides:int, _activation:bool):

    __x = Conv2D(filters=_filter, kernel_size=_kernel, strides=_strides, padding="same")(_input)
    __x = BatchNormalization()(__x)
    if _activation:
        __x = tf.nn.relu6(__x)
    return __x

def bottleneck(_input, _filter:int, _strides:int, _t:int, _a:int, _add:bool):
    
    __tchannels = int(_input.get_shape()[-1] * _t)
    __achannels = int(_filter * _a)

    __x = conv_block(_input, __tchannels, 1, 1, True)
    
    __x = DepthwiseConv2D(kernel_size=3, strides=_strides, padding="same")(__x)
    __x = BatchNormalization()(__x)
    __x = tf.nn.relu6(__x)

    __x = conv_block(__x, __achannels, 1, 1, False)

    if _add:
        __x = Add()([__x, _input])
    
    return __x

def inverted_residuals(_input, _filter:int, _strides:int, _expasion_factor:int, _alpha:float, _repeat:int):

    __x = bottleneck(_input, _filter, _strides, _expasion_factor, _alpha, False)
    for _ in range(1, _repeat):
        __x = bottleneck(__x, _filter, 1, _expasion_factor, _alpha, True)
    return __x

def mobilenetv2_architecture(_input_shape:tuple=(646,640,1)):

    __input = Input(shape=_input_shape)

    __x = conv_block(__input, 32, 3, 2, True)
    
    __x = inverted_residuals(__x, 16, 1, 1, 1, 1)
    __x = inverted_residuals(__x, 24, 2, 6, 1, 2)
    __x = inverted_residuals(__x, 32, 2, 6, 1, 3)
    __x = inverted_residuals(__x, 64, 2, 6, 1, 4)
    __x = inverted_residuals(__x, 96, 1, 6, 1, 3)
    __x = inverted_residuals(__x, 160, 2, 6, 1, 3)
    __x = inverted_residuals(__x, 320, 1, 6, 1, 1)

    __x = conv_block(__x, 1280, 1, 1, True)

    return Model(inputs=__input, outputs=__x, name="mobilenetv2")
