import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np

########################################################################################################################
EPSILON = tf.keras.backend.epsilon()
act = 'relu'

def _conv_norm_act(filters, kernel=3, depth=2):
    def _module(x):
        x = Conv2D(filters, kernel, padding='same')(x)
        x = BatchNormalization()(x)
        return Activation(act)(x)
    def main(x):
        for _ in range(depth):
            x = _module(x)
        return x
    return main

def _dense_block(filters, kernel=3):
    def main(x):
        c1 = _conv_norm_act(filters, kernel)(x)
        c2 = _conv_norm_act(int(filters / 4), kernel)(c1)
        c3 = _conv_norm_act(int(filters / 4), kernel)(tf.concat([c1, c2], -1))
        c4 = _conv_norm_act(int(filters / 4), kernel)(tf.concat([c1, c2, c3], -1))
        feature = tf.concat([c1, c2, c3, c4], -1)
        return tf.nn.relu(BatchNormalization()(feature))
    return main

########################################################################################################################
def KiUNet(num_class, depth=2, mode=None):
    assert depth > 1
    filters = [pow(2, i+4) for i in range(depth)] # 32
    upscale = [pow(4, j) for j in range(1, depth+2)]
    downscale = (1/np.array(upscale)).tolist()
    K_skip, U_skip = [], []
    #
    def call(x):
        K, U = x, x
        for i in range(depth):
            K = decoding(filters[i])(K)
            U = encoding(filters[i])(U)
            if i != depth-1:
                K_skip.append(K), U_skip.append(U)  # save skip connection
            K, U = CRFB(filters[i], 3, upscale[i], downscale[i])(K, U)

        for i in reversed(range(depth-1)):
            U = decoding(filters[i])(U)
            K = encoding(filters[i])(K)
            if i != 0:
                K, U = CRFB(filters[i], 3, upscale[i], downscale[i])(K, U)
                K, U = K + K_skip.pop(), U + U_skip.pop()
        K, U = encoding(16)(K), decoding(16)(U)
        logit = Conv2D(num_class, 1)(tf.concat([K, U], -1))
        prob = Softmax(-1)(logit)
        return prob

    def CRFB(filters, kernel=3, upscale=None, downscale=None):
        def main(from_Kite, from_UNet):
            to_Kite = decoding(filters, kernel, upscale)(from_UNet)
            to_UNet = encoding(filters, kernel, downscale)(from_Kite)
            return (from_Kite+to_Kite), (from_UNet+to_UNet)
        return main

    def encoding(filters, kernel=3, scale=2, mode=None):
        conv = _dense_block if mode == 'dense' else Conv2D
        def main(x):
            x = conv(filters, kernel, padding='same')(x)
            x = DepthwiseConv2D(kernel, padding='same')(x)
            x = DepthwiseConv2D(kernel, padding='same')(x)
            if scale < 1: # in CRFB
                x = tf.image.resize(x, [int(x.shape[1]*scale), int(x.shape[2]*scale)])
            else:
                x = AveragePooling2D(scale, scale)(x)
            x = tf.nn.relu(BatchNormalization()(x))
            return x
        return main

    def decoding(filters, kernel=3, scale=2, mode=None):
        div = 4
        assert scale > 1
        conv = _dense_block if mode == 'dense' else Conv2D
        def main(x):
            x = conv(filters, kernel, padding='same')(x)
            x = DepthwiseConv2D(kernel, padding='same')(x)
            for j in range(1, 4):
                x += Conv2D(filters, kernel, dilation_rate=j, padding='same', groups=div)(x)
            x = UpSampling2D(scale)(x)
            x = tf.nn.relu(BatchNormalization()(x))
            return x
        return main
    return call