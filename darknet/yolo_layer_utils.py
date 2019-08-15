from keras.layers import Permute, Reshape, Lambda, add
from keras.layers import Conv2D, BatchNormalization, LeakyReLU
from keras import regularizers, initializers
import tensorflow as tf

def reorg(input_tensor, stride):
    _, h, w, c = input_tensor.get_shape().as_list() 

    channel_first = Permute((3, 1, 2))(input_tensor)
    
    reshape_tensor = Reshape((c // (stride ** 2), h, stride, w, stride))(channel_first)
    permute_tensor = Permute((3, 5, 1, 2, 4))(reshape_tensor)
    target_tensor = Reshape((-1, h // stride, w // stride))(permute_tensor)
    
    channel_last = Permute((2, 3, 1))(target_tensor)
    return Reshape((h // stride, w // stride, -1))(channel_last)

def conv_batch_lrelu(input_tensor, numfilter, dim, strides=1):
    input_tensor = Conv2D(numfilter, (dim, dim), strides=strides, padding='same',
                        kernel_regularizer=regularizers.l2(0.0005),
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        use_bias=False
                    )(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    return LeakyReLU(alpha=0.1)(input_tensor)

def NetworkInNetwork(input_tensor, dims):
    for d in dims:
        input_tensor = conv_batch_lrelu(input_tensor, *d)
    return input_tensor

def Upsample(stride):
    def _upsample(x):
        h, w = x.get_shape().as_list()[1:3]
        return tf.image.resize_images(x, [h * 2, w * 2], align_corners=True)

    return Lambda(_upsample)

def residual(input_tensor, num_blocks, dims):
    out = input_tensor
    for _ in range(num_blocks):
        out = add([NetworkInNetwork(out, dims), out])
    return out