import tensorflow as tf


def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)


def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d_layer(input, out_channels, kernel_size, stride, pad='SAME', keep_prob_=0.8, bn=True, bias=False,is_training=True,activation=None):
    with tf.name_scope("conv2d"):
        conv = tf.layers.conv2d(input, out_channels, kernel_size, strides=stride, padding=pad,
                                use_bias=bias,activation=activation,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        if bn:
            batch_norm = tf.layers.batch_normalization(conv,training=is_training,gamma_initializer=tf.truncated_normal_initializer(stddev=0.1))
        else:
            batch_norm = conv

        out = tf.nn.relu(batch_norm)

        #return tf.layers.dropout(out, rate=keep_prob_)
        return out


def conv3d_layer(input, out_channels, kernel_size, stride, pad='SAME', keep_prob_=0.8, bn=True, bias=False,is_training=True,activation=None):
    with tf.name_scope("conv3d"):
        conv = tf.layers.conv3d(input, out_channels, kernel_size, strides=stride, padding=pad,
                                use_bias=bias,activation=activation,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

        if bn:
            batch_norm = tf.layers.batch_normalization(conv,training=is_training,gamma_initializer=tf.truncated_normal_initializer(stddev=0.1))
        else:
            batch_norm = conv
        out = tf.nn.relu(batch_norm)

        #return tf.layers.dropout(out, rate=keep_prob_)
        return out


def maxpool2d_layer(input, ksize, stride, pad='VALID'):
    return tf.layers.max_pooling2d(input, ksize, stride, padding=pad)


def maxpool3d_layer(input, ksize, stride, pad='VALID'):
    return tf.layers.max_pooling3d(input, ksize, stride, padding=pad)


def deconv2d_layer(input, out_channels, kernel_size, stride, pad='VALID', keep_prob_=0.8, bn=True, bias=False,is_training=True,activation=None):
    with tf.name_scope("deconv2d"):

        deconv = tf.layers.conv2d_transpose(input, out_channels, kernel_size, strides=stride, padding=pad,
                                            use_bias=bias,activation=activation,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

        if bn:
            batch_norm = tf.layers.batch_normalization(deconv,training=is_training,gamma_initializer=tf.truncated_normal_initializer(stddev=0.1))
        else:
            batch_norm = deconv

        out = tf.nn.relu(batch_norm)

        #return tf.layers.dropout(out, rate=keep_prob_)
        return out


def deconv3d_layer(input, out_channels, kernel_size, stride, pad='VALID', keep_prob_=0.8, bn=True, bias=True,is_training=True,activation=None):
    with tf.name_scope("deconv3d"):
        deconv = tf.layers.conv3d_transpose(input, out_channels, kernel_size, strides=stride,
                                            padding=pad, use_bias=bias,activation=activation,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

        if bn:
            batch_norm = tf.layers.batch_normalization(deconv,training=is_training,gamma_initializer=tf.truncated_normal_initializer(stddev=0.1))
        else:
            batch_norm = deconv

        out = tf.nn.relu(batch_norm)

        #return tf.layers.dropout(out, rate=keep_prob_)
        return out


def crop_and_concat(x1, x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)


def crop_and_concat3d(x1, x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2,
                   (x1_shape[3] - x2_shape[3]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 4)


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def cross_entropy(y_, output_map):
    return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")


def global_average_pool2d(input):
    return tf.reduce_mean(input, [1, 2])


def global_average_pool3d(input):
    return tf.reduce_mean(input, [1, 2, 3])