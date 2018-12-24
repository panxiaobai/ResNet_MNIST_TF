import tensorflow as tf
import layers as layer


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def maxpool_layer(input,stride,size):
    return tf.nn.max_pool(input, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='VALID')

def fc_layer(input,output_num):
    num_in=input.get_shape().as_list()[-1]
    w=weight_variable([num_in,output_num])
    b=bias_variable([output_num])

    #input_flat=tf.reshape(input,[-1])
    fc_output=tf.nn.relu(tf.matmul(input,w)+b)
    return fc_output

def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h

def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    '''
    mean, var = tf.nn.moments(conv, axes=[0, 1, 2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")

    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)
    '''
    batch_norm=tf.layers.batch_normalization(conv,training=True,gamma_initializer=tf.truncated_normal_initializer(stddev=0.1))
    out = tf.nn.relu(batch_norm)

    return out

def residual_block(input,base_depth,down_sample,training):
    input_depth=input.get_shape().as_list()[3]

    if down_sample:
        conv_down=layer.conv2d_layer(input,base_depth,[1,1],[1,1],is_training=training)
        conv=layer.conv2d_layer(conv_down,base_depth,[3,3],[2,2],is_training=training)
        conv_up=layer.conv2d_layer(conv,base_depth*4,[1,1],[1,1],is_training=training)
        input_layer=layer.conv2d_layer(input,base_depth*4,[1,1],[2,2],is_training=training)
        res=conv_up+input_layer
    else:
        conv_down = layer.conv2d_layer(input, base_depth, [1, 1], [1, 1], is_training=training)
        conv = layer.conv2d_layer(conv_down, base_depth, [3, 3], [1, 1], is_training=training)
        conv_up = layer.conv2d_layer(conv, base_depth * 4, [1, 1], [1, 1], is_training=training)
        if input_depth==(base_depth*4):
            input_layer=input
        else:
            input_layer=layer.conv2d_layer(input,base_depth*4,[1,1],[1,1], is_training=training)
        res=conv_up+input_layer

    return res

def resnet_50(x,input_channel,training):
    layers = []
    with tf.variable_scope('conv1'):
        conv1= layer.conv2d_layer(x,64,[7,7],[2,2], is_training=training)
        layers.append(conv1)
        print(str(conv1.get_shape()))

    with tf.variable_scope('conv2'):
        conv2_maxpool= layer.maxpool2d_layer(conv1,[3,3],[2,2])
        for i in range(3):
            if i==0:
                conv2 = residual_block(conv2_maxpool, 64, False,training=training)
            else:
                conv2 = residual_block(conv2, 64, False,training=training)
            layers.append(conv2)
        print(str(conv2.get_shape()))

    with tf.variable_scope('conv3'):
        for i in range(4):
            if i==0:
                conv3 = residual_block(conv2, 128, True,training=training)
            else:
                conv3 = residual_block(conv3, 128, False,training=training)
            layers.append(conv3)
        print(str(conv3.get_shape()))

    with tf.variable_scope('conv4'):
        for i in range(6):
            if i==0:
                conv4 = residual_block(conv3, 256, False,training=training)
            else:
                conv4 = residual_block(conv4, 256, False,training=training)
            layers.append(conv4)
        print(str(conv4.get_shape()))

    with tf.variable_scope('conv5'):
        for i in range(3):
            if i==0:
                conv5 = residual_block(conv4, 512, False,training=training)
            else:
                conv5 = residual_block(conv5, 512, False,training=training)
            layers.append(conv5)
        print(str(conv5.get_shape()))

    with tf.variable_scope('global_average_pool'):
        gap = tf.reduce_mean(conv5, [1, 2])
        layers.append(gap)

    with tf.variable_scope('fc'):
        fc=fc_layer(gap,1024)
        layers.append(fc)

    with tf.variable_scope('softmax'):
        out = softmax_layer(fc, [1024,10])
        layers.append(out)

    return out