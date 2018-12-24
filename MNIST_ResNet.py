import tensorflow as tf
import ResNet_dila
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "minst_resnet50_model.ckpt"


def train_resnet50(mnist):
    batch_size = 128

    X = tf.placeholder("float", [batch_size, 28, 28, 1])
    Y = tf.placeholder("float", [batch_size, 10])
    learning_rate = tf.placeholder("float", [])
    global_step = tf.Variable(0, trainable=False)
    # ResNet Models
    net = ResNet_dila.atrous_resnet_50(X,1)

    cross_entropy = -tf.reduce_sum(Y * tf.log(net))
    opt = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(cross_entropy,global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.Session()

    init=tf.global_variables_initializer()

    sess.run(init)
    saver = tf.train.Saver()



    for i in range(5000):
        #print(i)
        xs, ys = mnist.train.next_batch(batch_size)
        xx=np.reshape(xs,[batch_size,28,28,1])
        yy=np.reshape(ys, [batch_size, 10])
        #xs=tf.reshape(xs,[batch_size,28,28,1])
        #ys = tf.reshape(ys, [batch_size, 10])
        _, loss_value, step ,acc= sess.run([train_op, cross_entropy, global_step,accuracy],
                                       feed_dict={X: xx, Y: yy,learning_rate: 0.001})
        print("After %d training step(s), loss on training "
              "batch is %g.and accuracy is %g." % (step, loss_value, acc))
        # 每1000轮保存一次模型
        if i % 1000 == 0:
            # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失
            # 函数大小。通过损失函数的大小可以大概了解训练的情况。在验证数
            # 据集上正确率的信息会有一个单独的程序来生成
            print("After %d training step(s), loss on training "
                  "batch is %g.and accuracy is %g." % (step, loss_value,acc))
            # 保存当前的模型。注意这里给出了global_step参数，这样可以让每个
            # 被保存的模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000”，
            # 表示训练1000轮之后得到的模型。
            saver.save(
                sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                global_step=global_step
            )
    sess.close()


def test_resnet50(mnist):
    batch_size = 128

    X = tf.placeholder("float", [batch_size, 28, 28, 1])
    Y = tf.placeholder("float", [batch_size, 10])

    global_step = tf.Variable(0, trainable=False)
    # ResNet Models
    net = ResNet.resnet_50(X, 1)

    cross_entropy = -tf.reduce_sum(Y * tf.log(net))

    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    init = tf.global_variables_initializer()


    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, MODEL_SAVE_PATH + MODEL_NAME)
        for i in range(0, 200):
            xs, ys = mnist.test.next_batch(batch_size)
            xx = np.reshape(xs, [batch_size, 28, 28, 1])
            yy = np.reshape(ys, [batch_size, 10])
            # xs=tf.reshape(xs,[batch_size,28,28,1])
            # ys = tf.reshape(ys, [batch_size, 10])
            step,loss_value, acc = sess.run([global_step,cross_entropy, accuracy],
                                       feed_dict={X: xx, Y: yy})
            accuracy_summary = tf.scalar_summary("accuracy", accuracy)
            print("After %d training step(s), loss on training "
                  "batch is %g.and accuracy is %g." % (step, loss_value, acc))

def main(argv=None):
    print("begin:")
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    print("loadDataFinish,begin train....")
    train_resnet50(mnist)

    #print("loadDataFinish,begin test...")
    #test_resnet50(mnist)

if __name__ == "__main__":
    tf.app.run()