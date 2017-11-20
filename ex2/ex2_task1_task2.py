from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plot


def performance_wrt_learning_rates(learning_rates):
    for lr in learning_rates:
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        print("learning_rate: ", lr)
        y_plot = []
        x_plot = []

        for i in range(2000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            session.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, learning_rate: lr, keep_prob: 0.5})
            if i % 10 == 0:
                print("Step: ", i)
                correct_prediction = tf.equal(tf.argmax(session.run(pred, feed_dict={xs: mnist.validation.images, keep_prob: 1}), 1), tf.argmax(mnist.validation.labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                accuracy = session.run(accuracy, feed_dict={xs: mnist.validation.images,
                                                            ys: mnist.validation.labels, keep_prob: 1})
                print("Validation accuracy: ", accuracy)
                x_plot.append(i)
                y_plot.append(accuracy)

        plot.plot(x_plot, y_plot, label=lr)
        session.close()

    plot.xlabel("EPOCHS")
    plot.ylabel("ACCURACY")
    plot.legend(loc="lower right")
    plot.show()


if __name__ == '__main__':
    # Load mnist data
    mnist = input_data.read_data_sets("data/", one_hot=True)
    
    learning_rate = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [None, 784]) / 255.  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)

    # conv layer 1
    W_conv_layer1 = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1))  # patch 3x3, in size 1, out size 16
    b_conv_layer1 = tf.Variable(tf.constant(0.1, shape=[16]))
    h_conv_layer1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv_layer1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_layer1)  # output size 28x28x16
    h_pool1 = tf.nn.max_pool(h_conv_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # output size 14x14x16

    # conv layer 2
    W_conv_layer2 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1))  # patch 3x3, in size 16, out size 32
    b_conv_layer2 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv_layer2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv_layer2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_layer2)  # output size 14x14x32
    h_pool2 = tf.nn.max_pool(h_conv_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # output size 7x7x32

    # fully connected layer
    W_fully_connected_layer = tf.Variable(tf.truncated_normal([7 * 7 * 32, 128], stddev=0.1))
    b_fully_connected_layer = tf.Variable(tf.constant(0.1, shape=[128]))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])
    h_fully_connected_layer = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fully_connected_layer) + b_fully_connected_layer)
    h_fully_connected_layer_drop = tf.nn.dropout(h_fully_connected_layer, keep_prob)

    # Softmax output layer
    W_fc2 = tf.Variable(tf.truncated_normal([128, 10], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    pred = tf.nn.softmax(tf.matmul(h_fully_connected_layer_drop, W_fc2) + b_fc2)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(pred),
                                                  reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    performance_wrt_learning_rates([0.1, 0.01, 0.001, 0.0001])