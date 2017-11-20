from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plot
import time


if __name__ == '__main__':
    # Load mnist data
    mnist = input_data.read_data_sets("data/", one_hot=True)

    # placeholder for inputs
    xs = tf.placeholder(tf.float32, [None, 784]) / 255.
    ys = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)
    num_filters = [8, 16, 32, 64]
    runtimes = []
    parameters = []

    for filter_size in num_filters:
        print("Filter size: ", filter_size)

        #start timing
        start_time = time.time()

        # conv layer 1
        W_conv_layer1 = tf.Variable(tf.truncated_normal([3, 3, 1, filter_size], stddev=0.1))
        b_conv_layer1 = tf.Variable(tf.constant(0.1, shape=[filter_size]))
        h_conv_layer1 = tf.nn.relu(tf.nn.conv2d
                                   (x_image, W_conv_layer1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_layer1)
        h_pool1 = tf.nn.max_pool(h_conv_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        parameters_1 = (3*3*filter_size) + filter_size

        # conv layer 2
        W_conv2 = tf.Variable(tf.truncated_normal([3, 3, filter_size, (filter_size*2)], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[filter_size*2]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        parameters_2 = (3*3)*(filter_size)*(filter_size*2)

        # fully connected layer
        W_fully_connected_layer = tf.Variable(tf.truncated_normal([7 * 7 * (filter_size*2), 128], stddev=0.1))
        b_fully_connected_layer = tf.Variable(tf.constant(0.1, shape=[128]))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * (filter_size*2)])
        h_fully_connected_layer = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fully_connected_layer) + b_fully_connected_layer)
        h_fully_connected_layer_drop = tf.nn.dropout(h_fully_connected_layer, keep_prob)
        parameters_3 = (7*7*(filter_size*2)*128) + 128

        # Softmax output layer
        W_fc2 = tf.Variable(tf.truncated_normal([128, 10], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
        prediction = tf.nn.softmax(tf.matmul(h_fully_connected_layer_drop, W_fc2) + b_fc2)
        parameters_4 = (128*10)+10

        total_parameters = parameters_1 + parameters_2 + parameters_3 + parameters_4
        parameters.append(total_parameters)

        # the error between prediction and real data
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                      reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

        sess = tf.Session()
        # initialize global tf variables
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(2000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})

        sess.close()
        runtimes.append((time.time()-start_time)/60)

    plot.scatter(parameters, runtimes, alpha = 0.8)
    plot.xlabel('parameters')
    plot.ylabel('runtime (minutes)')
    plot.show()