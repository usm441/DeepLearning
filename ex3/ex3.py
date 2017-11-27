from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def compute_accuracy(sess, validation_xs):
    global output
    gen_img = sess.run(output, feed_dict={xs: validation_xs, keep_prob:1})
    correct_prediction = tf.equal(gen_img, validation_xs)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: validation_xs, keep_prob:1})
    return result


def plot_images(images, cls_true, label):
    img_shape = (28, 28)
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        xlabel = "True: {0}".format(cls_true[i])
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(label)
    plt.show()


def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_layer(input,
               num_input_channels,
               num_filters,
               filter_size=3,
               use_pooling=True,
               use_trans_conv=False,):

    if use_trans_conv:
        input = tf.layers.conv2d_transpose(inputs=input,
                                           filters=num_input_channels,
                                           kernel_size=2,
                                           strides=2,
                                           padding='same')

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = weight(shape=shape)
    # Create new biases, one for each filter.
    biases = bias([num_filters])
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases
    layer = tf.nn.relu(layer)
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    return layer




if __name__ == '__main__':
    # Load mnist data
    data = input_data.read_data_sets("data/", one_hot=True)
    data_class = np.array([label.argmax() for label in data.validation.labels])
    images = data.validation.images[0:9]
    cls_true = data_class[0:9]

    # # define placeholder for inputs to network
    learning_rate = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)

    # conv pooling layer 1
    layer1 = conv_layer(x_image, 1, 8)

    # conv pooling layer 2
    layer2 = conv_layer(layer1, 8, 4)

    # conv layer 3
    layer3 = conv_layer(layer2, 4, 2, use_pooling=False, use_trans_conv=False)

    # trans conv layer 4
    layer4 = conv_layer(layer3, 2, 4, use_pooling=False, use_trans_conv=True)

    # trans conv layer 5
    layer5 = conv_layer(layer4, 4, 8, use_pooling=False, use_trans_conv=True)

    # output conv
    output = conv_layer(layer5, 8, 1, use_pooling=False, use_trans_conv=False)

    output = tf.reshape(output, [tf.shape(output)[0], 784])
    loss = tf.reduce_mean(tf.square(output - xs))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    learning_rates = [0.1, 0.01, 0.001]
    gen_imgs = []
    for rate in learning_rates:
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        x_plot = []
        y_plot = []
        for i in range(1000):
            batch_xs, batch_ys = data.train.next_batch(64)
            sess.run(train_step, feed_dict={xs: batch_xs, learning_rate: rate, keep_prob: 0.5})
            if i%50 == 0:
                print("Step: ", i)
                accuracy = compute_accuracy(sess, data.validation.images)
                print("Validation accuracy: ", accuracy)
                x_plot.append(i)
                y_plot.append(accuracy)
        plt.plot(x_plot, y_plot, label=rate)
        gen_imgs.append(sess.run(output, feed_dict={xs: data.validation.images[0:9], keep_prob: 1}))
        sess.close()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()

    plot_images(images=images, label="Original", cls_true=cls_true)
    i=0
    for img in gen_imgs:
        img = np.reshape(img, [9, 28, 28])
        plot_images(images=img, label="Learning rate: " + str(learning_rates[i]), cls_true=cls_true)
        i += 1

