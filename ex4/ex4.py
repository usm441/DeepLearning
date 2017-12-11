import h5py
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        with h5py.File("cell_data.h5", "r") as data:
            self.train_images = [data["/train_image_{}".format(i)][:] for i in range(28)]
            self.train_labels = [data["/train_label_{}".format(i)][:] for i in range(28)]
            self.test_images = [data["/test_image_{}".format(i)][:] for i in range(3)]
            self.test_labels = [data["/test_label_{}".format(i)][:] for i in range(3)]

        self.input_resolution = 300
        self.label_resolution = 116

        self.offset = (300 - 116) // 2

    def get_train_image_list_and_label_list(self):
        n = random.randint(0, len(self.train_images) - 1)
        x = random.randint(0, (self.train_images[n].shape)[1] - self.input_resolution - 1)
        y = random.randint(0, (self.train_images[n].shape)[0] - self.input_resolution - 1)
        image = self.train_images[n][y:y + self.input_resolution, x:x + self.input_resolution, :]

        x += self.offset
        y += self.offset
        label = self.train_labels[n][y:y + self.label_resolution, x:x + self.label_resolution]

        return [image], [label]

    def get_test_image_list_and_label_list(self):
        coord_list = [[0, 0], [0, 116], [0, 232],
                      [116, 0], [116, 116], [116, 232],
                      [219, 0], [219, 116], [219, 232]]

        image_list = []
        label_list = []

        for image_id in range(3):
            for y, x in coord_list:
                image = self.test_images[image_id][y:y + self.input_resolution, x:x + self.input_resolution, :]
                image_list.append(image)
                x += self.offset
                y += self.offset
                label = self.test_labels[image_id][y:y + self.label_resolution, x:x + self.label_resolution]
                label_list.append(label)

        return image_list, label_list


def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def max_pooling_layer(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def conv_transpose_layer(x):
    output_filters = int(int(x.get_shape()[3]) / 2)
    return tf.layers.conv2d_transpose(inputs=x, filters=output_filters,
                                          kernel_size=2,
                                          strides=2,
                                          padding='VALID')


def conv_layer(input_image, num_filters,
                                 filter_size=3):
    num_input_channels = int(input_image.get_shape()[3])
    weight_shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = weight(shape=weight_shape)
    biases = bias([num_filters])

    layer = tf.nn.conv2d(input_image, weights, strides=[1, 1, 1, 1], padding='VALID')
    layer = tf.nn.relu(layer + biases)

    return layer


def crop_and_merge_layers(layer_to_be_cropped, second_layer):
    size = int(second_layer.get_shape()[1])
    cropped_tensor_1 = tf.image.resize_image_with_crop_or_pad(layer_to_be_cropped, size, size)
    return tf.concat([second_layer, cropped_tensor_1], 3)


def plot_image(im):
    figure = plt.figure()
    ax = plt.Axes(figure, [0., 0., 1., 1.])
    figure.add_axes(ax)
    ax.imshow(im, cmap='gray')
    plt.show()


def get_accuracy(prediction, labels):
    prediction = np.argmax(prediction, axis=3)

    correct_pix = np.sum(prediction == labels)
    inccorect_pix = np.sum(prediction != labels)
    total_pixels = correct_pix + inccorect_pix

    accuracy = correct_pix / (total_pixels + inccorect_pix)
    return accuracy


def train_and_plot():
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    y_plot_validation = []
    y_plot_train = []
    x_plot = []

    for i in range(40000):
        batch_xs, batch_ys = data.get_train_image_list_and_label_list()
        output, _ = sess.run([output_conv, train_step], feed_dict={x_image: batch_xs, y_label: batch_ys})

        if i % 5 == 0:
            print("Step: ", i)
            training_accuracy = get_accuracy(output, batch_ys)
            valid_images, valid_labels = data.get_test_image_list_and_label_list()

            y_prediction = sess.run(output_conv, feed_dict={x_image: valid_images, y_label: valid_labels})
            validation_accuracy = get_accuracy(y_prediction, valid_labels)
            print("Validation accuracy: ", validation_accuracy)

            # utils.write_to_file('epochs', str(i))
            x_plot.append(i)

            # utils.write_to_file('validation_accuracy', str(validation_accuracy))
            y_plot_validation.append(validation_accuracy)

            # utils.write_to_file('training_Accuracy', str(training_accuracy))
            y_plot_train.append(training_accuracy)

    plt.plot(x_plot, y_plot_validation, label='validation')
    plt.plot(x_plot, y_plot_train, label='training')

    plt.xlabel('steps')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()

    show_images(sess)
    sess.close()


def show_images(sess):
    test_im, _ = data.get_test_image_list_and_label_list()

    for i in range(2):
        # original images
        image = test_im[i]
        im = np.array([[p[0] for p in l] for l in image])
        plot_image(im)
        # segmented images
        test_out = sess.run(output_conv, feed_dict={x_image: test_im})
        test_prediction = np.argmax(test_out, axis=3)
        plot_image(test_prediction[i])


if __name__ == '__main__':
    data = Data()


    x_image = tf.placeholder(tf.float32, [None, 300, 300, 1])
    y_label = tf.placeholder(tf.int32, [None, 116, 116])

    # 2 convolutions and pooling
    layer1 = conv_layer(x_image, num_filters=32)
    layer2 = conv_layer(layer1, num_filters=32)
    pooling_layer1 = max_pooling_layer(layer2)

    # 2 convolutions and pooling
    layer3 = conv_layer(pooling_layer1, num_filters=64)
    layer4 = conv_layer(layer3, num_filters=64)
    pooling_layer2 = max_pooling_layer(layer4)

    # 2 convolutions and pooling
    layer5 = conv_layer(pooling_layer2, num_filters=128)
    layer6 = conv_layer(layer5, num_filters=128)
    pooling_layer3 = max_pooling_layer(layer6)

    # 2 convolutions and pooling
    layer7 = conv_layer(pooling_layer3, num_filters=256)
    layer8 = conv_layer(layer7, num_filters=256)
    pooling_layer4 = max_pooling_layer(layer8)

    # 2 convolutions and up conv
    layer9 = conv_layer(pooling_layer4, num_filters=512)
    layer10 = conv_layer(layer9, num_filters=512)
    up_conv_layer1 = conv_transpose_layer(layer10)

    # merge conv layer 8 and up conv layer 1, shape = 22,22,512
    merged_1 = crop_and_merge_layers(layer_to_be_cropped=layer8, second_layer=up_conv_layer1)

    # 2 convolutions and up conv
    layer11 = conv_layer(merged_1, num_filters=512)
    layer12 = conv_layer(layer11, num_filters=256)
    up_conv_layer2 = conv_transpose_layer(layer12)

    # merge conv layer 6 and up conv layer 2, shape = 36,36,256
    merged_2 = crop_and_merge_layers(layer_to_be_cropped=layer6, second_layer=up_conv_layer2)

    # 2 convolutions and up conv
    layer13 = conv_layer(merged_2, num_filters=256)
    layer14 = conv_layer(layer13, num_filters=128)
    up_conv_layer3 = conv_transpose_layer(layer14)

    # merge conv layer 4 and up conv layer 3, shape = 64,64,128
    merged_3 = crop_and_merge_layers(layer_to_be_cropped=layer4, second_layer=up_conv_layer3)

    # 2 convolutions and up conv
    layer15 = conv_layer(merged_3, num_filters=128)
    layer16 = conv_layer(layer15, num_filters=64)
    up_conv_layer4 = conv_transpose_layer(layer16)

    # merge conv layer 2 and up conv layer 4, shape = 120,120,64
    merged_4 = crop_and_merge_layers(layer_to_be_cropped=layer2, second_layer=up_conv_layer4)

    # 2 convolutions and output
    layer17 = conv_layer(merged_4, num_filters=64)
    layer18 = conv_layer(layer17, num_filters=32)
    output_conv = conv_layer(layer18, num_filters=2, filter_size=1)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label, logits=output_conv)
    total_loss = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(0.0001, 0.95, 0.99).minimize(total_loss)

    train_and_plot()