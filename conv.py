import argparse, cPickle, os, random, sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Trains or evaluates a convolutional neural network.')
group = parser.add_mutually_exclusive_group()
group.add_argument('-e', '--evaluate', action='store_true')
group.add_argument('-t', '--train', action='store_true')
args = parser.parse_args()

BATCH_SIZE = 100
SAVE_FREQUENCY = 50
SAVE_NAME = '/Users/SHETELEKIDES/Desktop/conv/saved_model.ckpt'
CIFAR_DIRECTORY = '/Users/SHETELEKIDES/Desktop/conv/cifar-10-batches-py'

class CIFARData:
  def __init__(self, directory):
    self.directory = directory

  def unpickle(self, file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

  def get_image(self, data):
    red = data[:1024]
    green = data[1024:2048]
    blue = data[2048:]

    all_pixels_row_wise = map(list, zip(red, green, blue))
    return [all_pixels_row_wise[i * 32:(i + 1) * 32] for i in xrange(32)]

  def read(self):
    pickle_1 = self.unpickle(os.path.join(self.directory, 'data_batch_1'))
    pickle_2 = self.unpickle(os.path.join(self.directory, 'data_batch_2'))
    pickle_3 = self.unpickle(os.path.join(self.directory, 'data_batch_3'))
    pickle_4 = self.unpickle(os.path.join(self.directory, 'data_batch_4'))
    pickle_5 = self.unpickle(os.path.join(self.directory, 'data_batch_5'))

    pickle_test = self.unpickle(os.path.join(self.directory, 'test_batch'))

    self.train_data = np.concatenate([pickle_1['data'], pickle_2['data'], pickle_3['data'], pickle_4['data'], pickle_5['data']])
    self.train_labels = np.concatenate([pickle_1['labels'], pickle_2['labels'], pickle_3['labels'], pickle_4['labels'], pickle_5['labels']])

    self.test_data = pickle_test['data']
    self.test_labels = pickle_test['labels']

  def random_train_batch(self, size):
    return self.get_random_batch(self.train_data, self.train_labels, size)

  def random_test_batch(self, size):
    return self.get_random_batch(self.test_data, self.test_labels, size)

  def get_train_size(self):
    return len(self.train_data)

  def get_test_size(self):
    return len(self.test_data)

  def get_label_name(self, index):
    LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return LABEL_NAMES[index]

  def get_random_batch(self, data, labels, size):
    image_batch = []
    label_batch = []

    for _ in xrange(size):
      index = random.randint(0, len(data) - 1)
      pixel_data = self.get_image(data[index])
      label = labels[index]
      
      image_batch.append(pixel_data)
      label_batch.append(label)

    return (image_batch, label_batch)

filter_conv_1_size = 3
filter_1_count = 32

filter_conv_2_size = 3
filter_2_count = 64

filter_conv_3_size = 3
filter_3_count = 64

fc_1_count = 512

# filters = [filter_height, filter_width, in_channels, out_channels]
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

pixels_as_floats = tf.divide(x, 255.0)
cleaned_x = tf.subtract(pixels_as_floats, 0.5)

filters_conv_1 = tf.Variable(tf.truncated_normal(shape=[filter_conv_1_size, filter_conv_1_size, 3, filter_1_count], stddev=0.2))
bias_conv_1 = tf.Variable(tf.truncated_normal(shape=[filter_1_count], stddev=0.2))
conv_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(input=cleaned_x, filter=filters_conv_1, strides=[1, 1, 1, 1], padding='SAME') + bias_conv_1), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

filters_conv_2 = tf.Variable(tf.truncated_normal(shape=[filter_conv_2_size, filter_conv_2_size, filter_1_count, filter_2_count], stddev=0.2))
bias_conv_2 = tf.Variable(tf.truncated_normal(shape=[filter_2_count], stddev=0.2))
conv_2 = tf.nn.relu(tf.nn.conv2d(input=conv_1, filter=filters_conv_2, strides=[1, 1, 1, 1], padding='SAME') + bias_conv_2)

filters_conv_3 = tf.Variable(tf.truncated_normal(shape=[filter_conv_3_size, filter_conv_3_size, filter_2_count, filter_3_count], stddev=0.2))
bias_conv_3 = tf.Variable(tf.truncated_normal(shape=[filter_3_count], stddev=0.2))
conv_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(input=conv_2, filter=filters_conv_3, strides=[1, 1, 1, 1], padding='SAME') + bias_conv_3), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

keep_prob = tf.placeholder(tf.float32)
fully_connected_w1 = tf.Variable(tf.truncated_normal(shape=[filter_3_count * 8 * 8, fc_1_count], stddev=0.2))
bias_fc_1 = tf.Variable(tf.truncated_normal(shape=[fc_1_count], stddev=0.2))
fully_connected_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.reshape(conv_3, shape=[tf.shape(x)[0], filter_3_count * 8 * 8]), fully_connected_w1) + bias_fc_1), keep_prob)

fully_connected_w2 = tf.Variable(tf.truncated_normal(shape=[fc_1_count, 10], stddev=0.2))
bias_fc_2 = tf.Variable(tf.truncated_normal(shape=[10], stddev=0.2))
fully_connected_2 = tf.matmul(fully_connected_1, fully_connected_w2) + bias_fc_2

actual_prediction = tf.arg_max(fully_connected_2, 1)

correct_labels = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(correct_labels, 10), logits=fully_connected_2))
step = tf.train.AdamOptimizer().minimize(loss)

percent_correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fully_connected_2, 1), correct_labels), tf.float32))

cifar_data = CIFARData(CIFAR_DIRECTORY)
cifar_data.read()

saver = tf.train.Saver()
with tf.Session() as sess:
  try:
    saver.restore(sess, SAVE_NAME)
    print 'Restored successfully.'
  except:
    sess.run(tf.global_variables_initializer())

  if args.train:
    print 'Training...'

    step_count = 0
    while True:
      train_images, train_labels = cifar_data.random_train_batch(BATCH_SIZE) #get_random_batch(data, labels, BATCH_SIZE)
      print sess.run([loss, percent_correct, step], feed_dict={x: train_images, correct_labels: train_labels, keep_prob: 0.5})
      step_count += 1

      if step_count % SAVE_FREQUENCY == 0:
        train_images, train_labels = cifar_data.random_test_batch(BATCH_SIZE)
        print '-- TEST SET % --'
        print sess.run(percent_correct, feed_dict={x: train_images, correct_labels: train_labels, keep_prob: 1.0})

        save_path = saver.save(sess, SAVE_NAME)
        print 'Saved to {0}'.format(save_path)
        print '----------------'
  elif args.evaluate:
    interactive = False

    num_of_tests = cifar_data.get_test_size()
    print 'Running tests.'

    test_images, test_labels = cifar_data.random_test_batch(num_of_tests) #get_random_batch(data_test, labels_test, num_of_tests)
    correct = 0.0
    for index in xrange(num_of_tests):
      display_image, true_label = (test_images[index], test_labels[index])
      raw_pixels = sess.run(pixels_as_floats, feed_dict={x: [display_image]})[0]
      network_prediction = sess.run(actual_prediction, feed_dict={x: [display_image], keep_prob: 1.0})[0]

      if network_prediction == true_label:
        correct += 1
      
      if interactive:
        plt.imshow(raw_pixels)
        plt.title(cifar_data.get_label_name(network_prediction) + ':' + cifar_data.get_label_name(true_label))
        plt.show()
    print 'Percent correct: ' + str(correct / num_of_tests)
