import tensorflow as tf

class Network:
  def __init__(self, sess):
    self.create_network()
    self.sess = sess
  
  def train(self, train_images, train_labels, keep_prob):
    ops_to_run = [self.loss, self.percent_correct, self.step]
    return tuple(self.sess.run(ops_to_run, feed_dict={self.x: train_images, self.correct_labels: train_labels, self.keep_prob: keep_prob}))

  def run(self, input):
    ops_to_run = [self.pixels_as_floats, self.actual_prediction]
    return tuple(self.sess.run(ops_to_run, feed_dict={self.x: input, self.keep_prob: 1.0}))

  def evaluate(self, input, expected_output):
    return self.sess.run(network.percent_correct, feed_dict={self.x: input, self.correct_labels: expected_output, self.keep_prob: 1.0})

  def create_network(self):
    filter_conv_1_size = 3
    filter_1_count = 32

    filter_conv_2_size = 3
    filter_2_count = 64

    filter_conv_3_size = 3
    filter_3_count = 64

    fc_1_count = 512

    self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

    self.pixels_as_floats = tf.divide(self.x, 255.0)
    self.cleaned_x = tf.subtract(self.pixels_as_floats, 0.5)

    self.filters_conv_1 = tf.Variable(tf.truncated_normal(shape=[filter_conv_1_size, filter_conv_1_size, 3, filter_1_count], stddev=0.2))
    self.bias_conv_1 = tf.Variable(tf.truncated_normal(shape=[filter_1_count], stddev=0.2))
    self.conv_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(input=self.cleaned_x, filter=self.filters_conv_1, strides=[1, 1, 1, 1], padding='SAME') + self.bias_conv_1), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    self.filters_conv_2 = tf.Variable(tf.truncated_normal(shape=[filter_conv_2_size, filter_conv_2_size, filter_1_count, filter_2_count], stddev=0.2))
    self.bias_conv_2 = tf.Variable(tf.truncated_normal(shape=[filter_2_count], stddev=0.2))
    self.conv_2 = tf.nn.relu(tf.nn.conv2d(input=self.conv_1, filter=self.filters_conv_2, strides=[1, 1, 1, 1], padding='SAME') + self.bias_conv_2)

    self.filters_conv_3 = tf.Variable(tf.truncated_normal(shape=[filter_conv_3_size, filter_conv_3_size, filter_2_count, filter_3_count], stddev=0.2))
    self.bias_conv_3 = tf.Variable(tf.truncated_normal(shape=[filter_3_count], stddev=0.2))
    self.conv_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(input=self.conv_2, filter=self.filters_conv_3, strides=[1, 1, 1, 1], padding='SAME') + self.bias_conv_3), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    self.keep_prob = tf.placeholder(tf.float32)
    self.fully_connected_w1 = tf.Variable(tf.truncated_normal(shape=[filter_3_count * 8 * 8, fc_1_count], stddev=0.2))
    self.bias_fc_1 = tf.Variable(tf.truncated_normal(shape=[fc_1_count], stddev=0.2))
    self.fully_connected_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.reshape(self.conv_3, shape=[tf.shape(self.x)[0], filter_3_count * 8 * 8]), self.fully_connected_w1) + self.bias_fc_1), self.keep_prob)

    self.fully_connected_w2 = tf.Variable(tf.truncated_normal(shape=[fc_1_count, 10], stddev=0.2))
    self.bias_fc_2 = tf.Variable(tf.truncated_normal(shape=[10], stddev=0.2))
    self.fully_connected_2 = tf.matmul(self.fully_connected_1, self.fully_connected_w2) + self.bias_fc_2

    self.actual_prediction = tf.reshape(tf.arg_max(self.fully_connected_2, 1), shape=[1])

    self.correct_labels = tf.placeholder(tf.int64)
    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.correct_labels, 10), logits=self.fully_connected_2))
    self.step = tf.train.AdamOptimizer().minimize(self.loss)

    self.percent_correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.fully_connected_2, 1), self.correct_labels), tf.float32))