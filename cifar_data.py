import cPickle, os, random, sys
import numpy as np

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

    indexes = random.sample(xrange(0, len(data)), size)
    for index in indexes:
      pixel_data = self.get_image(data[index])
      label = labels[index]
      
      image_batch.append(pixel_data)
      label_batch.append(label)

    return (image_batch, label_batch)