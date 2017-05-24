import argparse

import tensorflow as tf
import matplotlib.pyplot as plt

from network import Network
from cifar_data import CIFARData

def main():
  parser = argparse.ArgumentParser(description='Trains or evaluates a convolutional neural network.')
  group = parser.add_mutually_exclusive_group()
  group.required = True
  group.add_argument('-e', '--evaluate', action='store_true')
  group.add_argument('-t', '--train', action='store_true')
  group.add_argument('-i', '--interactive', type=int)
  args = parser.parse_args()

  BATCH_SIZE = 100
  SAVE_FREQUENCY = 50
  SAVE_NAME = '/Users/SHETELEKIDES/Desktop/CIFAR-CNN/saved_model.ckpt'
  CIFAR_DIRECTORY = '/Users/SHETELEKIDES/Desktop/CIFAR-CNN/cifar-10-batches-py'

  with tf.Session() as sess:
    cifar_data = CIFARData(CIFAR_DIRECTORY)
    cifar_data.read()

    network = Network(sess)

    saver = tf.train.Saver()

    try:
      saver.restore(sess, SAVE_NAME)
      print 'Restored successfully.'
    except:
      sess.run(tf.global_variables_initializer())

    if args.train:
      print 'Training...'

      step_count = 0
      while True:
        train_images, train_labels = cifar_data.random_train_batch(BATCH_SIZE)
        print network.train(train_images, train_labels, 0.5)
        step_count += 1

        if step_count % SAVE_FREQUENCY == 0:
          train_images, train_labels = cifar_data.random_test_batch(BATCH_SIZE)
          print '-- TEST SET % --'
          print network.evaluate(train_images, train_labels)

          save_path = saver.save(sess, SAVE_NAME)
          print 'Saved to {0}'.format(save_path)
          print '----------------'
    elif args.evaluate or args.interactive is not None:
      if args.interactive is not None:
        interactive = True
        num_of_tests = args.interactive
        print 'Entering interactive mode.'
      else:
        interactive = False
        num_of_tests = cifar_data.get_test_size()
        print 'Running tests.'

      test_images, test_labels = cifar_data.random_test_batch(num_of_tests)
      correct = 0.0
      for index in xrange(num_of_tests):
        display_image, true_label = (test_images[index], test_labels[index])
        all_raw_pixels, network_predictions = network.run([display_image])

        raw_pixels = all_raw_pixels[0]
        network_prediction = network_predictions[0]

        if network_prediction == true_label:
          correct += 1
        
        if interactive:
          plt.imshow(raw_pixels)
          plt.title(cifar_data.get_label_name(network_prediction) + ':' + cifar_data.get_label_name(true_label))
          plt.show()
      print 'Percent correct: ' + str(correct / num_of_tests)

if __name__ == "__main__":
    main()
