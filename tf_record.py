import math
import json
import os
import argparse
import sys

import tensorflow as tf

from models.slim.datasets import dataset_utils

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_json):
  """Returns the images and classes from a given json

  Args:
    dataset_json::str : the location of the json for this dataset
  Returns:
    images::list(dict) : list of all json dicts representing each image
    classes::list(str) : list of all unique class names
  """
  with open(dataset_json) as f:
    d = json.load(f)

  return d['images'], d['tags']

def _get_dataset_filename(dataset_dir, split_name, shard_id, dataset_name):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      dataset_name, split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, image_json, class_names_to_ids, output_dir, dataset_name):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    output_dir: The directory where the converted TFrecord will be written.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(image_json) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:
      
      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            output_dir, split_name, shard_id, dataset_name)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(image_json))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(image_json), shard_id))
            sys.stdout.flush()

            # Read the filename:
            try:
                image_data = tf.gfile.FastGFile(image_json[i]['location'], 'rb').read()
                height, width = image_reader.read_image_dims(sess, image_data)

                class_name = image_json[i]['annotation']['annotations'][0]['tag']
                class_id = class_names_to_ids[class_name]

                example = dataset_utils.image_to_tfexample(
                    image_data, b'jpg', height, width, class_id)
                tfrecord_writer.write(example.SerializeToString())
            except:
                sys.stdout.write('{}'.format(image_json[i]['location']))

  sys.stdout.write('\n')
  sys.stdout.flush()

def run(dataset_dir, output_dir, name):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    output_dir: Where to store new files.
  """
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  training_filenames = filter(lambda im : im['stage'] == 'train', photo_filenames)
  validation_filenames = filter(lambda im : im['stage'] == 'val', photo_filenames)

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, class_names_to_ids, output_dir, name)
  _convert_dataset('validation', validation_filenames, class_names_to_ids, output_dir, name)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, output_dir)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data','--dataset','--dataset_dir', type=str, dest='data')
  parser.add_argument('--out','--output','--output_dir', type=str, dest='out')
  parser.add_argument('--name',type=str)
  parser.add_argument('--num_shards',type=int, dest='shards', default=5)
  args = parser.parse_args()

  _NUM_SHARDS = args.shards

  run(args.data, args.out, args.name)
