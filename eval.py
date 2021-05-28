from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import imagenet
import inception_resnet_v2
from tqdm import tqdm 
import os
import tensorflow as tf
from scipy.misc import imread
import numpy as np
from tensorflow.contrib.slim.nets import inception


slim = tf.contrib.slim


tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'image_dir', '/tmp/image/',
    'The directory where the test images are '
    'image file.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3',
    'Name of the model to use, either "inception_v3" or "inception_resnet_v2"')


FLAGS = tf.app.flags.FLAGS


IMAGE_SIZE = 299
NUM_CLASSES = 1001
INPUT_DIR = FLAGS.image_dir
central_fraction=0.875

def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.
  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def create_model(x, reuse=None):
  """Create model graph.

  Args:
    x: input images
    reuse: reuse parameter which will be passed to underlying variable scopes.
      Should be None first call and True every subsequent call.

  Returns:
    (logits, end_points) - tuple of model logits and enpoints

  Raises:
    ValueError: if model type specified by --model_name flag is invalid.
  """
#   if FLAGS.model_name == 'inception_v3':
#     with slim.arg_scope(inception.inception_v3_arg_scope()):
#       return inception.inception_v3(
#           x, num_classes=NUM_CLASSES, is_training=False, reuse=reuse)
#   elif FLAGS.model_name == 'inception_resnet_v2':
#     with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
#       return inception_resnet_v2.inception_resnet_v2(
#           x, num_classes=NUM_CLASSES, is_training=False, reuse=reuse)
#   else:
#     raise ValueError('Invalid model name: %s' % (FLAGS.model_name))
  if FLAGS.model_name == 'inception_v3':
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      return inception.inception_v3(
          x, num_classes=NUM_CLASSES, is_training=False, reuse=reuse)
  elif FLAGS.model_name == 'inception_resnet_v2':
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      return inception_resnet_v2.inception_resnet_v2(
          x, num_classes=NUM_CLASSES, is_training=False, reuse=reuse)
  else:
    raise ValueError('Invalid model name: %s' % (FLAGS.model_name))

def main(_):
  
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = tf.train.get_or_create_global_step()
    
    print('input dir is')
    print(INPUT_DIR)
    ###################
    # Prepare dataset #
    ###################
    create_model(tf.placeholder(tf.float32, shape=[1,299,299,3]))
    for image_name in tqdm(os.listdir(INPUT_DIR)):    
        image_path = os.path.join(INPUT_DIR, image_name)
        image = imread(image_path, mode='RGB').astype(np.float) / 255.0  
        image = tf.convert_to_tensor(image, dtype=None, dtype_hint=None, name=None) 
        # print(image_tensor.shape)
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=central_fraction)

        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [299, 299],
                                        align_corners=False)
        image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        image = tf.reshape(image, shape=[1,299,299,3])
        print(image.shape)
        create_model(tf.placeholder(tf.float32, shape=[1,299,299,3]))
        logits, _ = create_model(image, reuse=True)
        predictions = tf.argmax(logits, 1)
        print(predictions)
 
    ########################################
    # Define the model and input exampeles #
    ########################################
    # create_model(tf.placeholder(tf.float32, shape=images.shape))
    # input_images = get_input_images(dataset_images)
    # logits, _ = create_model(input_images, reuse=True)


    ######################
    # Define the metrics #
    ######################
    # predictions = tf.argmax(logits, 1)
    # print(predictions)

    # print('Top1 Accuracy: ', top1_accuracy)
    # print('Top5 Accuracy: ', top5_accuracy)


if __name__ == '__main__':
  tf.app.run()