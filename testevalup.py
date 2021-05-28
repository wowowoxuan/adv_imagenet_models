import sys
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from tensorflow.contrib.slim.nets import inception
import os
import json
#python testeval.py --checkpoint_path=/data1/weiheng3_7_2021/AEG/ckpt/ens3_adv_inception_v3/ens3_adv_inception_v3.ckpt --image_dir=/data1/weiheng3_7_2021/AEG/dataset/imagenet/vggcel/ --model_name=inception_v3

pick_list = ['vgg16_pgd','vgg16_mifgsm','vgg16_dim','vgg16_tidr','vgg16_afv','resnet152_pgd','resnet152_mifgsm',
 'resnet152_dim','resnet152_tidr','resnet152_afv','inception_v3_pgd','inception_v3_mifgsm','inception_v3_dim',
 'inception_v3_tidr','inception_v3_afv','mobilenet_v2_pgd','mobilenet_v2_mifgsm','mobilenet_v2_dim','mobilenet_v2_tidr',
 'mobilenet_v2_afv']
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

with open('gt.json', 'r') as jsonfile:
    labels = json.load(jsonfile)
FLAGS = tf.app.flags.FLAGS

gt_dict = {}
with open('./valid_gt.csv') as f:
       for line in f:
            fields = line.split(',')

            (key,  value) = fields
            # print(key)
            # print(value)
            gt_dict[key] = value





slim = tf.contrib.slim
height = 299
width = 299
channels = 3
X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
gt = np.loadtxt('ILSVRC2012_validation_ground_truth.txt', dtype = int)
if FLAGS.model_name == 'inception_v3':
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=False)

elif FLAGS.model_name == 'inception_resnet_v2':
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(X, num_classes=1001,is_training=False)
else:
    raise ValueError('Invalid model name: %s' % (FLAGS.model_name))
saver = tf.train.Saver()
sess = tf.Session()

saver.restore(sess, FLAGS.checkpoint_path)
mean = np.array([0.485, 0.456, 0.406])
mean = mean.reshape(1,1,3)
std = np.array([0.229, 0.224, 0.225])
std = std.reshape(1,1,3)
print(mean)
print(std)
for item in pick_list:
    INPUT_DIR = os.path.join(FLAGS.image_dir,item)
    print(INPUT_DIR)
    a=open('incres_v2_ens.txt', 'a')
    correct_count = 0
    nullcount = 0
    all_count_c = 0
    for image_name in tqdm(os.listdir(INPUT_DIR)):    
        all_count_c += 1
        image_path = os.path.join(INPUT_DIR, image_name)
        image = Image.open(image_path)
        image = image.resize((299,299))
        image = image.convert('RGB')
        image = np.asarray(image, dtype=np.float32)/255
        image = (image-mean)/std
        catlogits = sess.run(logits, feed_dict={X:image.reshape(1,299,299,3)})
        # image_index = image_name.replace('ILSVRC2012_val_','')
        image_index = image_name.replace('.JPEG','')
        image_index = image_index.replace('.png','')
        image_index = image_index.replace('.jpg','')
        # image_index = int(image_index) - 1
        #predictions = tf.argmax(catlogits)
        # print(np.max(np.sort(catlogits[0,:])))
        # print(np.argmax(np.argsort(catlogits[0,:])))
        predict = np.argmax(catlogits[0,:])
        # predict = np.argsort(catlogits[0,:])[-5:]
        # if len(labels[image_index]) == 0:
        #             nullcount += 1
        # for item in labels[image_index]:
        #     for pred in predict:
        #         if pred == item:
        #             correct_count += 1
        #             break
        # for item in predict:
        #     if gt[image_index] == item:
        #         correct_count += 1
        if int(gt_dict[image_index]) == predict:
            correct_count += 1
    print(nullcount)
    print(correct_count/all_count_c)
    a.write('\n')
    a.write('========================================================')
    a.write('\n')
    a.write(FLAGS.model_name)
    a.write('\n')
    a.write(FLAGS.checkpoint_path)
    a.write('\n')
    a.write(INPUT_DIR)
    a.write('\n')
    a.write(str(correct_count/all_count_c))
    a.close()
# image = Image.open('/data1/weiheng3_7_2021/AEG/dataset/debugset/ILSVRC2012_test_00000083.jpg')
# image = image.resize((299,299))
# image = image.convert('RGB')
# image = np.asarray(image, dtype=np.float32)/255
# print(image.shape)
# #testimg = testimg.reshape(299,299,3)
# catlogits = sess.run(logits, feed_dict={X:image.reshape(1,299,299,3)})

# print(catlogits)
# print(np.sort(catlogits[0,:])[-5:])
# print(np.argsort(catlogits[0,:])[-5:])