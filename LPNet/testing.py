#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a implementation of testing code of this paper:
# X. Fu, B. Liang, Y. Huang, X. Ding and J. Paisley. “Lightweight Pyramid Networks for Image Deraining”, IEEE Transactions on Neural Networks and Learning Systems, 2019.
# author: Xueyang Fu (xyfu@ustc.edu.cn)

import os
from skimage import io
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt

import model

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.compat.v1.reset_default_graph()

model_path = './model/'
pre_trained_model_path = './model/model-epoch-10800'  # 'heavy_model' for Rain100H,  'light_model' for Rain100L


img_path = './test_img/img/' # the path of testing images
results_path = './test_img/results/' # the path of de-rained images


def _parse_function(filename):   
  image_string = tf.io.read_file(filename)  
  image_decoded = tf.image.decode_png(image_string, channels=3)  
  rainy = tf.cast(image_decoded, tf.float32)/255.0   
  return rainy 


if __name__ == '__main__':
   imgName = os.listdir(img_path)
   num_img = len(imgName)
   
   whole_path = []
   for i in range(num_img):
      whole_path.append(img_path + imgName[i])
      
    
   filename_tensor = tf.convert_to_tensor(whole_path, dtype=tf.string)     
   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
   dataset = dataset.map(_parse_function)    
   dataset = dataset.prefetch(buffer_size=10)
   dataset = dataset.batch(batch_size=1).repeat()  
   iterator = dataset.make_one_shot_iterator()
   
   rain = iterator.get_next()     
 
   pyramid = model.inference(rain)
   output = tf.clip_by_value(pyramid[-1], 0., 1.)
   output = output[0,:,:,:]

   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True   
   saver = tf.train.Saver()
   
   with tf.Session(config=config) as sess:
      with tf.device('/gpu:0'): 
          if tf.train.get_checkpoint_state(model_path):  
              ckpt = tf.train.latest_checkpoint(model_path)  # try your own model 
              saver.restore(sess, ckpt)
              print ("Loading model")
          else:
             saver.restore(sess, pre_trained_model_path) # try a pre-trained model 
             print ("Loading pre-trained model")

          for i in range(num_img):     
             derained, ori = sess.run([output, rain])              
             derained = np.uint8(derained* 255.)
             index = imgName[i].rfind('.')
             name = imgName[i][:index]
             io.imsave(results_path + name +'.png', derained)         
             print('%d / %d images processed' % (i+1,num_img))
              
      print('All done')
   sess.close()   
   
   plt.subplot(1,2,1)     
   plt.imshow(ori[0,:,:,:])          
   plt.title('rainy')
   plt.subplot(1,2,2)    
   plt.imshow(derained)
   plt.title('derained')
   plt.show()      