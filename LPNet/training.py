#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a implementation of training code of this paper:
# X. Fu, B. Liang, Y. Huang, X. Ding and J. Paisley. “Lightweight Pyramid Networks for Image Deraining”, IEEE Transactions on Neural Networks and Learning Systems, 2019.
# author: Xueyang Fu (xyfu@ustc.edu.cn)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import re
import time
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()
tf.debugging.set_log_device_placement(True)

from model import inference, GaussianPyramid 




num_pyramids = 5       # number of pyramid levels
learning_rate = 1e-4   # learning rate
iterations = int(1.082e5)  # iterations
batch_size = 10        # batch size
num_channels = 3       # number of input's channels 
patch_size = 80        # patch size 
save_model_path = './model/'  # path of saved model
model_name = 'model-epoch'    # name of saved model


input_path = './TrainData/input/' # rainy images
gt_path = './TrainData/label/'    # ground truth
 

# randomly select image patches
def _parse_function(input_path, gt_path, patch_size = patch_size):   
    image_string = tf.io.read_file(input_path)  
    image_decoded = tf.image.decode_png(image_string, channels=3)  
    rainy = tf.cast(image_decoded, tf.float32)/255.0
          
    image_string = tf.io.read_file(gt_path)  
    image_decoded = tf.image.decode_png(image_string, channels=3)  
    label = tf.cast(image_decoded, tf.float32)/255.0
          
    t = time.time()
    Data = tf.random_crop(rainy, [patch_size, patch_size ,3],seed = t)   # randomly select patch
    Label = tf.random_crop(label, [patch_size, patch_size ,3],seed = t)       
    return Data, Label 



if __name__ == '__main__':    
    tf.reset_default_graph()

    input_files = os.listdir(input_path)
    for i in range(len(input_files)):
        input_files[i] = input_path + input_files[i]
        
    label_files = os.listdir(gt_path)       
    for i in range(len(label_files)):
        label_files[i] = gt_path + label_files[i] 
    
    input_files = tf.convert_to_tensor(input_files, dtype=tf.string)  
    label_files = tf.convert_to_tensor(label_files, dtype=tf.string)  
     
    dataset = tf.data.Dataset.from_tensor_slices((input_files, label_files))
    dataset = dataset.map(_parse_function)    
    dataset = dataset.prefetch(buffer_size=batch_size * 10)
    dataset = dataset.batch(batch_size).repeat()  
    iterator = dataset.make_one_shot_iterator()   
    inputs, labels = iterator.get_next()  


    k = np.float32([.0625, .25, .375, .25, .0625])  # Gaussian kernel for image pyramid
    k = np.outer(k, k) 
    kernel = k[:,:,None,None]/k.sum()*np.eye(3, dtype = np.float32)
    labels_GaussianPyramid = GaussianPyramid( labels, kernel, (num_pyramids-1) ) # Gaussian pyramid for ground truth

    outout_pyramid = inference(inputs) # LPNet

    loss1 = tf.reduce_mean(tf.abs(outout_pyramid[0] - labels_GaussianPyramid[0]))    # L1 loss
    loss2 = tf.reduce_mean(tf.abs(outout_pyramid[1] - labels_GaussianPyramid[1]))    # L1 loss
    loss3 = tf.reduce_mean(tf.abs(outout_pyramid[2] - labels_GaussianPyramid[2]))    # L1 loss
  
    loss41 = tf.reduce_mean(tf.abs(outout_pyramid[3] - labels_GaussianPyramid[3]))   # L1 loss
    loss42 = tf.reduce_mean((1. - tf.image.ssim(outout_pyramid[3],labels_GaussianPyramid[3], max_val=1.0))/2.) # SSIM loss

    loss51 = tf.reduce_mean(tf.abs(outout_pyramid[4] - labels))  # L1 loss
    loss52 = tf.reduce_mean((1. - tf.image.ssim(outout_pyramid[4],labels, max_val=1.0))/2.) # SSIM loss
    
    loss = loss1 + loss2 + loss3 + loss41 + loss42 + loss51 + loss52
    g_optim =  tf.train.AdamOptimizer(learning_rate).minimize(loss) # Optimization method: Adam
    
    all_vars = tf.trainable_variables()  
    saver = tf.train.Saver(var_list=all_vars, max_to_keep = 5)  


    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
      
       sess.run(tf.group(tf.global_variables_initializer(), 
                         tf.local_variables_initializer()))
       tf.get_default_graph().finalize()	
              
       if tf.train.get_checkpoint_state(save_model_path):   # load previous trained model 
          ckpt = tf.train.latest_checkpoint(save_model_path)
          saver.restore(sess, ckpt)  
          ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
          start_point = int(ckpt_num[0]) + 1     
          print("loaded successfully")
       else:  # re-training when no models found
          print("re-training")
          start_point = 0  
          
       check_input, check_label =  sess.run([inputs,labels])
       print("check patch pair:")  
       plt.subplot(1,3,1)     
       plt.imshow(check_input[0,:,:,:])
       plt.title('input')         
       plt.subplot(1,3,2)    
       plt.imshow(check_label[0,:,:,:])
       plt.title('ground truth')      
       plt.show()    
     
       start = time.time()    
       
       for j in range(start_point,iterations):                   
                          
           _, Training_Loss = sess.run([g_optim,loss])  # training

           if np.mod(j+1,100) == 0 and j != 0:    
              end = time.time() 
              print ('%d / %d iteraions, Training Loss  = %.4f, running time = %.1f s' 
                     % (j+1, iterations, Training_Loss, (end-start)))          
              save_path_full = os.path.join(save_model_path, model_name) 
              saver.save(sess, save_path_full, global_step = j+1) # save model every 100 iterations
              start = time.time()  
              
       print ('training finished') 
    sess.close()