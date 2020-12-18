#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-7-19 上午10:28
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/attentive-gan-derainnet
# @File    : test_model.py
# @IDE: PyCharm
"""
test model
"""
import os
import os.path as ops
import argparse
from glob import glob

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

import sys
sys.path.insert(0,ops.abspath(os.getcwd()))
print(sys.path)
from attentive_gan_model import derain_drop_net
from config import global_config

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The input image path')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--label_path', type=str, default=None, help='The label image path')
    parser.add_argument('--batch_experiment', action='store_true', help='Run the model for all the images')
    parser.add_argument('--folder_path', type=str, help='The input image folder path')

    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def visualize_attention_map(attention_map):
    """
    The attention map is a matrix ranging from 0 to 1, where the greater the value,
    the greater attention it suggests
    :param attention_map:
    :return:
    """
    attention_map_color = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1], 3],
        dtype=np.uint8
    )

    red_color_map = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1]],
        dtype=np.uint8) + 255
    red_color_map = red_color_map * attention_map
    red_color_map = np.array(red_color_map, dtype=np.uint8)

    attention_map_color[:, :, 2] = red_color_map

    return attention_map_color


def test_model(image_path, weights_path, label_path=None, batch_experiment=False):
    """

    :param image_path:
    :param weights_path:
    :param label_path:
    :return:
    """
    assert ops.exists(image_path)

    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TEST.BATCH_SIZE, CFG.TEST.IMG_HEIGHT, CFG.TEST.IMG_WIDTH, 3],
                                  name='input_tensor'
                                  )

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    image_vis = image
    image = np.divide(np.array(image, np.float32), 127.5) - 1.0

    label_image_vis = None
    if label_path is not None:
        label_image = cv2.imread(label_path, cv2.IMREAD_COLOR)
        label_image_vis = cv2.resize(
            label_image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )

    phase = tf.constant('test', tf.string)

    net = derain_drop_net.DeRainNet(phase=phase)
    output, attention_maps = net.inference(input_tensor=input_tensor, name='derain_net')

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()

    save_folder = image_path[:-11]+'Attentive_GAN_res/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        output_image, atte_maps = sess.run(
            [output, attention_maps],
            feed_dict={input_tensor: np.expand_dims(image, 0)})

        output_image = output_image[0]
        for i in range(output_image.shape[2]):
            output_image[:, :, i] = minmax_scale(output_image[:, :, i])

        output_image = np.array(output_image, np.uint8)

        if label_path is not None:
            label_image_vis_gray = cv2.cvtColor(label_image_vis, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
            output_image_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
            psnr = compare_psnr(label_image_vis_gray, output_image_gray)
            ssim = compare_ssim(label_image_vis_gray, output_image_gray)

            print('SSIM: {:.5f}'.format(ssim))
            print('PSNR: {:.5f}'.format(psnr))

        # 保存并可视化结果

        if not batch_experiment:
            cv2.imwrite('src_img.png', image_vis)
            cv2.imwrite('derain_ret.png', output_image)

            plt.figure('src_image')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.figure('derain_ret')
            plt.imshow(output_image[:, :, (2, 1, 0)])
            plt.figure('atte_map_1')
            plt.imshow(atte_maps[0][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_1.png')
            plt.figure('atte_map_2')
            plt.imshow(atte_maps[1][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_2.png')
            plt.figure('atte_map_3')
            plt.imshow(atte_maps[2][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_3.png')
            plt.figure('atte_map_4')
            plt.imshow(atte_maps[3][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_4.png')
            plt.show()
        else:
            
            cv2.imwrite(save_folder+image_path[-11:-4]+'_src.png', image_vis)
            cv2.imwrite(save_folder+image_path[-11:-4]+'_res.png', output_image)

    return

def batch_test_model(folder_path, weights_path):

    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TEST.BATCH_SIZE, CFG.TEST.IMG_HEIGHT, CFG.TEST.IMG_WIDTH, 3],
                                  name='input_tensor'
                                  )

    phase = tf.constant('test', tf.string)

    net = derain_drop_net.DeRainNet(phase=phase)
    output, attention_maps = net.inference(input_tensor=input_tensor, name='derain_net')

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()

    save_folder = folder_path+'/Attentive_GAN_res/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        for image_path in glob(f'{folder_path}/*.png'):

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
            image_vis = image
            image = np.divide(np.array(image, np.float32), 127.5) - 1.0
            output_image, atte_maps = sess.run(
                [output, attention_maps],
                feed_dict={input_tensor: np.expand_dims(image, 0)})

            output_image = output_image[0]
            for i in range(output_image.shape[2]):
                output_image[:, :, i] = minmax_scale(output_image[:, :, i])

            output_image = np.array(output_image, np.uint8)

            
            cv2.imwrite(save_folder+image_path[-11:-4]+'_src.png', image_vis)
            cv2.imwrite(save_folder+image_path[-11:-4]+'_res.png', output_image)

if __name__ == '__main__':
    # init args
    args = init_args()

    # test model

    # modification to the model
    if args.batch_experiment:
        batch_test_model(args.folder_path, args.weights_path)
    else:
        test_model(args.image_path, args.weights_path, args.label_path)
