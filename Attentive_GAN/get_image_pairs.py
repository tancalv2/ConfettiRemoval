import os
from shutil import copy
from glob import glob
from tqdm import tqdm
import cv2

def get_image_pairs():
    """Function to extract frames from PS videos and form image pairs"""

    # make sure you have contents in video_dir and truth_dir
    # img_PS has all the ground truth image while video_dir contains
    # all the confetti animation videos in .mp4 format
    video_dir = "./PS_render"
    output_dir = "./image_pairs"
    truth_dir = "./img_PS"

    if not os.path.exists(output_dir+'/input'):
        os.makedirs(output_dir+'/input')

    if not os.path.exists(output_dir+'/label'):
        os.makedirs(output_dir+'/label')

    img_count = 1
    for video in tqdm(glob(video_dir+"/img*.mp4")):
        cv2_video = cv2.VideoCapture(video)
        success,image = cv2_video.read()
        while success:
            cv2.imwrite(f'{output_dir}/input/{img_count}.png', image)
            copy(f'{truth_dir}/{video[-9:-4]}.png', f'{output_dir}/label/{img_count}.png')
            success,image = cv2_video.read()
            img_count += 1

def attentive_gan_rename():
    image_dir = './image_pairs/attentive_gan_train'

    for image in tqdm(glob(image_dir+'/clean_image/*.png')):
        os.rename(image, image[:-4]+'_clean.png')

    for image in tqdm(glob(image_dir+'/rain_image/*.png')):
        os.rename(image, image[:-4]+'_rain.png')

attentive_gan_rename()

#get_image_pairs()