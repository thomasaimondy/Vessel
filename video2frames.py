#########################################
# Step 1: get each frame of the video
# Todo: Change the name of target video path : 20190306120335
# path_video = /home/bibo/Data/Vessel/videos/20190306120335.avi   # input video
# path_local_oriimgs = /home/bibo/Data/Vessel/20190306120335/images/  # output frames
# path_local_orimasks = /home/bibo/Data/Vessel/20190306120335/masks/  # output frames
#
#########################################
#from matplotlib import pyplot as plt
#import matplotlib
import os
from shutil import copyfile 
import configparser
import cv2
from moviepy.editor import VideoFileClip
from scipy.misc import imsave

def v2f():
    config = configparser.ConfigParser()
    config.readfp(open(r'./retina-unet/configuration.txt'))
    path_video = config.get('recognition settings', 'path_video')
    saveimgroot = config.get('recognition settings', 'path_local_oriimgs')
    savemaskroot = config.get('recognition settings', 'path_local_orimasks')
    standardmask = config.get('recognition settings', 'path_standardmask')

    image_height = int(config.get('recognition settings', 'image_height')) # 584
    image_width = int(config.get('recognition settings', 'image_width'))   # 565

    if not os.path.exists(saveimgroot):
        os.makedirs(saveimgroot)
    if not os.path.exists(savemaskroot):
        os.makedirs(savemaskroot)

    clip = VideoFileClip(path_video) 
    count =1
    for frames in clip.iter_frames():
    #     backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    #     gray_frames = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
        frames_new = cv2.resize(frames,(image_width,image_height))
        imsave(saveimgroot + str(count)+'.tif', frames_new)
        dstmask = savemaskroot + str(count) + '.tif'
        copyfile(standardmask, dstmask) # save masks
    #     print frames.shape
    #     print gray_frames.shape
        count+=1
    # print('full_images_to_recog = '+str(count-1))
    return count-1


if __name__ == '__main__':
    v2f()
