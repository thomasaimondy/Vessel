#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image
import sys
from skimage import color
import configparser
import cv2

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


sys.path.insert(0,'./retina-unet/')
# from run_prepare_recog import *

#========= CONFIG FILE TO READ FROM =======
config = configparser.ConfigParser()
config.read('../utils/configuration.txt')

#------------Path of the images --------------------------------------------------------------
# #train
data_root = config.get('training settings', 'raw_data_root_path')

original_imgs_train = data_root + "DRIVE/training_tif/images/"
groundTruth_imgs_train = data_root + "DRIVE/training_tif/1st_manual/"
borderMasks_imgs_train = data_root + "DRIVE/training_tif/mask/"
#test
original_imgs_test = data_root + "DRIVE/test_tif/images/"
groundTruth_imgs_test = data_root + "DRIVE/test_tif/1st_manual/"
borderMasks_imgs_test = data_root + "DRIVE/test_tif/mask/"


# original_imgs_train = "DRIVE/training_tif/images/"
# groundTruth_imgs_train = "DRIVE/training_tif/1st_manual/"
# borderMasks_imgs_train = "DRIVE/training_tif/mask/"
# #test
# original_imgs_test = "DRIVE/test_tif/images/"
# groundTruth_imgs_test = "DRIVE/test_tif/1st_manual/"
# borderMasks_imgs_test = "DRIVE/test_tif/mask/"

#---------------------------------------------------------------------------------------------

# Nimgs = 20
channels = 3
height = 584
width = 565
dataset_path = data_root + "DRIVE_datasets_training_testing/"
imgtype = "tif"

def checkrgb(inarray):
    # inarray.shape[-1]
    if (np.array(inarray).shape[-1]==3):
        # print("groundTruth max: " + str(np.max(np.array(inarray))))
        print("groundTruth max: " + str(np.max(cv2.cvtColor(np.array(inarray), cv2.COLOR_BGR2GRAY))))
        print("groundTruth min: " + str(np.min(cv2.cvtColor(np.array(inarray), cv2.COLOR_BGR2GRAY))))
        return cv2.cvtColor(np.array(inarray), cv2.COLOR_BGR2GRAY)
    else:
        return np.array(inarray)

def get_datasets(Nimgs,imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
    # print(imgs_dir)
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        # print("thomas:"+str(len(files)))
        for i in range(len(files)):
            #original
            print("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1." + imgtype
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            g_truth = checkrgb(g_truth) # need to check
            groundTruth[i] = np.asarray(g_truth)
            print("groundTruth max: " + str(np.max(groundTruth[i])))
            print("groundTruth min: " + str(np.min(groundTruth[i])))
            #corresponding border masks
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:2] + "_training_mask." + imgtype
            elif train_test=="test":
                border_masks_name = files[i][0:2] + "_test_mask." + imgtype
            else:
                print("specify if train or test!!")
                exit()
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    print("groundTruth max: " +str(np.max(groundTruth)))
    print("groundTruth min: " +str(np.min(groundTruth)))
    print("border_masks max: " +str(np.max(border_masks)))
    print("border_masks min: " +str(np.min(border_masks)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, groundTruth, border_masks

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(20,original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(1,original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
print("saving test datasets")
write_hdf5(imgs_test,dataset_path + "DRIVE_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")
