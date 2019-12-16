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
import configparser
import ipdb

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

def get_datasets(imgs_dir,borderMasks_dir,train_test="null"):
    # print(imgs_dir)
    config = configparser.ConfigParser()
    config.readfp(open('./utils/recog_config.txt'))

    n_from = int(config.get('recognition settings', 'Imgs_to_test_from'))
    n_to = int(config.get('recognition settings', 'Imgs_to_test_to'))
    Nimgs = n_to - n_from
    channels = int(config.get('recognition settings', 'image_channel'))
    height = int(config.get('recognition settings', 'image_height'))
    width = int(config.get('recognition settings', 'image_width'))
    imgtype = config.get('recognition settings', 'image_type')

    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        # print("thomas:"+str(len(files)))
        for i in range(len(files)):
            #original
            imgid = int(files[i][0:-4])-1 # know the id of image (name)
            if imgid not in range(n_from,n_to):
                continue
            print("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            idinarr = imgid - n_from      # idinarr should always start from 0      
            imgs[idinarr] = np.asarray(img)
            #corresponding border masks
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:-4] + "." + imgtype
            elif train_test=="test":
                border_masks_name = files[i][0:-4] + "." + imgtype
            else:
                print("specify if train or test!!")
                exit()
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[idinarr] = np.asarray(b_mask) # know the id of image

    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    assert(np.max(border_masks)==255)
    assert(np.min(border_masks)==0)
    print("border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, border_masks

def get_standart_dataformat():
    #config file to read from
    config = configparser.ConfigParser()
    config.readfp(open('./utils/recog_config.txt'))
    #===========================================
    original_imgs_test = config.get('recognition settings', 'path_local_oriimgs')
    borderMasks_imgs_test = config.get('recognition settings', 'path_local_orimasks')
    #---------------------------------------------------------------------------------------------

    #getting the testing datasets
    imgs_test, border_masks_test = get_datasets(original_imgs_test,borderMasks_imgs_test,"test")
    return imgs_test, border_masks_test

def frames2hdf5():
    config = configparser.ConfigParser()
    config.readfp(open('./utils/recog_config.txt'))

    dataset_path = config.get('recognition settings', 'path_local_hdf')
    imgs_original = config.get('recognition settings', 'imgs_original')
    border_masks = config.get('recognition settings', 'border_masks')

    # execute only if run as a script
    imgs_test, border_masks_test = get_standart_dataformat()

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # ipdb.set_trace()
    write_hdf5(imgs_test,dataset_path + imgs_original)
    write_hdf5(border_masks_test,dataset_path + border_masks)

if __name__ == "__main__":
    frames2hdf5()
    