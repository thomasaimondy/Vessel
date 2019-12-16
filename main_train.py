import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import configparser
import os, sys
import cv2
import os.path
#from vidstab import VidStab
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
from retina_unet.src.retinaNN_training import get_unet, get_gnet
from retina_unet.src.retinaNN_training import training_unet, training_gnet
from retina_unet.src.extract_patches import get_data_training
from retina_unet.src.help_functions import *
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
if __name__ == '__main__':
	config = configparser.ConfigParser()
	config.read('/home/zhangyue/codes/thomascodes/Vessel_new/utils/train_config.txt')
	name_dir = config.get('experiment dir', 'dir')
	nohup = config.getboolean('training settings', 'nohup')
	name_experiment = config.get('experiment name', 'name')
	path_data = config.get('data paths', 'path_local')
	N_epochs = int(config.get('training settings', 'N_epochs'))
	batch_size = int(config.get('training settings', 'batch_size'))
	patches_imgs_train, patches_masks_train = get_data_training(
    		DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    		DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    		patch_height = int(config.get('data attributes', 'patch_height')),
    		patch_width = int(config.get('data attributes', 'patch_width')),
    		N_subimgs = int(config.get('training settings', 'N_subimgs')),
    		inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
	)
	N_sample = min(patches_imgs_train.shape[0], 40)
	visualize(group_images(patches_imgs_train[0:N_sample, :, :, :], 5),name_dir + '/' + name_experiment + '/' + "sample_input_imgs")  # .show()
	visualize(group_images(patches_masks_train[0:N_sample, :, :, :], 5),name_dir + '/' + name_experiment + '/' + "sample_input_masks")  # .show()
	n_ch = patches_imgs_train.shape[1]
	patch_height = patches_imgs_train.shape[2]
	patch_width = patches_imgs_train.shape[3]
	model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
	
	json_string = model.to_json()
	open(name_dir+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)
	plot(model, to_file=name_dir+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
	patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
	training_unet(model, patches_imgs_train, patches_masks_train, N_epochs, batch_size,name_dir,name_experiment)
