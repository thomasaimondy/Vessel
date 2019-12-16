# -*- coding: utf-8 -*
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import configparser
import os, sys
import cv2
import os.path
#from vidstab import VidStab
from keras.models import model_from_json
from keras.models import Model
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
from retina_unet.src.retinaNN_training import get_unet, get_gnet
from retina_unet.src.retinaNN_training import training_unet, training_gnet
from retina_unet.src.extract_patches import get_data_training
from retina_unet.src.extract_patches import get_data_testing
from retina_unet.src.extract_patches import get_data_testing_overlap
from retina_unet.src.help_functions import *
from retina_unet.src.retinaNN_predict import ela_visualize,evaluate

if __name__ == '__main__':
	config = configparser.ConfigParser()
	config.read('/home/zhangyue/codes/thomascodes/Vessel_new/utils/test_config.txt')
	name_dir = config.get('experiment dir', 'dir')
	nohup = config.getboolean('testing settings', 'nohup')
	name_experiment = config.get('experiment name', 'name')
	if os.path.exists(name_dir):
		pass
	else:
		os.system('mkdir -p' + name_dir)
	path_data = config.get('data paths', 'path_local')
	DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
	test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
	full_img_height = test_imgs_orig.shape[2]
	full_img_width = test_imgs_orig.shape[3]
	#the border masks provided by the DRIVE
	DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
	test_border_masks = load_hdf5(DRIVE_test_border_masks)
	# dimension of the patches
	patch_height = int(config.get('data attributes', 'patch_height'))
	patch_width = int(config.get('data attributes', 'patch_width'))
	#the stride in case output with average
	stride_height = int(config.get('testing settings', 'stride_height'))
	stride_width = int(config.get('testing settings', 'stride_width'))
	assert (stride_height < patch_height and stride_width < patch_width)
	path_experiment = name_dir
	Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
	#Grouping of the predicted images
	N_visual = int(config.get('testing settings', 'N_group_visual'))
	#====== average mode ===========
	average_mode = config.getboolean('testing settings', 'average_mode')
	patches_imgs_test = None
	new_height = None
	new_width = None
	masks_test  = None
	patches_masks_test = None
	if average_mode == True:
   		 patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        		DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
        		DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
        		Imgs_to_test_from = 0,
        		Imgs_to_test_to = 19,
        		# Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
       		 	patch_height = patch_height,
        		patch_width = patch_width,
        		stride_height = stride_height,
        		stride_width = stride_width
    			)	
	else:
    		patches_imgs_test, patches_masks_test = get_data_testing(
        		DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
        		DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
        		Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
        		patch_height = patch_height,
        		patch_width = patch_width,
   			 )
#load the testing data and the configuration

	#load the model 
	best_last = config.get('testing settings', 'best_last')
	model = model_from_json(open(path_experiment+'/' + name_experiment + '/'+ name_experiment +'_architecture.json').read())
	model.load_weights(path_experiment+'/' + name_experiment + '/'+name_experiment + '_'+best_last+'_weights.h5')
	model.load_weights(path_experiment+'/' + name_experiment + '/'+name_experiment + '_'+best_last+'_weights.h5')
	#Calculate the predictions
	predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
	print ("predicted images size :")
	print (predictions.shape)
	#===== Convert the prediction arrays in corresponding images
	pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")
	
	orig_imgs, pred_imgs, gtruth_masks = ela_visualize(average_mode,pred_patches,new_height,new_width,stride_height,stride_width,test_imgs_orig,masks_test,test_border_masks,full_img_height,full_img_width,N_visual,path_experiment, name_experiment)
	evaluate(pred_imgs, gtruth_masks, test_border_masks,path_experiment, name_experiment)
