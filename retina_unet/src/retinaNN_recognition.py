import numpy as np
import configparser
import matplotlib.pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './retina-unet/lib/')
# help_functions.py
from retina_unet.src.help_functions import *
# extract_patches.py
from retina_unet.src.extract_patches import recompone
from retina_unet.src.extract_patches import recompone_overlap
from retina_unet.src.extract_patches import paint_border
from retina_unet.src.extract_patches import kill_border
from retina_unet.src.extract_patches import pred_only_FOV
from retina_unet.src.extract_patches import get_data_testing
from retina_unet.src.extract_patches import get_data_testing_overlap
# pre_processing.py
from retina_unet.src.pre_processing import my_PreProc
from scipy.misc import imsave
import os

sys.path.insert(0,'./retina-unet/')

def elaborate(average_mode,pred_patches, new_height, new_width, stride_height, stride_width, test_border_masks,full_img_height,full_img_width):
	pred_imgs = None
	if average_mode == True:
    		pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
	else:
    		pred_imgs = recompone(pred_patches,13,12)       # predictions
	# apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
	kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
	## back to original dimensions
	pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
	print("pred imgs shape: " +str(pred_imgs.shape))
	return pred_imgs
def save_result(path_data, video_name, Imgs_to_test_from, Imgs_to_test_to,pred_imgs):
	if not os.path.exists(path_data + video_name):
    		os.makedirs(path_data + video_name)
	# save images
	for ite in range(Imgs_to_test_from,Imgs_to_test_to):
    		imgdata = pred_imgs[ite-Imgs_to_test_from][0]
    		imgpath = path_data + video_name + '/' + str(ite+1)+'.tif'
    		imsave(imgpath, imgdata)


