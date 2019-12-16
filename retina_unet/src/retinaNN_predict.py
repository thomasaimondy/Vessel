###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################
import sys
sys.path.insert(0, './lib/')
#Python
import numpy as np
import configparser
from matplotlib import pyplot as plt
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

def ela_visualize(average_mode,pred_patches,new_height,new_width,stride_height,stride_width,test_imgs_orig,masks_test,test_border_masks,full_img_height,full_img_width,N_visual,path_experiment, name_experiment):
	#========== Elaborate and visualize the predicted images ====================
	pred_imgs = None
	orig_imgs = None
	gtruth_masks = None	
	if average_mode == True:
    		pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
    		orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])    #originals
    		gtruth_masks = masks_test  #ground truth masks
	else:
    		pred_imgs = recompone(pred_patches,13,12)       # predictions
    		orig_imgs = recompone(patches_imgs_test,13,12)  # originals
    		gtruth_masks = recompone(patches_masks_test,13,12)  #masks
	# apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
	kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
	## back to original dimensions
	orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
	pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
	gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
	print ("Orig imgs shape: " +str(orig_imgs.shape))
	print ("pred imgs shape: " +str(pred_imgs.shape))
	print ("Gtruth imgs shape: " +str(gtruth_masks.shape))
	visualize(group_images(orig_imgs,N_visual),path_experiment+'/' + name_experiment + '/'+"all_originals")#.show()
	visualize(group_images(pred_imgs,N_visual),path_experiment+'/' + name_experiment + '/'+"all_predictions")#.show()
	visualize(group_images(gtruth_masks,N_visual),path_experiment+'/' + name_experiment + '/'+"all_groundTruths")#.show()
	#visualize results comparing mask and prediction:
	assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
	N_predicted = orig_imgs.shape[0]
	group = N_visual
	assert (N_predicted%group==0)
	for i in range(int(N_predicted/group)):
    		orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
    		masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
    		pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
    		total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe),axis=0)
    		visualize(total_img,path_experiment+'/' + name_experiment + '/'+name_experiment +"_Original_GroundTruth_Prediction"+str(i))#.show()
	return orig_imgs, pred_imgs, gtruth_masks

def evaluate(pred_imgs, gtruth_masks, test_border_masks,path_experiment, name_experiment):
	print ("\n\n========  Evaluate the results =======================")
	#predictions only inside the FOV
	y_scores, y_true = pred_only_FOV(pred_imgs,gtruth_masks, test_border_masks)  #returns data only inside the FOV
	print ("Calculating results only inside the FOV:")
	print ("y scores pixels: " +str(y_scores.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(pred_imgs.shape[0]*pred_imgs.shape[2]*pred_imgs.shape[3]) +" (584*565==329960)")
	print ("y true pixels: " +str(y_true.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(gtruth_masks.shape[2]*gtruth_masks.shape[3]*gtruth_masks.shape[0])+" (584*565==329960)")
	y_true = y_true.astype(int)
	#Area under the ROC curve
	print("y_true max: " + str(np.max(y_true)))
	print("y_true min: " + str(np.min(y_true)))
	# print(y_scores)
	# print(type(y_scores))
	# print(np.size(y_scores))
	y_scores = (y_scores > 0.5).astype(int)
	# print(y_scores)
	# print(type(y_scores))
	# print(np.size(y_scores))
	print("y_scores max: " + str(np.max(y_scores)))
	print("y_scores min: " + str(np.min(y_scores)))
	fpr, tpr, thresholds = roc_curve(y_true, y_scores)

	AUC_ROC = roc_auc_score(y_true, y_scores)
	print ("\nArea under the ROC curve: " +str(AUC_ROC))
	plt.figure()
	plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
	plt.title('ROC curve')
	plt.xlabel("FPR (False Positive Rate)")
	plt.ylabel("TPR (True Positive Rate)")
	plt.legend(loc="lower right")
	plt.savefig(path_experiment+'/' + name_experiment + '/'+"ROC.png")

	#Precision-recall curve
	precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
	precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
	recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
	AUC_prec_rec = np.trapz(precision,recall)
	print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
	plt.figure()
	plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
	plt.title('Precision - Recall curve')
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.legend(loc="lower right")
	plt.savefig(path_experiment+'/' + name_experiment + '/'+"Precision_recall.png")

	
	#Confusion matrix
	threshold_confusion = 0.5
	print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
	y_pred = np.empty((y_scores.shape[0]))
	for i in range(y_scores.shape[0]):
    		if y_scores[i]>=threshold_confusion:
        		y_pred[i]=1
    		else:
        		y_pred[i]=0
	confusion = confusion_matrix(y_true, y_pred)
	print (confusion)
	accuracy = 0
	if float(np.sum(confusion))!=0:
    		accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
	print ("Global Accuracy: " +str(accuracy))
	
	specificity = 0
	if float(confusion[0,0]+confusion[0,1])!=0:
    		specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
	print ("Specificity: " +str(specificity))
	
	sensitivity = 0
	if float(confusion[1,1]+confusion[1,0])!=0:
    		sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
	print ("Sensitivity: " +str(sensitivity))
	
	precision = 0
	if float(confusion[1,1]+confusion[0,1])!=0:
    		precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
	print ("Precision: " +str(precision))

	#Jaccard similarity index
	jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
	print ("\nJaccard similarity score: " +str(jaccard_index))

	#F1 score
	F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
	print ("\nF1 score (F-measure): " +str(F1_score))

	#Save the results
	file_perf = open(path_experiment +'/' + name_experiment + '/' + 'performances.txt', 'w')
	file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
	file_perf.close()
