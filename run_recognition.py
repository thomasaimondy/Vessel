###################################################
#
#   Script to execute the one image prediction
#
##################################################

import os, sys
import configparser

def gpurecognition():
	#config file to read from
	config = configparser.ConfigParser()
	config.readfp(open(r'./retina-unet/configuration.txt'))
	#===========================================
	#name of the experiment!!
	name_experiment = config.get('experiment name', 'name')
	nohup = config.getboolean('recognition settings', 'nohup')   #std output on log file?
	name_dir = config.get('experiment dir', 'dir') # model place

	run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu{0,1,2,3},floatX=float32 '

	# finally run the prediction
	if nohup:
		print("Run the prediction on GPU  with nohup")
		os.system(run_GPU +' nohup python -u ./retina-unet/src/retinaNN_recognition.py > ' + name_dir + name_experiment+'/'+name_experiment+'_recognition.nohup')
	else:
		print("Run the prediction on GPU (no nohup)")
		os.system(run_GPU +' python ./retina-unet/src/retinaNN_recognition.py')
if __name__ == '__main__':
	gpurecognition()
