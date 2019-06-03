###################################################
#
#   Script to launch the training
#
##################################################

import os, sys
import configparser


#config file to read from
config = configparser.ConfigParser()
config.readfp(open(r'./configuration.txt'))
#===========================================
#name of the experiment
name_experiment = config.get('experiment name', 'name')
name_dir = config.get('experiment dir', 'dir')
nohup = config.getboolean('training settings', 'nohup')   #std output on log file?

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu{0,1,2,3},floatX=float32 '

#create a folder for the results
result_dir = name_dir
print("1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    print("Dir already existing")
elif sys.platform=='win32':
    os.system('mkdir ' + result_dir)
else:
    os.system('mkdir -p ' +result_dir)

print("copy the configuration file in the results folder")
if sys.platform=='win32':
    os.system('copy ./configuration.txt ' + name_dir +name_experiment+'\\'+name_experiment+'_configuration.txt')
else:
    os.system('cp ./configuration.txt ' + name_dir +name_experiment+'/'+name_experiment+'_configuration.txt')

# run the experiment
if nohup:
    print("2. Run the training on GPU with nohup")
    os.system(run_GPU +' nohup python3 -u ./src/retinaNN_training.py > ' + name_dir + name_experiment+'/'+name_experiment+'_training.nohup')
else:
    print("2. Run the training on GPU (no nohup)")
    os.system(run_GPU +' python3 ./src/retinaNN_training.py')

#Prediction/testing is run with a different script
