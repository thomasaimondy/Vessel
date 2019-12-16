# -*- coding: utf-8 -*
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import configparser
from utils import video2frames
from utils import run_prepare_recog
import os, sys
# import run_recognition
import main_recognition
import cv2
import os.path
#from vidstab import VidStab
import numpy as np
np.set_printoptions(threshold=np.inf) 
import moviepy.editor as mpe
config = configparser.ConfigParser()
config.read(r'./utils/recog_config.txt')
import ipdb
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

#fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)
datarootpath = '../Data/Vessel/'
train = False
test = False
#videoid = '20190215085036'  #20190215085036　20190215085145  20190306120208  20190306120335

def video_frame_func():
	#########################################
	# Step 1: get each frame of the video
	# Todo: Change the name of target video path : 20190306120335
	# path_video = /home/bibo/Data/Vessel/videos/20190306120335.avi   # input video
	# path_local_oriimgs = /home/bibo/Data/Vessel/20190306120335/images/  # output frames
	# path_local_orimasks = /home/bibo/Data/Vessel/20190306120335/masks/  # output frames
	#
	#########################################
	# videoid = '20190215085145'
	# videoid = '20190306120208'
	# videoid = '20190306120335'
	config.set('recognition settings', 'path_video', \
		datarootpath+'videos/'+videoid+'.avi')
	config.set('recognition settings', 'path_local_oriimgs', \
		datarootpath+videoid+'/images/')
	config.set('recognition settings', 'path_local_orimasks', \
		datarootpath+videoid+'/masks/')

	with open('./utils/recog_config.txt', 'w') as configfile:
		config.write(configfile)

	imgnum = video2frames.v2f()
	print("image number = "+str(imgnum))
	return imgnum

def img_hdf_func():
	# Step 2: Preprossing, images to hdf5 file
	config.set('recognition settings', 'path_local_oriimgs', \
		datarootpath+videoid+'/images/')
	config.set('recognition settings', 'path_local_orimasks', \
		datarootpath+videoid+'/masks/')
	config.set('recognition settings', 'path_local_hdf',\
		datarootpath + 'datasets_recognition/')
	config.set('recognition settings', 'imgs_original','dataset_imgs_test.hdf5')
	config.set('recognition settings', 'border_masks','dataset_borderMasks_test.hdf5')
	run_prepare_recog.frames2hdf5()


def imgs_video_func():
	video_name = datarootpath+'datasets_recognition/result_'+videoid +'.mp4'
	path_data = config.get('recognition settings', 'path_result_hdf')
	filesnumber = len(os.listdir(path_data+videoid))
	img_list = []
	count = 0
	for i in range(0,filesnumber*2,2):
		filename = path_data+videoid+'/'+str(i+2)+'.tif'
		gray = cv2.imread(filename,0)
		color = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
		print('f_'+str(i)),
		img_list.append(np.array(color))
	clip = mpe.ImageSequenceClip(img_list, fps=25)
	clip.write_videofile(video_name)

def img_and_img():
	path_data = config.get('recognition settings', 'path_after_hdf')
	path_data_after = path_data+'dilate/'
	path_data_before = config.get('recognition settings', 'path_before_hdf')
	path_data_result = config.get('recognition settings', 'path_result_hdf')
	filesnumber = len(os.listdir(path_data_after+videoid))
	print("filesnumber"+str(filesnumber))
	for i in range(filesnumber):
		filename_after = path_data_after+videoid+'/'+str(i+1)+'.tif'
		gray_after = cv2.imread(filename_after)
		print(type(gray_after))
		print(gray_after.shape[0])
		print(gray_after.shape[1])
		filename_before = path_data_before+videoid+'/'+"images"+'/'+str(i+1)+'.tif'
		gray_before = cv2.imread(filename_before)
		result = cv2.bitwise_and(gray_before, gray_after)
		filename_result = path_data_result+videoid+'/'+str(i+1)+'.tif'
		cv2.imwrite(filename_result,result)
		print ("filename_result"+filename_result)

def img_average():
	path_data_after = config.get('recognition settings', 'path_after_hdf')
	filesnumber = len(os.listdir(path_data_after+videoid))
	print("filesnumber:"+str(filesnumber))

	for i in range(filesnumber-1):
		sum = np.zeros(shape=(584,565,3))
		for j in range (2):
			filename= path_data_after+videoid+'/'+str(i+j+1)+'.tif'
			print(filename)
			gray = cv2.imread(filename)
			gray = gray.astype(np.float32)
			sum = sum + gray
		average = sum / 2
		average = average.astype(np.uint8)
		cv2.imwrite(filename,average)

		print ("filename_result:"+filename)

def img_dilate():
	path_data_after = config.get('recognition settings', 'path_after_hdf')
	filesnumber = len(os.listdir(path_data_after+videoid))
	print("filesnumber:"+str(filesnumber))
	os.mkdir(path_data_after+'dilate/'+videoid)

	kernel = np.ones((5,5),np.uint8)
	for i in range(filesnumber-1):
		filename= path_data_after+videoid+'/'+str(i+1)+'.tif'
		print(filename)
		gray = cv2.imread(filename)
		print(type(gray))
		print(gray.shape[0])
		print(gray.shape[1])
		dst = cv2.dilate(gray,kernel)
		print("dst")
		print(type(dst))
		print(dst.shape[0])
		print(dst.shape[1])

		erosion = cv2.erode(dst,kernel)
		filename2= path_data_after+'dilate/'+videoid+'/'+str(i+1)+'.tif'
		cv2.imwrite(filename2,erosion)

		print ("filename_result:"+filename2)

# 计算单通道的直方图的相似值
def calculate(image1, image2):
	hist1 = cv2.calcHist([image1], [0], None, [255], [1, 255.0])
	hist2 = cv2.calcHist([image2], [0], None, [255], [1, 255.0])
	# 计算直方图的重合度
	degree = 0
	for i in range(len(hist1)):
		if hist1[i] != hist2[i]:
			degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
		else:
			degree = degree + 1
	degree = degree / len(hist1)
	return degree

def video_stable_func():
    video_name = datarootpath+'datasets_recognition/generated_'+videoid +'.mp4'
    stable_name = datarootpath+'datasets_recognition/generated_stable_'+videoid +'.mp4'
    #stabilizer = VidStab()
	#stabilizer.stabilize(input_path=video_name, output_path=stable_name)


videoid = ""
#videoids = ['20190306120335']
videoids=['20190215085036','20190215085145','20190306120208']
for i in range(len(videoids)):
	videoid=videoids[i]
	# print(videoid)
	config.set('recognition settings', 'video_name',videoid)
	with open('./utils/recog_config.txt', 'w') as configfile:
		config.write(configfile)

	samples = video_frame_func()
	patches = 20
	for i in range(int(samples/patches)+1):
		idfrom = i*20
		idto = min((i+1)*20,samples)
		print('in '+ str(i))
		config.set('recognition settings', 'Imgs_to_test_from', str(idfrom))
		config.set('recognition settings', 'Imgs_to_test_to', str(idto))
		with open('./utils/recog_config.txt', 'w') as configfile:
			config.write(configfile)
		img_hdf_func()
		main_recognition.gpurecognition()

