# -*- coding: utf-8 -*
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import configparser
import video2frames
import run_prepare_recog
import os, sys
import run_recognition
import cv2
import os.path
#from vidstab import VidStab
import numpy as np
np.set_printoptions(threshold=np.inf) 
import moviepy.editor as mpe
config = configparser.ConfigParser()
config.read(r'/media/zhangyue/80253940-da51-443b-af59-03e8ba01ec83/zhangyue/thomascodes/Vessel(code)/retina-unet/configuration.txt')
import ipdb
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

#fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)
datarootpath = './Vessel(data)/'
train=False
#videoid = '20190215085036'  #20190215085036　20190215085145  20190306120208  20190306120335

def train_func():
	#########################################
	# Step 1: train the network
	#########################################
	config.set('experiment name', 'name', 'test') # the model saved
	config.set('experiment dir', 'dir', datarootpath) # the model saved
	config.set('training settings', 'nohup','False') 
	os.system('python3 ./retina-unet/run_training.py')

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

	with open('./retina-unet/configuration.txt', 'w') as configfile:
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

def recognition_func():
	# Step 3: Image Recognition
	config.set('recognition settings', 'name','test')	# model folder
	config.set('recognition settings', 'dir',datarootpath)# result saved in datasets_recognition folder
	config.set('recognition settings', 'nohup','False') 
	run_recognition.gpurecognition()

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
		#antinoise = cv2.fastNlMeansDenoisingColored(color, None, 10, 10, 7,21)
		print('f_'+str(i)),
		#print(colorframe.shape)
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
	#os.mkdir(path_data_result+videoid)

	for i in range(filesnumber):
		filename_after = path_data_after+videoid+'/'+str(i+1)+'.tif'
		gray_after = cv2.imread(filename_after)
		print(type(gray_after))
		print(gray_after.shape[0])
		print(gray_after.shape[1])
		filename_before = path_data_before+videoid+'/'+"images"+'/'+str(i+1)+'.tif'
		gray_before = cv2.imread(filename_before)
		# print(type(gray_before))
		# print(gray_before.shape[0])
		# print(gray_before.shape[1])
		result = cv2.bitwise_and(gray_before, gray_after)
		filename_result = path_data_result+videoid+'/'+str(i+1)+'.tif'
		#os.mkdir(path_data_result+videoid)
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
			# print(type(gray))
			# print(gray.shape[0])
			# print(gray.shape[1])
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

def classify_hist_with_split(image1, image2):
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    # image1 = cv2.resize(image1, size)
    # image2 = cv2.resize(image2, size)
    # sub_image1 = cv2.split(image1)
    # sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(image1, image2):
        sub_data += calculate(im1, im2)
    return sub_data

def get_data_testing_overlap(filename,test_imgs_original):

	test_imgs_original = test_imgs_original[0:580, 0:560] 
	#filename2= path_data_after+videoid+'/'+str(i+1)+'.tif'
	#test_masks = cv2.imread(filename2,0)
	#test_masks = test_masks[0:580, 0:560] 
	test_imgs=test_imgs_original
	print ("\n test images shape:")
	print (test_imgs.shape)
	#print ("\n test mask shape:")
	#print (test_masks.shape)
	print ("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))

	# 按照顺序提取图像块 方便后续进行图像恢复（作者采用了overlap策略）
	patch_height = 116
	patch_width = 112
	stride_height = 116
	stride_width =112
	patches_imgs_test = extract_ordered_overlap(filename,test_imgs,patch_height,patch_width,stride_height,stride_width)
	print ("\n test PATCHES images shape:")
	print (patches_imgs_test.shape)
	print ("test PATCHES images range (min-max): " +
		str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))
	print (test_imgs.shape)
	#return patches_imgs_test, test_imgs.shape[0], test_imgs.shape[1], test_masks #原始大小
	return patches_imgs_test, test_imgs.shape

#Divide all the full_imgs in pacthes
def extract_ordered_overlap(filename,full_imgs, patch_h, patch_w,stride_h,stride_w):
    #assert (len(full_imgs.shape)==4)  #4D arrays
    #assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[0]  #height of the full image
    img_w = full_imgs.shape[1] #width of the full image
    print("img_h:"+str(img_h))
    print("pathc_h:"+str(patch_h))
    print("stride_h:"+str(stride_h))
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    print("Number of patches on h : " +str(((img_h-patch_h)//stride_h+1)))
    print("Number of patches on w : " +str(((img_w-patch_w)//stride_w+1)))
    print("number of patches per image: " +str(N_patches_img))
    patches = np.empty((N_patches_img,patch_h,patch_w))
    #iter over the total number of patches (N_patches)
    #loop over the full images 
    iter_tot = 0
    os.mkdir("/media/zhangyue/80253940-da51-443b-af59-03e8ba01ec83/zhangyue/thomascodes/Vessel(code)/Vessel(data)/split/"+videoid+"/"+filename)
    for h in range(5):
        for w in range(5):
            patch = full_imgs[h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
            cv2.imwrite("/media/zhangyue/80253940-da51-443b-af59-03e8ba01ec83/zhangyue/thomascodes/Vessel(code)/Vessel(data)/split/"+videoid+"/"+filename+"/"+str(iter_tot+1)+".tif",patch)
            print (patch.shape)
            #print (patch)
            patches[iter_tot]=patch
            iter_tot +=1  #total
			#print(patch.shape)
    print("iter_tot:"+str(iter_tot)) 
    assert (iter_tot==25)
    #print (patches)
    print(patches.shape)
    return patches  #array with all the full_imgs divided in patches

def velocity_predict():
    path_data_result = config.get('recognition settings', 'path_result_hdf')
    filesnumber = len(os.listdir(path_data_result+videoid))
    print("filesnumber:"+str(filesnumber))
    v=np.zeros((500,25))
    #for i in range(filesnumber-1):
    for i in range(0,filesnumber-3,2):
        filename1 = path_data_result+videoid+'/'+str(i+2)+'.tif'
        filename2 = path_data_result+videoid+'/'+str(i+4)+'.tif'
        test_imgs_original1 = cv2.imread(filename1,0)
        test_imgs_original2 = cv2.imread(filename2,0)

        # patches1 = np.empty(((100,58,56,3)))
        # patches2 = np.empty(((100,58,56,3)))
        # patches1 = np.array(patches1) 
        # patches2 = np.array(patches2) 
        if not os.path.exists("/media/zhangyue/80253940-da51-443b-af59-03e8ba01ec83/zhangyue/thomascodes/Vessel(code)/Vessel(data)/split/"+videoid+"/"+str(i+2)):
            get_data_testing_overlap(str(i+2),test_imgs_original1)
        get_data_testing_overlap(str(i+4),test_imgs_original2)

        for j in range(24):
            filename_1= "/media/zhangyue/80253940-da51-443b-af59-03e8ba01ec83/zhangyue/thomascodes/Vessel(code)/Vessel(data)/split/"+videoid+"/"+str(i+2)+"/"+str(j+1)+".tif"
            filename_2= "/media/zhangyue/80253940-da51-443b-af59-03e8ba01ec83/zhangyue/thomascodes/Vessel(code)/Vessel(data)/split/"+videoid+"/"+str(i+4)+"/"+str(j+2)+".tif"
            gray1 = cv2.imread(filename_1,0)
            gray2 = cv2.imread(filename_2,0)
            n= classify_hist_with_split(gray1, gray2)
            if n < 94:
                v[i][j] = 5
            elif 94 < n < 96:
                v[i][j] = 4
            elif 94 < n < 100:
                v[i][j] = 3
            elif 94 < n < 104:
                v[i][j] = 2
            elif 94 < n < 104:
                v[i][j] = 1
            elif n > 104:
                v[i][j] = 0
            x = int(j/5+1) ; y = int((j+1)%5)

            if (y==0):
                y=5
            x_2 = x*97 ; y_2 = y*93

            im = cv2.imread("/media/zhangyue/80253940-da51-443b-af59-03e8ba01ec83/zhangyue/thomascodes/Vessel(code)/Vessel(data)/finish/"+videoid+"/"+str(i+2)+".tif",0)
            image = drawRectBox(im,int(x_2),int(y_2),int(v[i][j]))
            print("第"+str(i+2)+"张照片"+"第"+str(j+1)+"块的直方图算法差异度(速度）：",v[i][j])
            cv2.imwrite("/media/zhangyue/80253940-da51-443b-af59-03e8ba01ec83/zhangyue/thomascodes/Vessel(code)/Vessel(data)/finish/"+videoid+"/"+str(i+2)+".tif",image)
        print("第"+str(i+2)+"张照片对比完毕") 
        print(v[i])

        #os.mkdir("/media/zhangyue/80253940-da51-443b-af59-03e8ba01ec83/zhangyue/thomascodes/Vessel(code)/Vessel(data)/finish/"+videoid)
        
def drawRectBox(im,x,y,addText):
    #cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2, cv2.LINE_AA)
    #cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1, cv2.LINE_AA)
    #draw = ImageDraw.Draw(image)
    #draw.text((int(rect[0]+1), int(rect[1]-16)), addText.decode("utf-8"), (255, 255, 255), font=fontC)
    #draw.text(x, y, addText, (255, 255, 255), font=fontC)
    font=cv2.FONT_HERSHEY_SIMPLEX#使用默认字体
    #im=np.zeros((50,50,3),np.uint8)#新建图像，注意一定要是uint8
    img=cv2.putText(im,str(addText),(x,y),font,0.6,(255,255,255),2)#添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细

    return img

def video_stable_func():
	video_name = datarootpath+'datasets_recognition/generated_'+videoid +'.mp4' 
	stable_name = datarootpath+'datasets_recognition/generated_stable_'+videoid +'.mp4' 
	#stabilizer = VidStab()
	#stabilizer.stabilize(input_path=video_name, output_path=stable_name)

videoid = ""

videoids=['20190215085036','20190215085145','20190306120208','20190306120335']
for i in range(4):
	videoid=videoids[i]
	# print(videoid)
	config.set('recognition settings', 'video_name',videoid)
	with open('./retina-unet/configuration.txt', 'w') as configfile:
		config.write(configfile)

	if (train):
		train_func()
	else:
		# samples = video_frame_func()
		# patches = 20
		# for i in range(int(samples/patches)+1):
		# 	idfrom = i*20
		# 	idto = min((i+1)*20,samples)
		# 	print('in '+ str(i))
		# 	config.set('recognition settings', 'Imgs_to_test_from', str(idfrom))
		# 	config.set('recognition settings', 'Imgs_to_test_to', str(idto))
		# 	with open(r'./retina-unet/configuration.txt', 'w') as configfile:
		# 		config.write(configfile)
		# 	#img_hdf_func()
		# 	#recognition_func()

		#img_dilate()
		#img_average()
		#img_and_img()
		#get_data_testing_overlap()
		velocity_predict()
		#imgs_video_func()
        #video_stable_func()