import os

imgs_dir = "/media/zhangyue/80253940-da51-443b-af59-03e8ba01ec83/zhangyue/leftData/DRIVE/training_tif/images"

for path, subdirs, files in os.walk(imgs_dir):
    print(len(files))


