# Vessel
Vessel Recognition and speed prediction.

<<<<<<< HEAD
Additional package need to be installed:
```
conda install opencv

```

Training procedure:
```
python prepare_datasets_DRIVE.py # for prepare the training dataset, change the directory name in it.
python main.py # with the setting of ‘train=True’ in it.
```


Test procedure:
```
python main.py # with the settting of 'train=False' in it.
```
=======
The main steps of this project are: firstly, the blood vessel is identified and extracted, the blood vessel part is white, the background part is black, and the image of the extracted blood vessel is equivalent to a mask, which performs an "and" operation with the original video to retain the blood flow part, and the rest is black. Then, the speed is predicted by comparing the Grayscale histogram between adjacent frames.

Original image

![Image text](https://github.com/thomasaimondy/Vessel/blob/master/images/1.jpg)

Image of extracted blood vessels

![Image text](https://github.com/thomasaimondy/Vessel/blob/master/images/2.jpg)

the image after "and"

![Image text](https://github.com/thomasaimondy/Vessel/blob/master/images/3.jpg)

Speed recognition image

![Image text](https://github.com/thomasaimondy/Vessel/blob/master/images/4.jpg)


Additional package need to be installed:
```
conda env create -f vesselenv.yml

conda install opencv

```
(1) Change the environment of variables in retina-unet/configuration.txt

(2) Training procedure:
```
python main.py # with the setting of ‘train=True’ in it.
```

(3) Test procedure:
```
python main.py # with the settting of 'train=False' in it.
```
(4)The image of the extracted blood vessel is saved in the datasets_recognition folder, and the result of the speed recognition is saved in the finish folder.
>>>>>>> 4bdd8a3a25a10b3bd01f6512109927970f646738
