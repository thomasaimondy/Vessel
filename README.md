# Vessel
Vessel Recognition and speed prediction.

The main steps of this project are: firstly, the blood vessel is identified and extracted, the blood vessel part is white, the background part is black, and the image of the extracted blood vessel is equivalent to a mask, which performs an operation with the original video to retain the blood flow part, and the rest It is black. Then, the speed is predicted by comparing the gray fat map between adjacent frames.



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
