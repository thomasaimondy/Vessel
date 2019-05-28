# Vessel
Vessel Recognition and speed prediction.

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
