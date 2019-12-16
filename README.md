# Vessel
Vessel Recognition and speed prediction.

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
