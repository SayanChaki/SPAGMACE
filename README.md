# SPAGMACE
This is an official implementation of the Spatial Attention and Gaussian Mixture Component Extraction

## Prepare
This project uses Python 3.9 and Pytorch 1.8.1.
```
git clone https://github.com/EmoFuncs/GMAIR-pytorch.git
pip install -r requirements.txt
```

Build bbox:
```
cd SPAGMACZ-pytorch/gmair/utils/bbox
python setup.py build
cp build/lib/bbox.so .
```

## Datasets
### CLEVRTEX dataset
[link]([https://drive.google.com/file/d/1BIzWAExc0NDSF_a6RnTvfBMvbXhTAns5/view?usp=sharing](https://github.com/karazijal/clevrtex))

### CLEVR dataset
[link][https://github.com/facebookresearch/clevr-dataset-gen)
Note that annotations are only used for evaluation.



## Train
For CLEVRTEX, download CLEVRTEX dataset. Unzip it, and put it into 'data/CLEVRTEX/'.
Substitute 'config.py' with 'clevrtex.py' in 'gmair/config'
```
cd gmair/config
cp mnist_config.py config.py
cd ../..
```

For CLEVR, download CLEVR dataset. Unzip them, and put them into 'data/clevr/'.
Substitute 'config.py' with 'clevr.py' in 'gmair/config'
```
cd gmair/config
cp mnist_config.py config.py
cd ../..
```

The architecture should be:
```
data
|---CLEVR
|   |---test_images
|   |   |---x.png
|   |
|   |---test_labels
|   |   |---x.txt
|   |
|   |---train_images
|   |   |---y.png
|   |
|   |---train_labels
|       |---y.txt
|   
|---CLEVRYEX
    |---
```

Then,
```
python train.py
```



## Test


```
python test.py
```
