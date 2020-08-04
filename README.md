# ACM-MM-FRDT
Code repository for our paper entilted "Feature Reintegration over Differential Treatment: A Top-down and Adaptive Fusion Network for RGB-D Salient Object Detection" accepted at ACM MM 2020 (poster).

## Usage Instructions
### > Requirment

+ Ubuntu 18
+ PyTorch 1.3.1
+ CUDA 10.1
+ Cudnn 7.5.1
+ Python 3.7
+ Numpy 1.17.3

#### Train/Test
+ test     
Download related dataset [**link**](https://github.com/jiwei0921/RGBD-SOD-datasets), and set the param '--phase' as "**test**" and '--param' as '**True**' in ```demo.py```. Meanwhile, you need to set **dataset path** and **checkpoint name** correctly.
```
python demo.py
```
+ train     
Our train-augment dataset [**link**](https://pan.baidu.com/s/18nVAiOkTKczB_ZpIzBHA0A) [ fetch code **haxl** ] / [train-ori dataset](https://pan.baidu.com/s/1B8PS4SXT7ISd-M6vAlrv_g), and set the param '--phase' as "**train**" and '--param' as '**True**'(loading checkpoint) or '**False**'(no loading checkpoint) in ```demo.py```. Meanwhile, you need to set **dataset path** and **checkpoint name** correctly.  
```
python demo.py
```



