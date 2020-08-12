# ACM-MM-FRDT
Code repository for our paper entilted "Feature Reintegration over Differential Treatment: A Top-down and Adaptive Fusion Network for RGB-D Salient Object Detection" accepted at ACM MM 2020 (poster).

# Overall
![avatar](https://github.com/jack-admiral/ACM-MM-FRDT/blob/master/figures/overview.png)


# Usage Instructions
### > Requirment

+ Ubuntu 18
+ PyTorch 1.3.1
+ CUDA 10.1
+ Cudnn 7.5.1
+ Python 3.7
+ Numpy 1.17.3

#### Train/Test
Before training or testing, please make sure the size of all images is same.
+ test     
Download related dataset [**link**](https://github.com/jiwei0921/RGBD-SOD-datasets) and the pretrained model [**link**](https://pan.baidu.com/s/1EIfJ-8-RxrRrEneBLtTWYw) [fetch code **53x0**], and set the param '--phase' as "**test**" and '--param' as '**True**' in ```demo.py```. Meanwhile, you need to set **dataset path** and **checkpoint name** correctly.
```
python demo.py
```
+ train     
Our train-augment dataset [**link**](https://pan.baidu.com/s/18nVAiOkTKczB_ZpIzBHA0A) [ fetch code **haxl** ] , and set the param '--phase' as "**train**" and '--param' as '**True**'(loading checkpoint) or '**False**'(no loading checkpoint) in ```demo.py```. Meanwhile, you need to set **dataset path** and **checkpoint name** correctly.  
```
python demo.py
```

# Comparsion
![avatr](https://github.com/jack-admiral/ACM-MM-FRDT/blob/master/figures/comparsion.png)

# Results
The results of our method in 7 datasets in [**here**](https://pan.baidu.com/s/1uCHCUDqpVBZ6Lg-0THfugA) [fetch code **t2bx**]

# Contact Us
If you have any questions, please contact us ( zhangyu4195@mail.dlut.edu.cn ).



