# Iterative fully convolutional neural networks for automatic vertebra segmentation
This is a Pytorch implementation of the paper [Iterative fully convolutional neural networks for automatic vertebra segmentation](https://openreview.net/forum?id=S1NnlZnjG) accepted in MIDL2018. This paper provides and automatic mechanism for precise vertebrae segmentation on CT images. I create this project for polishing up my knowledge of deep learning in medical image. There is an updated version [Iterative fully convolutional neural networks for automatic vertebra segmentation and identification](https://arxiv.org/abs/1804.04383)in 2019 with similiar structure. For the reason of computational cost, I decided to implement the early version in 2018.

## Model
This is the model illustration from the paper. This model contains a similiar shape like [3D U-Net](https://arxiv.org/abs/1606.06650) but with constant channel in every layer and a extend branch for classification propose. There are 2 inputs for this model, inclusing image patch and correspond instanace memory patches. Instance Memory is used to remind the model to segment the first 'unsegmented vertebrae' so as to make sure the vertebrae are segmented one by one.

![ad](https://github.com/leohsuofnthu/Pytorch-IterativeFCN/blob/master/imgs/model.png)

## Dataset and Pre-processsing

### 1. Dataset
I choose one of the dataset used in the paper, The spine segmentation challenge in CSI2014. The dataset can be obtain in the Dataset 2 posted on [SpineWeb](http://spineweb.digitalimaginggroup.ca/spineweb/index.php?n=Main.Datasets#Dataset_2.3A_Spine_and_Vertebrae_Segmentation)

### 2. Data preprocessing
The preprocessing steps of each CT images and corresponded masks(both train and test set) includes:
* **Resample the images and masks to isotropic (1mm * 1mm * 1mm)**
* **Calculate the weight penalty coefficient for each images via distance transform.**
* **Crop the images and masks to remove the vertebrae that not have labels in masks.**
* **Prepare the training patches, including "image patches", "instance memory patches", "mask patches" and "weight patches".**

### 3. Illustration of training patches.
A normal set of a training patches is showned as follows:



Since our model using slide window to segment the vertebrae, we need to teach it to produce empty prediction when their is no vertebrae in the image or all vertebrae are segmented and recorded in instnace memory:


## Training Detail
I apply the same setting as suggested in papers:
* **Batch-size = 1 due to GPU memory limitation.**
* **Adam with learning rate = 1e3**
* **Apply data augmentation via elastic deformation, gaussain blur, gaussian noise, random crop along z-axis**

I train this model on Google Colab, which has similiar CUDA Memory(12GB) with NVIDIA TITANX. Since we generate new patches every iteration, there is no concept of epoch here, and generated patches are always new to the model. I discard the part of validation. I use test data only for segmentation result for evaluation. The training is around 20000 iterations. I set the learning rate at 1e3 from 1 to 10000 iterations and 1e4 for 10001 to 20000, which is different from paper that use 1e3 all the time. 

## Segmentation Result
The following are some segmentation result from both train and test data.


### Visual Result

### Averge Dice Score and ASSD
**TO BE UPDATED**

## Environment Requirement
This project is developed under following environment:
```
python 3.6
pytorch 1.2.0
numpy 1.16.5
scipy 1.3.1
matplotlib 3.0.3
scikit-image 0.15.0
SimpleITK 1.2.3
medpy 0.25.1
```

## Authors

* **HSU, CHIH-CHAO** - *Professional Machine Learning Master Student at [MILA](https://mila.quebec/)* 

## Acknowlegements
