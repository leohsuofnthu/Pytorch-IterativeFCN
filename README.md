# Iterative fully convolutional neural networks for automatic vertebra segmentation
This is a Pytorch implementation of the paper [Iterative fully convolutional neural networks for automatic vertebra segmentation](https://openreview.net/forum?id=S1NnlZnjG) accepted in MIDL2018. This paper provides and automatic mechanism for precise vertebrae segmentation on CT images. I create this project for polishing up my knowledge of deep learning in medical image. There is an updated version [Iterative fully convolutional neural networks for automatic vertebra segmentation and identification](https://arxiv.org/abs/1804.04383)in 2019 with similiar structure. For the reason of computational cost, I decided to implement the early version in 2018.

## Model
This is the model illustration from paper. This model contains a similiar shape like [3D U-Net](https://arxiv.org/abs/1606.06650) but with 
constant channel in every layer and a extend branch for classification propose. There are 2 inputs for this model, inclusing image patch and correspond instanace memory patches. Instance Memory is used to remind the model to segment the first 'unsegmented vertebrae' so as to make sure the vertebrae are segmented one by one.

![ad](https://github.com/leohsuofnthu/Pytorch-IterativeFCN/blob/master/imgs/model.png)

## Data Pre-processsing and Preparation

## Training Detail

## Segmentation Result

## Environment Requirement

## Acknowlegements
