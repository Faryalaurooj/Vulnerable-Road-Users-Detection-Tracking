# Vulnerable-Road-Users-Dataset

This repository contains a deep learning approach based on YOLOv5 , YOLOv7 and YOLOv8 DeepSORT (DST) for detection and tracking of Vulnerable road users (VRUs) for faciliatating self driving vehicles and assiting human drivers for safety of VRUs. Results have been compared for benchmarking. YOLOv5x , YOLOv7x and YOLOv8x models are trained on a NVIDIA GeForce RTX 2080 SUPER, 7982.312 MB after installing all dependencies.

VRUs include all live actors which can be hit by moving vehicles on roads and are prone to much larger damage as compared to other cars. 
![train_batch0](https://github.com/Faryalaurooj/Vulnerable-Road-Users-Dataset/assets/138756263/07644322-113c-429b-a4b7-3f0cf9541dbb)
![val_batch0_pred](https://github.com/Faryalaurooj/Vulnerable-Road-Users-Dataset/assets/138756263/28ab2e28-649c-42ed-abee-41ee5c7f7c15)


## Requirements

## Dataset 

For this project VisDrone2019 dataset is used as a baseline and a new customized dataset has been evolved specifically for VRUs which is in .txt format and can be used to train cutting-edge, state-of-the-art (SOTA) models like YOLO models for object detection tasks.

This dataset is labelled for 03 classes of VRUs namely :

class 0-people(moving humans) and pedestrians(standing humans) both in one class
class 1-tricycles and awning-tricycles
class 2- Bicycles

Upon publication, this customized dataset will be made public.

## Quick Start

(1) To run pretrained YOLOv5x , YOLOv7 and YOLOv8 models to generate VRUs detection, follow these steps:

  Clone this repository into yolo_series_deepsort:


    
    git clone https://github.com/Faryalaurooj/Vulnerable-Road-Users-Dataset.git
    cd yolo_series_deepsort
    pip install -r requirements.txt  # install all requirements
    

  By cloning this repo, you will get the trained weights of the three models. You can then use these trained models for testing and inference on images of VRUs avaiable in test folder 


(2) Running inference 

```
```

# Training your own Model

To train your own YOLO models on this new custom dataset or any other dataset for VRU detection, follow these steps:
Preparation

## Requirements

(1) If you have not done so already, clone this repository into yolo_series_deepsort:
    
    
    
    
    git clone https://github.com/Faryalaurooj/Vulnerable-Road-Users-Dataset.git
    cd yolo_series_deepsort
    pip install -r requirements.txt  # install all requirements
    
(2) Download and extract the Visdrone2019_Dataset folder which comprises of images and annotations in .txt format: The data has to be copied inside YOLOv5 folder. visdrone.yaml file is also to be copied inside YOLOv5 folder.

i will provide the link very soon

(3) Training

To train YOLOv5x model on a GPU as i did, launch the train.py script. It contains several options, i recommend this one

```
# Single-GPU
python train.py --model yolov5s-cls.pt --data cifar100 --epochs 5 --img 224 --batch 128

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3
```

  



