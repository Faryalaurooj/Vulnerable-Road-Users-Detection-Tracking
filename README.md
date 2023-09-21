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

## Training your own Model

To train your own YOLO models on this new custom dataset e.g VRU_dataset or any other new dataset for VRU detection, follow these steps:

Preparation


## YOLOv5x
### (1) If you have not done so already, clone this repository into yolo_series_deepsort:
    
    
    
    
    git clone https://github.com/Faryalaurooj/Vulnerable-Road-Users-Dataset.git
    cd yolo_series_deepsort
    pip install -r requirements.txt  # install all requirements
    
### (2) Download and extract the VRU_Dataset folder which comprises of images and annotations in .txt format: The data has to be copied inside YOLOv5 folder. VRU.yaml file is also to be copied inside YOLOv5 folder.

i will provide the link very soon

### (3) Training

To train YOLOv5x model on a GPU as i did, launch the train.py script. It contains several options, i recommend this one

```
# Single-GPU
python train.py --model yolov5x-cls.pt --data cifar100 --epochs 5 --img 224 --batch 128

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3
```
### (4) Val

Validate YOLOv5x-cls accuracy on dataset:

```bash
bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate
```

### (5) Predict

Use pretrained YOLOv5x-cls.pt to predict bus.jpg:

```bash
python classify/predict.py --weights yolov5s-cls.pt --source data/images/bus.jpg
```
  
## YOLOv8x

To train YOLOV8x model follow following instructions

### (1) Download YOLOv8 model from repository

```
git clone https://github.com/ultralytics/ultralytics.git
cd yolov8
pip install -r requirements
pip install ultralytics
```
Now copy the VRU_DATASET folder inside yolov8 folder and delete catch files from the labels folder. Also copy the VRU.yaml file inside yolov8 directory. THe .yaml file should appear like this 


train: VRU_Dataset/images/train  # train images (relative to 'path')  6471 images
val: VRU_Dataset/images/validation  # val images (relative to 'path')  548 images
test: VRU_Dataset/images/test  # test images (optional)  1610 images
nc: 3
names: ["people", "tricycle", "bicycle"]

Now make some changes in downloads/.config/Ultralytics/settings.yaml file. It should look like this

settings_version: 0.0.4
datasets_dir: /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8
weights_dir: weights
runs_dir: runs
uuid: 1dae3a9401e5298a444fec2d464f0a2f1640150617e52ef6faf3b202ebf2672e
sync: true
api_key: ''
clearml: true
comet: true
dvc: true
hub: true
mlflow: true
neptune: true
raytune: true
tensorboard: true
wandb: true




### (2) Train 

Now we are ready to run training command

```
yolo task=detect mode=train model=yolov8x.pt data=VRU.yaml epochs=300 imgsz=640 batch=4
```
so we have defined model as YOLOv8x , custom data whose path is defined inside VRU.yaml file , if you want to use some other data you can change this file and give that path in .yaml file , we have defined batch size as 4 to meet GPU memory requirements.
During training, we can see the precision, recall and mAP values on scree with help of tensorboard with following command

```
tensorboard --logdir runs/detect/train7
```
and see by ctrl+click  at http://localhost:6006/


### (3) Testing
Once training is complete , we can test the trained YOLOv8x model for VRU detection from test folder of images from VRU_dataset folder or on any other images containing VRUs



