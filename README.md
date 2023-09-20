# Vulnerable-Road-Users-Dataset
THis repository contains a deep learning approach based on YOLOv5x , YOLOv7x and YOLOv8 DeepSORT (DST) for detection and tracking of Vulnerable road users (VRUs) for faciliatating self driving vehicles and assiting human drivers for safety of VRUs. Results have been compared for benchmarking.
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

To run pretrained YOLOv5x models to generate object detection, follow these steps:

Clone this repository into YOLO_DST:

``git clone https://github.com/Faryalaurooj/Vulnerable-Road-Users-Dataset.git
  cd yolo_series_deepsort ``

