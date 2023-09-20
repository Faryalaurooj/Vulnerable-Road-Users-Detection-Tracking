# Vulnerable-Road-Users-Dataset
THis repository contains a deep learning approach for detection and tracking of Vulnerable road users (VRUs) for faciliatating self driving vehicles and assiting human drivers for safety of VRUs. 
VRUs include all live actors which can be hit by moving vehicles on roads and are prone to much larger damage as compared to other cars. 
![train_batch0](https://github.com/Faryalaurooj/Vulnerable-Road-Users-Dataset/assets/138756263/07644322-113c-429b-a4b7-3f0cf9541dbb)
![val_batch0_pred](https://github.com/Faryalaurooj/Vulnerable-Road-Users-Dataset/assets/138756263/28ab2e28-649c-42ed-abee-41ee5c7f7c15)


## Requirements

## Dataset 

VisDrone2019 dataset is downloaded which originally is in voc format and is labelled for 09 classes including pedestrians, people, cars, trucks, bicycles, etc. THis dataset has been reformatted and re-annotated for developing a specific dataset for this use case. The prepared data is in .txt format tand it can be used to train cutting-edge, state-of-the-art (SOTA) models like YOLO models for object detection tasks.

This dataset is labelled for 03 classes namely :
class 0-people(moving) and pedestrians(standing) both
class 1-tricycles and awning-tricycles
class 2- Bicycles

Upon publication, dataset will be made public 
