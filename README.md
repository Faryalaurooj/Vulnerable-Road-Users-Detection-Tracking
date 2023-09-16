# Vulnerable-Road-Users-Project
Vulnerable road users include all live actors which can be hit by moving vehicles on roads. In this project various versions of YOLO detection model are trained on a specialized dataset to formulate a system that that automatically effectively detect and track the vulnerable users. THis system can be deployed in self driving cars and it can also be used in man driven cars for assistance in protecting vulnerable road users from potential accidents.

THere are following steps involved in this project:

#1 Data set preparation
The dataset is prepared for this project by using VisDRone2019 dataset. That data was initially labelled for 09 classes and was available in voc format which is not usable for training Yolo model. I re-labelled the data with the help of a script for 03 classes namely people , tricycles and bicycles. VisDrone dataset is downloaded and reformatted and re annotated for developing this specific dataset. It is now in .txt format, thus it can be used to train all types of YOLO models for object detection tasks. This re-labelling effort not only met the project requirements but also increased the performance of detection model.

#2 Training YOLOV5x
IN next step, Yolov5x is trained on new 03 class dataset for 300 epochs on a NVIDIA GFORCE RTX-2080 SUPER, 7982.3125MB  machine which took 48 hours
THe results of this training are shared in this github repository

Summary
mAP@50 = 0.353 ALL CLASSES
mAP@50 = 0.583 People
mAP@50 = 0.304 Tricycle
mAP@50 = 0.172 bicycle




