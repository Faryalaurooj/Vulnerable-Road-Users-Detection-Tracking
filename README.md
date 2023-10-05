# Vulnerable-Road-Users-Detection_Tracking

This repository contains a deep learning approach based on YOLOv5 , YOLOv7 and YOLOv8 for detection and tracking of Vulnerable road users (VRUs) for faciliatating self driving vehicles and assiting human drivers for safety of VRUs. Results have been compared for benchmarking. Models are trained on a NVIDIA GeForce RTX 2080 SUPER, 7982.312 MB after installing all dependencies. 

VRUs include all live actors which can be hit by moving vehicles on roads and are prone to much larger damage as compared to other cars. 
![0000074_10218_d_0000020](https://github.com/Faryalaurooj/Vulnerable-Road-Users-Detection-Tracking/assets/138756263/2de8e7e3-ae95-428c-a7fd-4754ff93ec64)

![val_batch0_pred](https://github.com/Faryalaurooj/Vulnerable-Road-Users-Dataset/assets/138756263/28ab2e28-649c-42ed-abee-41ee5c7f7c15)

YOLO is state-of-the-art object detection algorithm which is popularly used in computer cision for object detection tasks. Some of the reasons why YOLO is leading the competition and  include:

(a) Speed 
    
(b) Detection accuracy 
    
(c) Good generalization 
    
(d) Open-source
    
Therefore, i used various YOLO models in this project , compared their performances and in the end propose fastest, most accurate model for this application. At end of this project, i found that in terms of accuracy mAP YOLOv8x surpasses others and in terms of speed (FPS) YOLOv8s is best and YOLOv8x is second choice. Overall on GPU, YOLOv8 is best then comes YOLOv5 in terms of accuracy and speed both. YOLOv7 showed minimum mAP in this case study.

[detection_results_comparison.ods](https://github.com/Faryalaurooj/Vulnerable-Road-Users-Detection-Tracking/files/12791634/detection_results_comparison.ods)


### Requirements

In order to perform this project following are the requirements

(a) Python-3.9.17 

(b) torch-1.13.1+cu117 

(c) CUDA:0 

(d) NVIDIA GeForce RTX 2080 SUPER, 7982MiB OR HIGHER   (# you can alternatively do the project on google collab with small changes in commands)


### Dataset 

For this project VisDrone2019 dataset is used as a baseline and a new customized dataset VRU_Dataset has been evolved specifically for VRUs (vulnerable road users) which is in .txt format and can be used to train cutting-edge, state-of-the-art (SOTA) models like YOLO models for object detection tasks. This dataset is labelled for 03 classes of VRUs namely :

(a) Class 0- People(moving humans) and Pedestrians(standing humans) both in one class

(b) Class 1- Tricycles and Awning-tricycles

(c) Class 2- Bicycles

Dataset is divided into three folders Train , Test , Validation , size of each is :

(a) Train : 6,471 images , 1.5 GB , each Image sizes 640 x 640

(b) Test  : 1,610 images , 311 MB , each Image sizes 640 x 640

(c) Val   : 548 images   , 81  MB , each Image sizes 640 x 640

Upon publication, this customized VRU_Dataset will be made public.

## Quick Start

To run pretrained YOLOv5x and YOLOv8x models to generate VRUs detections and tracking, follow these steps:

 ### (1) Clone this repository into yolo_series_deepsort:
    
    git clone https://github.com/Faryalaurooj/Vulnerable-Road-Users-Dataset.git
    cd yolo_series_deepsort
    pip install -r requirements.txt  # install all requirements
    pip install ultralytics          # install for running YOLOv8

  By cloning this repo, you will get the trained weights of the three models. The results achieved for three models after training are given in this repo
  You can then use these trained models for testing and inference on images of VRUs avaiable in test folder 


### (2) Running inference 

If you want to run inference with YOLO5x, copy the best.pt weights from /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/runs/train/last folder in the main folder ie yolov5 and then run following command :
```
cd /yolo_series_deepsort_pytorch/yolov5
python detect.py --weights best.pt --source /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test  #yolov5 command


yolo task=detect mode=predict model=yolov8x.pt source=/home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8/VRU_Dataset/images/test   #yolov8 command
```
For detection with trained YOLOv7x model, best.pt are the best weight of yolov7x training so copy that best.pt file from yolov7/runs/training/last folder into main folder i.e yolov7 and then run above command. Same is true for YOLOv8x and YOLOv5 cases. Also do not forget to cd in yolov5 or yolov7 or yolov8 directories respectively before running inference command . To copy the complete path of any folder press ctrl+l+c .

## Training your own Model

If you do not want a quick start and want to train your own YOLO models on this new custom VRU_Dataset or any other new dataset for VRUs detection, follow these steps:

## YOLOv5x
### (1) Clone repository into YOLOv5 folder:
    
         
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    pip install -r requirements.txt  # install all requirements
    
### (2) Download and extract the VRU_Dataset folder 

The VRU_DAtaset folder comprises of images and annotations in .txt format. The data has to be copied inside YOLOv5 folder. VRU.yaml file is also to be copied inside YOLOv5 folder. i will provide the link to dataset very soon. 

### (3) Training

To train YOLOv5 model on a GPU as i did, launch the train.py script. It contains several options, i recommend this one:

```
python train.py --data VRU.yaml --epochs 300 --img 640  --batch 4 --cfg  ./models/yolov5s.yaml --weights ''  --workers 8 --name yolov5


```

Start training for YOLOv5s now
When i run this command, my system shows this:

YOLOv5s summary: 214 layers, 7027720 parameters, 7027720 gradients, 16.0 GFLOPs
hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=3
Plotting labels to runs/train/yolov5/labels.jpg... 
Image sizes 640 train, 640 val
Using 4 dataloader workers
Logging results to runs/train/yolov5
Starting training for 300 epochs...

 Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
0/299     0.533G      0.147    0.04498    0.02542         73        640: 1
          Class     Images  Instances          P          R      mAP50   
         all        548      16833         0.00032     0.0037   0.000165   3.98e-05   .... so on for 300 epochs

......

300 epochs completed in 16.620 hours.
Optimizer stripped from runs/train/yolov52/weights/last.pt, 14.3MB
Optimizer stripped from runs/train/yolov52/weights/best.pt, 14.3MB

Validating runs/train/yolov52/weights/best.pt...
Fusing layers... 
YOLOv5s summary: 157 layers, 7018216 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 69/69 [00:07<00:00,  9.19it/s]
                   all        548      16833      0.455      0.292      0.292      0.117
                people        548      13969      0.562      0.521      0.526       0.21
              tricycle        548       1577      0.492      0.217      0.235      0.103
               bicycle        548       1287      0.312      0.139      0.116     0.0366
Results saved to runs/train/yolov52
for yolov5x replace models with yolov5x.yaml only

### (4) Validate 

When the training is over, it is good practice to validate the new model on images it has not seen before. Therefore, when creating a dataset, we divide it into three parts, and one of them that we will use now as a validation dataset.


Running for YOLOv5s trained weights by copying yolov5s best.pt weights into yolov5 main directory:

```
python val.py --data VRU.yaml -weights best.pt --img 640 --conf 0.001 --iou 0.65 
```
 when i run this command following results appear

Fusing layers... 
YOLOv5s summary: 157 layers, 7018216 parameters, 0 gradients, 15.8 GFLOPs
val: Scanning /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/Visdrone2
                 Class     Images  Instances          P          R      mAP50   
                   all        548      16833      0.441      0.287      0.288      0.117
                people        548      13969      0.546      0.522      0.522      0.212
              tricycle        548       1577      0.485      0.212      0.228      0.102
               bicycle        548       1287      0.291      0.127      0.112     0.0365
Speed: 0.1ms pre-process, 5.8ms inference, 2.2ms NMS per image at shape (32, 3, 640, 640)
Results saved to runs/val/exp2

When we want to run validation with yolov5x , we will copy those best.pt weights in main yolov5 directory before running above command , as i did and got following results:

Fusing layers... 
YOLOv5x summary: 322 layers, 86186872 parameters, 0 gradients, 203.8 GFLOPs
val: Scanning /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/Visdrone2
                 Class     Images  Instances          P          R      mAP50   
                   all        548      16833      0.554      0.335       0.35      0.154
                people        548      13969      0.622      0.558       0.58      0.252
              tricycle        548       1577      0.632      0.274      0.299      0.147
               bicycle        548       1287      0.406      0.172      0.172     0.0638
Speed: 0.1ms pre-process, 17.0ms inference, 1.5ms NMS per image at shape (32, 3, 640, 640)
Results saved to runs/val/exp3


### (5) Testing / Inference 
Copy the best.pt weights folder from train/runs folder into main yolov5 directory and then run following command :

```
python detect.py --weights best.pt --source ./VRU_Dataset/images/test
```
This command will run inference on all the images placed inside  test/VRU_Dataset folder and save the results inside runs/detect/exp folder with all detected classes.

First detection with YOLOv5x model :
On my terminal when i ran this command , it appeared like this:

detect: weights=['best.pt'], source=/home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_csv=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
YOLOv5 🚀 v7.0-217-g8c45e51 Python-3.9.17 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 2080 SUPER, 7982MiB)

Fusing layers... 
YOLOv5x summary: 322 layers, 86186872 parameters, 0 gradients, 203.8 GFLOPs
image 1/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_00159_d_0000001.jpg: 384x640 6 peoples, 39.9ms
image 2/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_00611_d_0000002.jpg: 384x640 12 peoples, 1 tricycle, 36.4ms
image 3/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_01111_d_0000003.jpg: 384x640 2 peoples, 41.9ms
image 4/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_01275_d_0000004.jpg: 384x640 2 peoples, 3 bicycles, 41.7ms

...... so on for all images

Speed: 1.3ms pre-process, 41.1ms inference, 7.9ms NMS per image at shape (1, 3, 640, 640) which means 24 FPS
Results saved to runs/detect/exp2

then i run the same command by using best.pt weights of YOLOv5s , copying the best.pt folder into main yolov5 directory and running above command. i get this ; 

using layers... 
YOLOv5s summary: 157 layers, 7018216 parameters, 0 gradients, 15.8 GFLOPs
image 1/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_00159_d_0000001.jpg: 384x640 5 peoples, 8.5ms
......
image 1610/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/9999996_00000_d_0000029.jpg: 480x640 7 peoples, 5.6ms

Speed: 0.2ms pre-process, 5.7ms inference, 0.5ms NMS per image at shape (1, 3, 640, 640) , 175 FPS
Results saved to runs/detect/exp5

so we can observe pre-process speed and inference time is improved in case of yolov5s as compared to yolov5x but overall P curve R curve are infererioir 



## YOLOv7

In order to train YOLOv7x follow following instructions

### (1) Download Repository 

```
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip install -r requirements
```

### (2) Train

In order to start training , first copy the VRU_Dataset folder and VRU.yaml file inside the yolov7 directory. Delete catch files from the labels folder inside dataset folder. Correct the path inside .yaml file by copying path of dataset and replacing it inside .yaml file. 
Now run the command

```
python train.py --workers 8 --device 0,1 --batch-size 4 --data VRU.yaml --img 640 640 --cfg cfg/training/yolov7x.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

```

so we are using two GPUs for training device 0 and 1 , you can change as per your machine, batch size 4 is good for my GPU memory needs, you can adjust as per your machine. Dataset is referred by the link VRU.yaml , cfg refers to the initial dataset of yolov7X model on which it was trained through yolov7x.yaml file , hyp refers to model type , weights are pre-trained so '' is enough here IT MEANS IT WILL AUTOMATICALLY DOWNLOAD THE WEIGHTS FROM INTERNET AND THEN IT WILL START TRAINING ITSELF AND ON COMPLETION OF TRAINING IT WILL GENERATE ITS OWN BEST.PT FILE WHICH WILL BE NEW WEIGHTS.

If some interruption occurs and training stops, resume training by referring to last best possible weights by using following command
```
python train.py --workers 8 --device 0,1 --batch-size 4 --data VRU.yaml --img 640 640 --cfg cfg/training/yolov7x.yaml --weights 'runs/train/yolov710/weights/last.pt' --hyp data/hyp.scratch.p5.yaml
```

Here you can see that rest everything is same except for weight file where we will refer to the location of folder in which last weights were saved when training cycle disrupts. 
So on my machine , when training starts it gives following details 

Model Summary: 467 layers, 70828568 parameters, 70828568 gradients, 188.9 GFLOPS

Scaled weight_decay = 0.0005
Optimizer groups: 108 .bias, 108 conv.weight, 111 other

train: Scanning 'VRU_Dataset/labels/train.cache' images and labels... 5772 found, 699 missing, 0 empty, 2 corrupted: 100%|███████████████████████████████████████████| 6471/6471 [00:00<?, ?it/s]

val: Scanning 'VRU_Dataset/labels/validation.cache' images and labels... 535 found, 13 missing, 0 empty, 0 corrupted: 100%|████████████████████████████████████████████| 548/548 [00:00<?, ?it/s]

Logging results to runs/train/yolov711

Starting training for 300 epochs...


### (3) Testing / Inference 
Copy the best.pt weights folder from yolov7/runs/train/yolov711 folder into main yolov7 directory and then run following command
```
python detect.py --weights best.pt --source /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov7/VRU_Dataset/images/test
```
This command will run inference on all the images placed inside  test/VRU_Dataset folder and save the results inside runs/detect/exp folder with all detected classes.
On my terminal when i ran this command , it appeared like this:


img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, save_txt=False, save_conf=False, nosave=False, classes=None, agnostic_nms=False, augment=False, update=False, project='runs/detect', name='exp', exist_ok=False, no_trace=False)

YOLOR 🚀 v0.1-126-g84932d7 torch 1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 2080 SUPER, 7982.3125MB)

                                             CUDA:1 (NVIDIA GeForce RTX 2080 SUPER, 7981.0MB)

Fusing layers... 
 
Model Summary: 362 layers, 70795920 parameters, 0 gradients, 188.0 GFLOPS
 Convert model to Traced-model... 
 traced_script_module saved! 
 model is traced! 

Done. (21.4ms) Inference, (4.5ms) NMS
 The image with the result is saved in: runs/detect/exp/0000006_00159_d_0000001.jpg
Done. (21.3ms) Inference, (4.6ms) NMS
 The image with the result is saved in: runs/detect/exp/0000006_00611_d_0000002.jpg
Done. (21.5ms) Inference, (4.5ms) NMS
 The image with the result is saved in: runs/detect/exp/0000006_01111_d_0000003.jpg
Done. (19.4ms) Inference, (4.5ms) NMS
...... so on for 
Done. (91.956s)

in another attempt 

Fusing layers... 
Model Summary: 362 layers, 70795920 parameters, 0 gradients, 188.0 GFLOPS
Done. (11.5ms) Inference, (0.2ms) NMS
The image with the result is saved in: runs/detect/exp3/0000006_00159_d_0000001.jpg
......
Done. (13.0ms) Inference, (0.1ms) NMS
The image with the result is saved in: runs/detect/exp3/9999996_00000_d_0000029.jpg
Done. (68.661s)

## YOLOv8x

To train YOLOV8x model follow following instructions

### (1) Download YOLOv8 model from repository

```
git clone https://github.com/ultralytics/ultralytics.git
cd yolov8
pip install -r requirements
pip install ultralytics
```
Now copy the VRU_DATASET folder inside yolov8 folder and delete cache  files from the labels folder. Also copy the VRU.yaml file inside yolov8 directory. THe .yaml file should appear like this 


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
so we have defined model as ##YOLOv8x , custom data whose path is defined inside VRU.yaml file , if you want to use some other data you can change this file and give that path in .yaml file , we have defined batch size as 4 to meet GPU memory requirements.
During training, we can see the precision, recall and mAP values on scree with help of tensorboard with following command

```
tensorboard --logdir runs/detect/train7
```
and see by ctrl+click  at http://localhost:6006/

so when training staerted on my machine following information was displayed:

Model summary: 365 layers, 68155497 parameters, 68155481 gradients, 258.1 GFLOPs
Transferred 589/595 items from pretrained weights
TensorBoard: Start with 'tensorboard --logdir runs/detect/train7', view at http://localhost:6006/
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt to 'yolov8n.pt'...
.....

  Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    176/300      5.64G      1.564     0.8106       0.85        162        640: 100%|██████████| 1618/1618 [05:16<00:00,  5.11it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 69/69 [00:06<00:00, 11.35it/s]
                   all        548      16833      0.576      0.389      0.417       0.21

My system Stopped training early as no improvement observed in last 50 epochs. Best results observed at epoch 126, best model saved as best.pt.
To update EarlyStopping(patience=50) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

176 epochs completed in 15.831 hours.
Optimizer stripped from runs/detect/train7/weights/last.pt, 136.7MB
Optimizer stripped from runs/detect/train7/weights/best.pt, 136.7MB

Validating runs/detect/train7/weights/best.pt...
Ultralytics YOLOv8.0.176 🚀 Python-3.9.17 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 2080 SUPER, 7982MiB)
Model summary (fused): 268 layers, 68126457 parameters, 0 gradients, 257.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   1%|▏         | 1/69 [00:00<00:10,  6.79it/s]WARNING ⚠️ NMS time limit 0.900s exceeded
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 69/69 [00:09<00:00,  7.07it/s]
                   all        548      16833      0.563      0.385       0.41      0.207
                people        548      13969      0.701      0.534      0.589      0.272
              tricycle        548       1577      0.604      0.388      0.428      0.249
               bicycle        548       1287      0.385      0.234      0.213     0.0985
Speed: 0.1ms preprocess, 9.9ms inference = 101 FPS , 0.0ms loss, 4.2ms postprocess per image
Results saved to runs/detect/train7

Lets run the train command for YOLOv8s model also, by only changing yolov8x.pt with yolov8s.pt. When we run this command, first of all system will automatically download the weights from internet and then displays this:

Model summary: 225 layers, 11136761 parameters, 11136745 gradients, 28.7 GFLOPs
Transferred 349/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
300 epochs completed in 8.675 hours.

Validating runs/detect/train8/weights/best.pt...
Ultralytics YOLOv8.0.176 🚀 Python-3.9.17 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 2080 SUPER, 7982MiB)
Model summary (fused): 168 layers, 11126745 parameters, 0 gradients, 28.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 69/69 [00:06<00:00, 10.90it/s]
                   all        548      16833      0.508      0.335      0.357      0.168
                people        548      13969      0.664        0.5      0.556      0.244
              tricycle        548       1577      0.553      0.328      0.362      0.195
               bicycle        548       1287      0.309      0.177      0.155     0.0635
Speed: 0.1ms preprocess, 1.6ms inference, 0.0ms loss, 0.5ms postprocess per image, 625 FPS

### (3) Validation

When the training is over, it is good practice to validate the new model on images it has not seen before.In order to perfrom validation with YOLOv8 , copy the best.pt weights in main yolov8 directory and then perfrom this command: 
```
yolo task=detect mode=val model=best.pt data=VRU.yaml
```
when i run this command i get this result

Ultralytics YOLOv8.0.146 🚀 Python-3.9.17 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 2080 SUPER, 7982MiB)
Model summary (fused): 268 layers, 68126457 parameters, 0 gradients, 257.4 GFLOPs
val: Scanning /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8/VRU_Dataset/labels/validation... 535 images, 13 backgrounds, 0 corrupt: 100%|██████████| 548/548 [00:00<00:00, 714.90it/s]
val: New cache created: /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8/VRU_Dataset/labels/validation.cache
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 35/35 [00:15<00:00,  2.26it/s]
                   all        548      16833      0.573      0.396      0.426      0.217
                people        548      13969      0.704      0.551      0.608      0.286
              tricycle        548       1577      0.597      0.403      0.442      0.258
               bicycle        548       1287      0.419      0.235      0.228      0.106
Speed: 0.3ms preprocess, 20.8ms inference, 0.0ms loss, 1.3ms postprocess per image
Results saved to runs/detect/val


### (4) Testing
Once training and validation is complete , we can test the trained YOLOv8x model for VRU detection from test folder of images from VRU_dataset folder or on any other images containing VRUs. First copy  the best.pt weights from runs/detect/train7 folder into main yolov8 directory and then run this command:

```
yolo task=detect mode=predict model=best.pt source=/home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8/VRU_Dataset/images/test

```
It will run prediction / testing on all the images inside VRU_Dataset/images/test folder. You can do testing on any other set of images as well by placing that folder inside yolov8 directory.
After completion of testing on images (1610 total) , it shows me these results :

Ultralytics YOLOv8.0.176 🚀 Python-3.9.17 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 2080 SUPER, 7982MiB)
Model summary (fused): 268 layers, 68126457 parameters, 0 gradients, 257.4 GFLOPs
image 1/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8/VRU_Dataset/images/test/0000006_00159_d_0000001.jpg: 384x640 4 peoples, 22.2ms
.......
image 1610/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8/VRU_Dataset/images/test/9999996_00000_d_0000029.jpg: 480x640 7 peoples, 24.5ms
Speed: 1.3ms preprocess, 20.1ms inference, 0.7ms postprocess per image at shape (1, 3, 480, 640)
Results saved to runs/detect/predict2





