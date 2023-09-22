# Vulnerable-Road-Users-Detection_Tracking

This repository contains a deep learning approach based on YOLOv5 , YOLOv7 and YOLOv8 DeepSORT (DST) for detection and tracking of Vulnerable road users (VRUs) for faciliatating self driving vehicles and assiting human drivers for safety of VRUs. Results have been compared for benchmarking. YOLOv5x , YOLOv7x and YOLOv8x models are trained on a NVIDIA GeForce RTX 2080 SUPER, 7982.312 MB after installing all dependencies. 

VRUs include all live actors which can be hit by moving vehicles on roads and are prone to much larger damage as compared to other cars. 
![0000074_10218_d_0000020](https://github.com/Faryalaurooj/Vulnerable-Road-Users-Detection-Tracking/assets/138756263/2de8e7e3-ae95-428c-a7fd-4754ff93ec64)

![val_batch0_pred](https://github.com/Faryalaurooj/Vulnerable-Road-Users-Dataset/assets/138756263/28ab2e28-649c-42ed-abee-41ee5c7f7c15)

YOLO is state-of-the-art object detection algorithm which is popularly used in computer cision for object detection tasks. Some of the reasons why YOLO is leading the competition and  include:

(a) Speed 
    
(b) Detection accuracy 
    
(c) Good generalization 
    
(d) Open-source
    
Therefore, i used various YOLO models in this project , compared their performances and in the end propose fastest, most accurate model for this application. 

### Requirements

In order to perform this project following are the requirements

Python-3.9.17 

torch-1.13.1+cu117 

CUDA:0 

NVIDIA GeForce RTX 2080 SUPER, 7982MiB OR HIGHER   (# you can alternatively do the project on google collab with small changes in commands)


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

To run pretrained YOLOv5x , YOLOv7x and YOLOv8x models to generate VRUs detections and tracking, follow these steps:

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
python train.py --data VRU.yaml --epochs 300 --img 640  --batch 4 --cfg  models /yolov5s.yaml --weights ''  --workers 8 --name yolov5  #for yolov5s

python train.py --model yolov5x.pt --data VRU_Dataset --epochs 300 --img 224 --batch 4                                               #for yolov5x

```
When i run this command, my system shows this:

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



### (4) Testing / Inference 
Copy the best.pt weights folder from train/runs folder into main yolov5 directory and then run following command :

```
python detect.py --weights best.pt --source /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test
```
This command will run inference on all the images placed inside  test/VRU_Dataset folder and save the results inside runs/detect/exp folder with all detected classes.
On my terminal when i ran this command , it appeared like this:

detect: weights=['best.pt'], source=/home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_csv=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
YOLOv5 🚀 v7.0-217-g8c45e51 Python-3.9.17 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 2080 SUPER, 7982MiB)

Fusing layers... 
YOLOv5x summary: 322 layers, 86186872 parameters, 0 gradients, 203.8 GFLOPs
image 1/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_00159_d_0000001.jpg: 384x640 6 peoples, 39.9ms
image 2/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_00611_d_0000002.jpg: 384x640 12 peoples, 1 tricycle, 36.4ms
image 3/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_01111_d_0000003.jpg: 384x640 2 peoples, 41.9ms
image 4/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_01275_d_0000004.jpg: 384x640 2 peoples, 3 bicycles, 41.7ms
image 5/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_01659_d_0000004.jpg: 384x640 (no detections), 41.8ms
image 6/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_02138_d_0000006.jpg: 384x640 (no detections), 41.8ms
image 7/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_02616_d_0000007.jpg: 384x640 (no detections), 39.8ms
image 8/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_03636_d_0000009.jpg: 384x640 (no detections), 30.1ms
image 9/1610 /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov5/VRU_Dataset/images/test/0000006_04050_d_0000010.jpg: 384x640 (no detections), 41.9ms

...... so on for all images

Speed: 1.3ms pre-process, 41.1ms inference, 7.9ms NMS per image at shape (1, 3, 640, 640) which means 24 FPS
Results saved to runs/detect/exp2


### (5) Validate 

YOLOv5x-cls accuracy on dataset:

```bash
bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate
```


## YOLOv7x

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

so we are using two GPUs for training device 0 and 1 , you can change as per your machine, batch size 4 is good for my GPU memory needs you can adjust as per your machine. Dataset is referred by the link VRU.yaml , cfg refers to the initial dataset of yolov7s model on which it was trained through yolovx.yaml file , hyp refers to model type , weights are pre-trained so '' is enough here.

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

autoanchor: Analyzing anchors... anchors/target = 1.79, Best Possible Recall (BPR) = 0.8191. Attempting to improve anchors, please wait...
autoanchor: WARNING: Extremely small objects found. 20031 of 124506 labels are < 3 pixels in size.
autoanchor: Running kmeans for 9 anchors on 124255 points...
autoanchor: thr=0.25: 0.9994 best possible recall, 5.39 anchors past thr
autoanchor: n=9, img_size=640, metric_all=0.356/0.750-mean/best, past_thr=0.499-mean: 3,5,  6,9,  8,15,  17,13,  11,22,  17,31,  33,23,  27,50,  63,53
autoanchor: Evolving anchors with Genetic Algorithm: fitness = 0.7905: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 528.86it/s]
autoanchor: thr=0.25: 0.9997 best possible recall, 6.96 anchors past thr
autoanchor: n=9, img_size=640, metric_all=0.436/0.790-mean/best, past_thr=0.515-mean: 2,4,  3,8,  5,7,  5,12,  8,10,  8,18,  16,13,  13,25,  28,35
autoanchor: New anchors saved to model. Update model *.yaml to use these anchors in the future.

Image sizes 640 train, 640 test
Using 4 dataloader workers
Logging results to runs/train/yolov7
Starting training for 300 epochs...

### (3) Testing / Inference 
Copy the last.pt weights folder from yolov7/train/runs folder into main yolov7 directory and then run following command
```
python detect.py --weights last.pt --source /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov7/VRU_Dataset/images/test
```
This command will run inference on all the images placed inside  test/VRU_Dataset folder and save the results inside runs/detect/exp folder with all detected classes.
On my terminal when i ran this command , it appeared like this:


img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, save_txt=False, save_conf=False, nosave=False, classes=None, agnostic_nms=False, augment=False, update=False, project='runs/detect', name='exp', exist_ok=False, no_trace=False)
YOLOR 🚀 v0.1-126-g84932d7 torch 1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 2080 SUPER, 7982.3125MB)
                                             CUDA:1 (NVIDIA GeForce RTX 2080 SUPER, 7981.0MB)

Fusing layers... 
IDetect.fuse
/home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
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

so when training staerted on my machine following information was displayed

Model summary: 365 layers, 68155497 parameters, 68155481 gradients, 258.1 GFLOPs

Transferred 589/595 items from pretrained weights
TensorBoard: Start with 'tensorboard --logdir runs/detect/train7', view at http://localhost:6006/
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt to 'yolov8n.pt'...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6.23M/6.23M [00:00<00:00, 6.82MB/s]
AMP: checks passed ✅
train: Scanning /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8/VRU_Dataset/labels/train.cache... 5772 images, 699 backgrounds, 0 corrupt: 100%|██████████| 6471/6471 [00:00<?, ?it/s]
val: Scanning /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8/VRU_Dataset/labels/validation.cache... 535 images, 13 backgrounds, 0 corrupt: 100%|██████████| 548/548 [00:00<?, ?it/s]
Plotting labels to runs/detect/train7/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 97 weight(decay=0.0), 104 weight(decay=0.0005), 103 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 4 dataloader workers
Logging results to runs/detect/train7
Starting training for 300 epochs...
.....
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

### (3) Testing
Once training is complete , we can test the trained YOLOv8x model for VRU detection from test folder of images from VRU_dataset folder or on any other images containing VRUs. First copy  the best.pt weights from runs/detect/train7 folder into main yolov8 directory and then run this command:

```
yolo task=detect mode=predict model=yolov8x.pt source=/home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8/VRU_Dataset/images/test

```
It will run prediction / testing on all the images inside VRU_Dataset/images/test folder. You can do testing on any other set of images as well by placing that folder inside yolov8 directory.
After completion of testing on images (1610 total) , it shows me these results :

Speed: 1.3ms preprocess, 20.4ms inference, 0.7ms postprocess per image at shape (1, 3, 480, 640)
Results saved to runs/detect/predict





