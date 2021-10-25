---
slug: license-plate-detector
title: Automatic License Plate Detector
authors:
  name: Amir Afshari
  title: Machine Learning Engineer
  url: https://github.com/amirafshari
  image_url: https://avatars.githubusercontent.com/u/17769927?s=400&u=d630f608970a53d00295f2e87e88526b41b7d0b1&v=4
tags: [Object Detection, Computer Vision, Deep Learning]
---


## Exploratory Data Analysis
How our data looks like?  
Annotations format (YOLO Format): [class, x_center, y_center, obj_width, obj_height]  

<!--truncate-->


### Distributions

![1](https://user-images.githubusercontent.com/17769927/134396237-178893ef-18f1-4df6-b3ea-fe4b235e3a27.png)
     


They make sense for number plate images  
*   x values are well distributed, which means the cameraman did a good job :D
*   y values are well distributed as well, but, most of the objects are on top of our images.
*   both height and width make sense, because our object is licence plate and they all have almost similiar sizes.


### X vs Y & Height vs Width

![2](https://user-images.githubusercontent.com/17769927/134396293-df5113b7-9237-4dfc-81ac-1a2bf6187826.png)

*   As mentioned above, there is a lack in our dataset in buttom-half part of xy plane.
*   As we can see, the center of our x axis is dense, it's beacuse humans put the object in the center of the camera.



## Tensorflow Implementation for YOLOv4
**It's [recommended](https://github.com/hunglc007/tensorflow-yolov4-tflite#traning-your-own-model) to train your custom detector on [darknet](https://amirafshari.com/blog/train-custom-object-detector), rather than this implemntation, and then convert your weights and use this implemntation.**


```python
!git clone https://github.com/hunglc007/tensorflow-yolov4-tflite
```

### Environment Setup

#### Conda Environment


```python
# Create
# tf < 2.5 | python = 3.7
# tf > 2.5 | python > 3.9
!conda create --name envname python=3.7

# Activate
!activate envname
```

#### Requirements


```python
# in tf > 2.5 both cpu and gpu use the same package

# GPU
!pip install -r requirements-gpu.txt

# CPU
!pip install -r requirements.txt
```

#### Check


```python
!conda list # installed packages in current env
!python --version
```

#### Set the environment as jupyter kernel


```python
!pip install ipykernel
```


```python
!python -m ipykernel install --user --name=envname
```

Then choose yolov4tf from kernels in your notebook

### Tensorflow

#### Convert weights


```python
!python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4
```

#### COCO Dataset

```python
!python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --image ./data/kite.jpg
```

#### Custom Dataset

*   Create a custom.names file in data/classes and type your class (based on your weights and training)
*   Call the custom.names in config.py (change coco.names to custom.names)
*   Change the paths in detect.py
    


```python
!python detect.py --weights ./checkpoints/custom --size 416 --model yolov4 --image ./data/custom.jpg
```
![result](https://user-images.githubusercontent.com/17769927/134549864-703159d9-a8f2-41d0-b4ef-48e52bf770b9.jpg)

### 3. Tflite
Recommended for mobile and edge devices.

#### Convert

```python
# Save tf model for tflite converting
!python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 --framework tflite

# YOLOv4
!python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416.tflite
```

#### Demo

```python
!python detect.py --weights ./checkpoints/yolov4-416.tflite --size 416 --model yolov4 --image ./data/kite.jpg --framework tflite
```
![result-9](https://user-images.githubusercontent.com/17769927/134549834-da73a045-05c9-4d6c-8772-90c4dca67cf7.jpg)


## Metrics

*   Precision: 91 %
*   Average Precision: 89.80 %
*   Recall: 86 %
*   F1-score: 88 %
*   Average IoU: 74.06 %
*   mAP@0.5: 89.80 %
*   Confusion Matrix:
    *   TP = 439
    *   FP = 45
    *   FN = 73
    *   unique_truth_count (TP+FN) = 512
    *   detections_count = 805
