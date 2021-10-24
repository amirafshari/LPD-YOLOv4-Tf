# Tensorflow Implementation for YOLOv4

**1. I prefer to do these in command palette rather than jupyter, because we can see the results.**  
**2. It's [recommended](https://github.com/hunglc007/tensorflow-yolov4-tflite#traning-your-own-model) to train your custom detector on [darknet](https://github.com/AlexeyAB/darknet), rather than this implemntation, and then convert your weights and use this implemntation.**


```python
!git clone https://github.com/hunglc007/tensorflow-yolov4-tflite
```

## 1. Environment Setup

#### Conda Environment


```python
# Creat

# tf < 2.5 | 3.6 < python < 3.8
# tf > 2.5 | python > 3.9
# opencv 4.1.1.26 | python 3.7 <
!conda create --name envname python=3.7

# Activate
!activate envname
```

#### Requirements


```python
# tf > 2.5 both cpu and gpu use the same package

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

## 2. Tensorflow

#### Convert weights


```python
!python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4
```

#### Yolo Detector (COCO Dataset)


```python
!python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --image ./data/kite.jpg
```

#### Custom Detector (Your Dataset)

* 1. Create a custom.names file in data/classes and type your class (based on your weights and training)
* 2. Call the custom.names in config.py (change coco.names to custom.names)
* 3. Change the paths in detect.py
    


```python
!python detect.py --weights ./checkpoints/custom --size 416 --model yolov4 --image ./data/custom.jpg
```
![result](https://user-images.githubusercontent.com/17769927/134549864-703159d9-a8f2-41d0-b4ef-48e52bf770b9.jpg)

## 3. Tflite
#### Recommended for mobile and edge devices.

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

