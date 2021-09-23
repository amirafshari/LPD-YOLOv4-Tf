# Tensorflow Implementation

**1. I prefer to do these in command palette rather than jupyter, because we can see the results.**  
**2. It's [recommended](https://github.com/hunglc007/tensorflow-yolov4-tflite#traning-your-own-model) to train your custom detector on [darknet](https://github.com/AlexeyAB/darknet), rather than this implemntation, and then convert your weights and use this implemntation.**


```python
!git clone https://github.com/hunglc007/tensorflow-yolov4-tflite
```

## Environment Setup

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
!python -m ipykernel install --user --name=yolov4tf
```

Then choose yolov4tf from kernels in your notebook

## YOLO Setup

#### Convert weights


```python
!python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4
```

#### Demo


```python
!python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --image ./data/kite.jpg
```

**If you are using a detector with other classes than yolo's, you need to do the following:**

    1. Create a custom.names file in data/classes and type your class (based on your weights and training)
    2. Call the custom.names in config.py (change coco.names to custom.names)
    3. Change the paths in detect.py
    


```python
!python detect.py --weights ./checkpoints/custom --size 416 --model yolov4 --image ./data/custom.jpg
```
