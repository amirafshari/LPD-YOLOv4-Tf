# YOLOv4 for Tensorflow and TFLite
I prefer to do these in cmd rather than jupyter, because we can see the results.


```python
!git clone https://github.com/hunglc007/tensorflow-yolov4-tflite
```

### Enviroment Setup

#### Conda Enviroment


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

#### Set the enviroment as jupyter kernel


```python
!pip install ipykernel
```


```python
!python -m ipykernel install --user --name=yolov4tf
```

### Darknet Setup

#### Convert weights


```python
!python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4
```

#### Demo


```python
!python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --image ./data/kite.jpg
```
