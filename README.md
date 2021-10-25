# Exploratory Data Analysis
How our data looks like?  
1. Annotations format (YOLO Format): [class, x_center, y_center, obj_width, obj_height]  
2. Create a DataFrame from annotations to visualize our objects.  



```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv('./data/eda.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>width</th>
      <th>height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.480508</td>
      <td>0.898340</td>
      <td>0.041250</td>
      <td>0.030019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.990586</td>
      <td>0.770524</td>
      <td>0.013750</td>
      <td>0.018762</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.356875</td>
      <td>0.666562</td>
      <td>0.093750</td>
      <td>0.036667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.877812</td>
      <td>0.514688</td>
      <td>0.048125</td>
      <td>0.027500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.284570</td>
      <td>0.873530</td>
      <td>0.057500</td>
      <td>0.020599</td>
    </tr>
  </tbody>
</table>
</div>



## Distributions


```python
plt.figure(figsize=(13,8))
bins=40

plt.subplot(2,2,1)
sns.histplot(data=df, x='x', bins=bins)

plt.subplot(2,2,2)
sns.histplot(data=df, x='y', bins=bins)

plt.subplot(2,2,3)
sns.histplot(data=df, x='width', bins=bins)

plt.subplot(2,2,4)
sns.histplot(data=df, x='height', bins=bins)
```




    <AxesSubplot:xlabel='height', ylabel='Count'>




![1](https://user-images.githubusercontent.com/17769927/134396237-178893ef-18f1-4df6-b3ea-fe4b235e3a27.png)
     


They make sense for number plate images  
*   x values are well distributed, which means the cameraman did a good job :D
*   y values are well distributed as well, but, most of the objects are on top of our images.
*   both height and width make sense, because our object is licence plate and they all have almost similiar sizes.


## X vs Y | Height vs Width


```python
plt.figure(figsize=(13,5))

plt.subplot(1,2,1)
sns.scatterplot(data=df, x='x', y='y', alpha=.4)

plt.subplot(1,2,2)
sns.scatterplot(data=df, x='width', y='height', alpha=.4)
```




    <AxesSubplot:xlabel='width', ylabel='height'>



![2](https://user-images.githubusercontent.com/17769927/134396293-df5113b7-9237-4dfc-81ac-1a2bf6187826.png)

1.   As mentioned above, there is a lack in our dataset in buttom-half part of xy plane.
2.   As we can see, the center of our x axis is dense, it's beacuse humans put the object in the center of the camera.



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

