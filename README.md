# CiFAR_10
CiFAR_10 Image Classification using CNN model

## Description
CIFAR-10 is a dataset that consists of several images divided into the following 10 classes:
Airplanes<br>
Cars<br>
Birds<br>
Cats<br>
Deer<br>
Dogs<br>
Frogs<br>
Horses<br>
Ships<br>
Trucks<br><br>
The dataset stands for the Canadian Institute For Advanced Research (CIFAR). The dataset consists of 60,000 32x32 color images and 6,000 images of each class. Images have low resolution (32x32).<br>

Data Source: https://www.cs.toronto.edu/~kriz/cifar.html



## Libraries 
* Keras
* numpy
* pandas
* matplotlib

## Tool
* Google Colab

## Train and Test

```python
from keras.datasets import cifar10
(X_train,y_train),(X_test,y_test) = cifar10.load_data()
```

## Keras Layer

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
```

## Model Arhitecture
```python
model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = Input_shape))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.4))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(units = 512, activation = 'relu'))

model.add(Dense(units = 10, activation = 'softmax'))
```

## Model Compile
```python
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
```

## Training the model
```python
model.fit(X_train, y_train, batch_size = 32, epochs = 10, shuffle = True)
```
