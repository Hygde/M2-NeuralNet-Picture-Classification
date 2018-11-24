import cv2
import numpy as np
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential

input_shape = (350,350,3)#dim of pic

model = Sequential()
model.add(Conv2D(8, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu", input_shape=input_shape))
model.add(Conv2D(18, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate = 0.25))

model.add(Conv2D(32, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate = 0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(rate = 0.5))
model.add(Dense(3, activation="softmax"))

model.load_weights("model.h5")

cam = cv2.VideoCapture(0)
continuer = True
frame = None
while continuer:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (350, 350))
    cv2.imshow("pic", frame)
    if cv2.waitKey(33) == ord(' '):continuer = False
cam.release()
frame = cv2.resize(frame, (350,350))
frame = np.reshape(frame, [1,350,350,3])

print(model.predict_classes(frame))
