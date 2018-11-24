import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras import backend as K

def getFileContent(fname):
    list = []
    with open(fname,"r") as f:
        for line in f:list.append(line.replace("\n","").split(" "))
        f.close()
    return list

def updateNotes(list):
    for data in list:
        score = int(round(float(data[1])))
        if(score <= 2):score = 1
        elif(score == 3):score = 2
        elif(score > 3):score = 3
        data[1] = score
    return list

def moveToFolder(data, src_path, dst_path):
    i = 0
    for d in data:
        if(os.path.exists(src_path + d[0])):
            infos = d[0].split(".")
            os.system("mv " + src_path + d[0] + " " + dst_path + str(d[1]) + "/" + "{0:0=4d}".format(i) + "." + infos[1])
            i += 1

def getInputShape(img_width, img_height):
    input_shape = None
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    return input_shape

def listToLabels(list):
    for i in range(len(list)):list[i] = list[i][1]
    return list

def increaseDatasetSize(path, dst_path = None):
    if dst_path is None:dst_path = path
    datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
    gen = datagen.flow_from_directory(path,target_size=(350, 350), batch_size=batch_size, save_to_dir=dst_path, save_format='jpeg')#data must be in a subdir
    i = 0
    for batch in gen:
        i += 1
        if(i>100):break

def saveModel(netmodel):
    save = netmodel.to_json()
    with open("model.json", "w") as f:
        f.write(save)
        f.close()
    netmodel.save_weights("model.h5")

train_list = getFileContent("dataset/train_test_files/60training-40testing/train.txt")
train_list  = updateNotes(train_list)
nb_train_samples = len(train_list)
test_list = getFileContent("dataset/train_test_files/60training-40testing/test.txt")
test_list = updateNotes(test_list)
nb_validation_samples = len(test_list)
batch_size = 32
epochs = 8
print(nb_train_samples, nb_validation_samples)

if(not os.path.exists("dataset/Images/train")):
    os.mkdir("dataset/Images/train")
    os.mkdir("dataset/Images/train/1")# x < 33%
    os.mkdir("dataset/Images/train/2")# 33% < x < 66%
    os.mkdir("dataset/Images/train/3")# x > 66%
if(not os.path.exists("dataset/Images/test")):
    os.mkdir("dataset/Images/test")
    os.mkdir("dataset/Images/test/1")# x < 33%
    os.mkdir("dataset/Images/test/2")# 33% < x <66%
    os.mkdir("dataset/Images/test/3")# x > 66%

moveToFolder(train_list, "dataset/Images/", "dataset/Images/train/")
moveToFolder(test_list, "dataset/Images/", "dataset/Images/test/")

"""increaseDatasetSize("./dataset/Images/train/1")
increaseDatasetSize("./dataset/Images/train/2")
increaseDatasetSize("./dataset/Images/train/3")

increaseDatasetSize("./dataset/Images/test/1","./dataset/Images/validation/1")
increaseDatasetSize("./dataset/Images/test/2","./dataset/Images/validation/2")
increaseDatasetSize("./dataset/Images/test/3","./dataset/Images/validation/3")"""

train_label = np.array(listToLabels(train_list))
test_label = np.array(listToLabels(test_list))
print(type(train_label), type(test_label))

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory("./dataset/Images/train",target_size=(350, 350), batch_size=batch_size)#data must be in a subdir
test_generator = test_datagen.flow_from_directory("./dataset/Images/test",target_size=(350, 350), batch_size=batch_size)

input_shape = getInputShape(350,350)#dim of pic
print(input_shape)
"""
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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs, validation_data=test_generator, validation_steps=nb_validation_samples // batch_size)
saveModel(model)
"""