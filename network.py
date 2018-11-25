import os
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from takeapic import TakeAPic

class Network:
    def __init__(self):
        self.input_shape = (350, 350, 3)
        self.batch_size = 32
        self.epochs = 12
        self.nb_train_samples = 12928#nb of pic in train dir
        self.nb_validation_samples = 2206#nb of pic in test dir
        self.model = None

    def createNetwork(self):
        self.model = Sequential()
        self.model.add(Conv2D(8, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu", input_shape=self.input_shape))
        self.model.add(Conv2D(18, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(rate = 0.25))

        self.model.add(Conv2D(32, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(64, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(rate = 0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(rate = 0.5))
        self.model.add(Dense(3, activation="softmax"))

    def compileNet(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def fitNetwork(self, train_dir_path, test_dir_path):
        datagen = ImageDataGenerator(rescale=1./255)
        train_generator  = datagen.flow_from_directory(train_dir_path, target_size=(350,350), batch_size=self.batch_size)
        test_generator = datagen.flow_from_directory(test_dir_path, target_size=(350,350), batch_size=self.batch_size)
        self.model.fit_generator(train_generator, steps_per_epoch=self.nb_train_samples // self.batch_size, epochs=self.epochs, validation_data=test_generator, validation_steps=self.nb_validation_samples // self.batch_size)

    def saveModel(self):
        self.model.save_weights("model.h5")

    def loadModel(self):
        self.model.load_weights("model.h5")

    def networkPredict(self, x):
        print(self.model.predict_classes(x))

if __name__ == "__main__":    
    net = Network()
    net.createNetwork()
    
    if not os.path.isfile("model.h5"):
        net.compileNet()
        net.fitNetwork("./dataset/Images/train", "./dataset/Images/test")
    else:
        net.loadModel()
    
    cam = TakeAPic()
    pic = cam.getPicture()
    pic = cam.picFormatter(pic, (350,350), 3)
    net.networkPredict(pic)
