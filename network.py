import os
import matplotlib.pyplot as plt
from datetime import datetime
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

class Network:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.target_size = (input_shape[0], input_shape[1])
        self.batch_size = 64
        self.epochs = 64
        self.model = None

    def createNetwork(self):
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu", input_shape=self.input_shape))
        self.model.add(Conv2D(128, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(rate = 0.25))

        self.model.add(Conv2D(64, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(128, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(rate = 0.25))

        self.model.add(Conv2D(64, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(128, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(rate = 0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(rate = 0.5))
        self.model.add(Dense(4, activation="softmax"))

    def compileNet(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def fitNetwork(self, train_dir_path, test_dir_path):
        datagen = ImageDataGenerator(rescale=1./255)
        train_generator  = datagen.flow_from_directory(train_dir_path, target_size=self.target_size, batch_size=self.batch_size)
        test_generator = datagen.flow_from_directory(test_dir_path, target_size=self.target_size, batch_size=self.batch_size)
        return self.model.fit_generator(train_generator, epochs=self.epochs, validation_data=test_generator)

    def evaluateNetwork(self, validation_dir):
        datagen = ImageDataGenerator(rescale=1./255)
        valid_generator = datagen.flow_from_directory(validation_dir, target_size=self.target_size, batch_size=self.batch_size)
        score = self.model.evaluate_generator(valid_generator)
        with open("output.txt","a") as f:
            f.write("score = "+str(score)+"\n")
            f.close()

    def plotHistory(self, history):
        plt.plot(history.history["acc"])
        plt.plot(history.history["val_acc"])
        plt.legend(['train', 'test'], loc='upper left')  
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig(str(datetime.today())+".png")

    def saveModel(self):
        self.model.save_weights(str(datetime.today())+".h5")

    def loadModel(self):
        self.model.load_weights("model.h5")

    def networkPredict(self, x):
        print(self.model.predict_classes(x))

if __name__ == "__main__":    
    net = Network((64,64,3))
    net.createNetwork()
    
    if not os.path.isfile("coucou.h5"):
        net.compileNet()
        history = net.fitNetwork("./data/train", "./data/test")
        net.plotHistory(history)
        net.saveModel()
    else:
        net.loadModel()
    net.evaluateNetwork("./data/validation")