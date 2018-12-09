import os
import sys
from math import pow
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

class ArchProcessing:

    def __init__(self):
        pass

    def extractZip(self,fname):
        os.system("unzip -q "+fname)

    def createClassDir(self, nclass):
        try:
            os.mkdir("data")
            os.mkdir("data/train")
            os.mkdir("data/test")
            for iclass in range(nclass):
                os.mkdir("data/train/"+str(iclass))
                os.mkdir("data/train/"+str(iclass)+"/originals")
                os.mkdir("data/test/"+str(iclass))
                os.mkdir("data/test/"+str(iclass)+"/originals")
        except:
            print("Directories already exist")

    def getFileContent(self, fname):
        li = []
        with open(fname,"r") as f:
            for line in f:li.append(line.replace("\n","").split(" "))
            f.close()
        return li

    def plotData(self, li):
        tmp = []
        for data in li: tmp.append(float(data[1]))
        tmp.sort()
        plt.plot([i for i in range(len(tmp))], tmp)
        plt.show()

    def updateNotes(self, li):
        for data in li:
            score = pow(float(data[1])-1.0, 2)
            #todo: automate the following part
            if(score < 2.25):score = 0
            elif(score >= 2.25 and score < 3.3612):score = 1
            elif(score >= 3.3612 and score < 6.4178):score = 2
            else: score = 3
            data[1] = score
        

    def moveToFolder(self, data, src_path, dst_path):
        print(src_path)
        print(dst_path)
        i = 0
        for d in data:
            if(os.path.exists(src_path + d[0])):
                infos = d[0].split(".")
                os.system("mv " + src_path + d[0] + " " + dst_path + str(d[1]) + "/originals/" + "{0:0=4d}".format(i) + "." + infos[1])
                i += 1

    def increaseDatasetSize(self, path, nclasses, batch_size):
        for j in range(nclasses):
            datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
            gen = datagen.flow_from_directory(path+str(j),target_size=(350, 350), batch_size=batch_size, save_to_dir=path+str(j), save_format='jpeg')#data must be in a subdir
            i = 0
            for _ in gen:
                i += 1
                if(i>300):break#todo: find a way to remove break

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        nclasses = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        arch = ArchProcessing()
        arch.extractZip("dataset.zip")
        arch.createClassDir(nclasses)
        
        li = arch.getFileContent("SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/train.txt")
        arch.plotData(li)
        arch.updateNotes(li)
        arch.moveToFolder(li, "SCUT-FBP5500_v2/Images/", "data/train/")

        li = arch.getFileContent("SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/test.txt")
        arch.plotData(li)
        arch.updateNotes(li)
        arch.moveToFolder(li, "SCUT-FBP5500_v2/Images/", "data/test/")

        arch.increaseDatasetSize("data/train/",nclasses, batch_size)
        arch.increaseDatasetSize("data/test/",nclasses, batch_size)
