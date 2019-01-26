from keras.preprocessing.image import ImageDataGenerator

def increaseDatasetSize(path, nclasses, batch_size, break_val):
    for j in range(nclasses):
        datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.1,zoom_range=0.1, rotation_range=360, horizontal_flip=True, vertical_flip=True)
        print(path+str(j)+"/")
        gen = datagen.flow_from_directory(path+str(j), batch_size=batch_size, save_to_dir=path+str(j), save_format='jpeg')#data must be in a subdir
        i = 0
        for _ in gen:
            i += 1
            if(i>break_val):break#todo: find a way to remove break"""

if __name__ == "__main__":
    increaseDatasetSize("data_v2/train/", 3 , 32, 600)
    increaseDatasetSize("data_v2/test/", 3 ,32, 100)
    increaseDatasetSize("data_v2/validation/", 3 , 32, 50)