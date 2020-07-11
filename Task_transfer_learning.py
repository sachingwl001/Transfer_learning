
from keras.applications import resnet50
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

model=resnet50.ResNet50(weights='imagenet' ,input_shape=(224,224,3) , include_top=False)

for layers in model.layers:
    layers.trainable = False 	


t_layer=model.output
t_layer=Flatten()(t_layer)
t_layer=Dense(units=1024,activation='relu')(t_layer)
t_layer=Dense(units=1024,activation='relu')(t_layer)
t_layer=Dense(units=512,activation='relu')(t_layer)
t_layer=Dense(units=3,activation='softmax')(t_layer)

n_model=Model(inputs= model.input , outputs=t_layer)

n_model.summary()

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'dataset/train_set/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        'dataset/test_set/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')


n_model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("face_detect.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# Enter the number of training and validation samples here
nb_train_samples = 600
nb_validation_samples = 130	

# We only train 5 EPOCHS 
epochs = 3
batch_size = 16

history = n_model.fit_generator(
    training_set,
    steps_per_epoch = nb_train_samples // batch_size ,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = test_set,
    validation_steps = nb_validation_samples // batch_size )

training_set.class_indices

from keras.models import load_model
model=load_model('face_detect.h5')


import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

pred_dict = {"[1]": "Rahul ", 
             "[0]": "Sachin",
             "[2]": "Papa",
            }

pred_dict_n = {"n0": "Rahul", 
               "n1": "Sachin",
               "n2": "Papa",
                       }

def draw_test(name, pred, im):
    face_dict = pred_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, face_dict, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + pred_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage("dataset/train_set/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(model.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()