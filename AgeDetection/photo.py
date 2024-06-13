import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Sequential 
import numpy as np 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import ResNet50
import pandas as pd


def load_train(path): 
    labels = pd.read_csv(path+'labels.csv')
    train_datagen = ImageDataGenerator(
        validation_split = 0.25, 
        rescale = 1/255., 
        horizontal_flip = True, 
        vertical_flip = False,
        width_shift_range = 0.2, 
        height_shift_range = 0.2
    )
    train_dataflow = train_datagen.flow_from_dataframe(
        dataframe=labels, 
        directory = path + 'final_files/',
        x_col = 'file_name',
        y_col = 'real_age', 
        class_mode = 'raw', 
        subset = 'training', 
        target_size = (150, 150),
        batch_size = 16
    )
    return train_dataflow
  



def load_test(path): 
   
     labels = pd.read_csv(path + 'labels.csv')
     validation_datagen = ImageDataGenerator(validation_split = .25, rescale = 1/255.)
     val_dataflow = validation_datagen.flow_from_dataframe(
         dataframe = labels, 
         directory = path + 'final_files/',
         x_col = 'file_name',
         y_col = 'real_age', 
         class_mode = 'raw', 
         subset = 'validation', 
         target_size = (150, 150), 
         batch_size = 16

     )
     return val_dataflow


def create_model(input_shape): 

    backbone = ResNet50(input_shape=input_shape,
                 
                 include_top=False,
                 weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5') 
    


    optimizer = Adam(lr=0.00001) 
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu')) 

    model.compile(optimizer=optimizer, loss='mse', 
                  metrics=['mae']) 
    return model 
  
  
def train_model(model, train_data, test_data, batch_size=None, epochs=5, 
               steps_per_epoch=None, validation_steps=None): 
 
    model.fit(train_data, 
              validation_data=test_data, 
              batch_size=batch_size, epochs=epochs, 
              steps_per_epoch=steps_per_epoch, 
              validation_steps=validation_steps, 
              verbose=2, shuffle=True) 
    return model