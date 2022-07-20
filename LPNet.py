import numpy as np
import math
import cv2
import pywt
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, Sequential
import seaborn as sb
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,AveragePooling2D, Dense, Flatten, BatchNormalization, Activation, Dropout, Lambda, GlobalAveragePooling2D, GaussianNoise
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Layer

train_data = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 20,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest')

train_data = train_data.flow_from_directory('/dataset/colon/Training',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_data = ImageDataGenerator(rescale = 1./255)
test_data = test_data.flow_from_directory('/dataset/colon/Testing',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                class_mode = 'categorical')

pywt.wavelist(kind='discrete')

K.set_image_data_format('channels_first')

#define the wavelet
wavelet = pywt.Wavelet('bior1.3')

class DWT_Pooling(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(DWT_Pooling, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(DWT_Pooling, self).build(input_shape) 
    
    @tf.function
    def call(self, inputs):
        band_low = wavelet.rec_lo
        band_high = wavelet.rec_hi
        assert len(band_low) == len(band_high)
        band_length = len(band_low)
        assert band_length % 2 == 0
        band_length_half = math.floor(band_length / 2)

        input_height = inputs.shape[2]
        input_width = inputs.shape[3]

        L1 = input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + band_length - 2 ), dtype=np.float32)
        matrix_g = np.zeros( ( L1 - L, L1 + band_length - 2 ), dtype=np.float32)
        end = None if band_length_half == 1 else (-band_length_half+1)
        
        index = 0
        for i in range(L):
            for j in range(band_length):
                matrix_h[i, index+j] = band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(input_height / 2)), 0:(input_height + band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(input_width / 2)), 0:(input_width + band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(band_length):
                matrix_g[i, index+j] = band_high[j]
            index += 2

        matrix_g_0 = matrix_g[0:(input_height - math.floor(input_height / 2)),0:(input_height + band_length - 2)]
        matrix_g_1 = matrix_g[0:(input_width - math.floor(input_width / 2)),0:(input_width + band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        matrix_low_0 = tf.convert_to_tensor(matrix_h_0,dtype=tf.float32)
        matrix_low_1 = tf.convert_to_tensor(matrix_h_1,dtype=tf.float32)
        matrix_high_0 = tf.convert_to_tensor(matrix_g_0,dtype=tf.float32)
        matrix_high_1 = tf.convert_to_tensor(matrix_g_1,dtype=tf.float32)
        
        L = tf.matmul(matrix_low_0, inputs)
        H = tf.matmul(matrix_high_0, inputs)
        LL = tf.matmul(L, matrix_low_1)
        LH = tf.matmul(L, matrix_high_1)
        HL = tf.matmul(H, matrix_low_1)
        HH = tf.matmul(H, matrix_high_1)
        return LL    
    
    def get_config(self):
        config = super(DWT_Pooling, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2]//2, input_shape[3]//2)

K.set_image_data_format('channels_first')

model = Sequential([
    Conv2D(16,(3,3),padding='same',input_shape=(224, 224, 3)),
    GaussianNoise(0.1),
    BatchNormalization(),
    Activation('relu'),
    DWT_Pooling(),
    
    Conv2D(32,(3,3),padding='same'),
    GaussianNoise(0.1),
    BatchNormalization(),
    Activation('relu'),
    DWT_Pooling(),
    
    Conv2D(64,(3,3),padding='same'),
    GaussianNoise(0.1),
    BatchNormalization(),
    Activation('relu'),
    DWT_Pooling(),
    
    Conv2D(128,(3,3),padding='same'),
    GaussianNoise(0.1),
    BatchNormalization(),
    Activation('relu'),
    DWT_Pooling(),
  
    Flatten(),
    Dense(2),
    Activation('relu'),
    Dense(2,activation='softmax')
])

from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
]

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=METRICS)
model.summary()

reduce_lr=ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=8, min_lr=0.001)

filepath = "/Works-LpNet/weights.best8.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [reduce_lr, checkpoint]

history=model.fit(train_data,validation_data=test_data,epochs=50,batch_size=20,callbacks=callbacks_list)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.load_weights('weights.best8.hdf5')

results = model.evaluate(test_data, batch_size=10)
print("val_loss, val_accuracy:", results)
