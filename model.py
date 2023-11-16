from __future__ import division, print_function
# coding=utf-8

import os

import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(9876)
#np.random.seed(1337)
train_dir = 'train'
validation_dir = 'validation'
train_length = len(os.listdir(os.path.join(train_dir, 'NON-Scratch'))) + len(os.listdir(os.path.join(train_dir, 'Scratch')))
validation_length = len(os.listdir(os.path.join(validation_dir, 'NON-Scratch'))) + len(os.listdir(os.path.join(validation_dir, 'Scratch')))
batch_size = 7
print(train_length,validation_length)
model = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(32, (3, 3), activation='relu' , input_shape=(640,128, 3)),
 tf.keras.layers.MaxPool2D((2, 2)),
 tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
 tf.keras.layers.MaxPool2D((2, 2)),
 tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
 tf.keras.layers.MaxPool2D((2, 2)),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(units=512, activation='relu'),
 tf.keras.layers.Dense(units=1, activation='sigmoid'),
])

model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='Nadam',
              metrics=['acc'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(640,128), batch_size=batch_size, class_mode='binary')

validation_generator = train_datagen.flow_from_directory(validation_dir,
                                                         target_size=(640,128), batch_size=batch_size,
                                                         class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=int(train_length / batch_size),
      epochs=22,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=int(validation_length / batch_size)
)

model.save(os.path.join(os.path.dirname(__file__), '_crack_detection.h5'))
