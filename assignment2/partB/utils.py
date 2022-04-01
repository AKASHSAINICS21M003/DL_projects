#! /usr/bin/env python3

from os.path import join
from tf.keras.preprocessing.image import ImageDataGenerator


def get_data(path, target_size, augmentation=True, seed=0):
  train_path = join(path,"train")
  test_path = join(path,"val")
  if augmentation:
    train_generator = ImageDataGenerator(rescale=1./255,
                                         rotation_range=90,
                                         zoom_range=0.2,
                                         shear_range=0.2,
                                         validation_split=0.1,
                                         horizontal_flip=True)
  else:
    train_generator = ImageDataGenerator(rescale=1./255, validation_split=0.1)
  test_generator = ImageDataGenerator(rescale=1./255)
  train_data = train_generator.flow_from_directory(directory=train_path,
                                                   target_size=target_size,
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   shuffle=True,
                                                   seed=seed)
  valid_data = train_generator.flow_from_directory(directory=train_path,
                                                   target_size=target_size,
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   shuffle=True,
                                                   seed=seed)
  test_data = test_generator.flow_from_directory(directory=test_path,
                                                 target_size=target_size,
                                                 color_mode="rgb",
                                                 class_mode="categorical",
                                                 shuffle=True,
                                                 seed=seed)
  return train_data, valid_data, test_data


