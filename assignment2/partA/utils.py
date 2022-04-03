#! /usr/bin/env python3

import yaml
from os.path import join
from tf.keras.preprocessing.image import ImageDataGenerator


class Config:
  @staticmethod
  def from_yaml(filepath):
    """
    reads configurations from yaml file
    """
    with open(filepath, 'r') as fp:
      dataMap = yaml.safe_load(fp)
    return Config.from_dict(dataMap)

  @staticmethod
  def from_dict(dataMap):
    """
    reads configurations from dictonary

    For example:
    params = {"name": "CNN", "type": "Neural Network"}
    config = Config.from_dict(params)
    # then one can access the config parameters as
    print(config.name) # prints: CNN
    print(config.type) # prints: Neural Network
    """
    config = Config()
    for name, value in dataMap.items():
      setattr(config, name, value)


def get_data(path, target_img_size, augmentation=True, seed=0):
  """
  path: data directory path and the path should contain data in "train" and "val" directory.
  target_img_size: is of the format (img_height, img_width)
                   and it refers to the desired size that we want our data to be resized.
  augmentation: if true train dataset will contain augumented data 
  """
  train_path, test_path = join(path,"train"), join(path,"val")
  if len(target_img_size) == 3:
    target_img_size = target_img_size[:2] # assuming last value is channel_size
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
                                                   target_size=target_img_size,
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   shuffle=True,
                                                   seed=seed)
  valid_data = train_generator.flow_from_directory(directory=train_path,
                                                   target_size=target_img_size,
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   shuffle=True,
                                                   seed=seed)
  test_data = test_generator.flow_from_directory(directory=test_path,
                                                 target_size=target_img_size,
                                                 color_mode="rgb",
                                                 class_mode="categorical",
                                                 shuffle=True,
                                                 seed=seed)
  return train_data, valid_data, test_data
