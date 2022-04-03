#! /usr/bin/env python3

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


class CNN(object):
  def __init__(self, config):
    """
    config: can be a wandb config class or custom Config class
    """
    self.config = config
    self.initialize()

  def initialize(self):
    config = self.config
    self.model = self.get_model(config)
    self.model.compile(optimizer=Adam(config.lr),
                       loss="categorical_crossentropy"
                       metrics="categorical_accuracy")

  def get_model(self, config):
    """
    get CNN model as defined in assignment 2
    """
    num_of_conv_layer = 5 # defined in assignment 2
    model = Sequential()
    for layer in range(num_conv_layers):
      if layer == 0:
        model.add(Conv2D(filters=config.num_filters,
                         kernel_size=config.kernel_size,
                         input_shape=config.input_size,
                         padding=config.padding,
                         kernel_initializer="he_uniform",
                         data_format="channels_last"))
      else:
        num_filters = config.num_filters * (config.filter_org**layer)
        model.add(Conv2D(filters=num_filters,
                         kernel_size=config.kernel_size, 
                         padding=config.padding,
                         kernel_initializer="he_uniform"))
      model.add(Activation(config.activation_func))
      if config.batch_norm:
        model.add(BatchNormalization())
      model.add(MaxPooling2D(pool_size=(2,2)))
    if config.batch_norm:
      model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(config.dense_size))
    model.add(Activation(config.activation_func))
    if config.batch_norm:
      model.add(BatchNormalization())
    model.add(Dropout(config.drop_out))
    model.add(Dense(config.output_size))
    model.add(Activation("softmax"))
    return model

  def train(self, train_data, val_data, epochs=3, callbacks=None):
    self.model.fit(train_data, batch_size=self.config.batch_size,
                   epochs=epochs, validation_data=val_data, callbacks=callbacks)

  def evaluate(self, data):
    return self.model.evaluate(data, batch_size=self.config.batch_size)

  def predict(self, image):
    return self.model.predict(image)

  def save(self, path):
    self.model.save(path)


