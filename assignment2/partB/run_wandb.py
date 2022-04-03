#! /usr/bin/env python3

import wandb

import numpy as np
from wandb.keras import WandbCallback
from models import inc_v2, inc_v3, resnet, xception
from tensorflow.keras.optimizers import Adam
from utils import get_data


DATA_PATH = "../data/in.."

IMAGE_SIZE = (256, 256, 3)

TARGET_SIZE = 10

MODEL_MAP = {"inception_v2":inc_v2,
             "inception_v3":inc_v3,
             "xception":xception,
             "resnet_50":resnet
            }


def run_wandb():
    wandb.init()
    config = wandb.config
    base_model = MODEL_MAP[config.models]
    wandb.run.name = f"model_{config.models}_e_{config.epochs}_" +\
                     f"bs_{config.batch_size}_rate_{config.lr}_" +\
                     f"aug_{config.augmentation}_BN_{config.batch_norm}_" +\
                     f"drp_{config.drop_out}_dense_{config.dense_size}"
    train_data, val_data, test_data = get_data(DATA_PATH, IMAGE_SIZE, config.augmentation)
    model_1 = base_model(IMAGE_SIZE, TARGET_SIZE, config.drop_out, config.batch_norm,config.dense_size)
    model_1.model.compile(optimizer=Adam(config.lr),  # Optimizer
                          loss="categorical_crossentropy",
                          metrics="categorical_accuracy")
    model_1.model.fit(train_data, epochs=config.epochs,
                      batch_size=config.batch_size,
                      validation_data=val_data,
                      callbacks=[WandbCallback()])
    model_1.model_b.trainable=True


def do_hyperparameter_search_using_wandb():
    sweep_config = {
    "name": "random sweep",
    "method": "random",
    "metric":{
      "name": "ValidationAccuracy",
      "goal": "maximize"
    },
    "parameters":{
      "epochs": {"values": [5,10]}, 
      "models": {"values": ["inception_v2","inception_v3","xception","resnet_50"]},
      "batch_size": {"values": [64]}, 
      "lr": {"values": [1e-3, 1e-4]}, 
      "augmentation": {"values": [True,False]} , 
      "batch_norm": {"values": [True,False]},
      "drop_out": {"values": [0.3,0.4,0.5]},
      "dense_size": {"values": [32,64]}}}
  
    sweep_id = wandb.sweep(sweep_config, project = "pretrained_model",entity='cs21m003_cs21d406')
    wandb.agent(sweep_id, function=run_wandb,count=10)


if __name__ == '__main__':
  do_hyperparameter_search_using_wandb()

