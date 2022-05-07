#! /usr/bin/env python3

import os
import yaml
import numpy as np
import tensorflow as tf
from model_runner import Runner
from dataset import Dataset
from visualizer import Visualizer


RNN_MAP = {
    "lstm": tf.keras.layers.LSTM,
    "gru": tf.keras.layers.GRU,
    "rnn": tf.keras.layers.SimpleRNN
}

DATA_PATH = os.path.abspath(os.path.realpath(__file__) + '/data/dakshina_dataset_v1.0/hi/lexicons')

CONFIG_PATH = os.path.abspath(os.path.realpath(__file__) + "/parameters.yaml")


def read_config(config_path):
  with open(config_path, 'r') as fp:
    params = yaml.safe_load(fp)
  return params


def train(dataset, params): # with attention
  dataset = Dataset(DATA_PATH)
  train_encoder_input, train_decoder_target, val_encoder_input, val_decoder_target = dataset.get_training_data()
  encoder_vocab_size, decoder_vocab_size = dataset.vocab_size
  params['encoder_vocab_size'] = encoder_vocab_size
  params['decoder_vocab_size'] = decoder_vocab_size
  params['use_attention'] = True

  runner = Runner(params, RNN_MAP[params['rnn_type']], dataset.encoder_tokenizer, dataset.decoder_tokenizer)
  train_loss, val_acc = run.train(train_encoder_input, train_decoder_target,
                                   val_encoder_input, val_decoder_target, epochs=params['epochs'])
  print(f"[With Attention] Training Loss: {train_loss} Validation Accuracy: {val_acc}")
  return runner


def run_on_test_data(dataset, runner):
  save_path = os.path.abspath(os.path.realpath(__file__) + "/predictions_attention/test_with_attention.csv")
  test_encoder_input, test_decoder_target = dataset.get_testing_data()
  run.test_and_save(test_encoder_input, test_decoder_target, save_path)


def heat_map(dataset, runner):
  test_encoder_input, test_decoder_target = dataset.get_testing_data()
  max_target_len = test_decoder_target.shape[1]
  viz = Visualizer(runner)
  viz.viz_attention(test_encoder_input[:9], max_target_len)



if __name__ == '__main__':
  dataset = Dataset(DATA_PATH)
  params = read_config(CONFIG_PATH)
  runner = train(dataset, params)
  run_on_test_data(dataset, runner)
  heat_map(dataset, runner)


