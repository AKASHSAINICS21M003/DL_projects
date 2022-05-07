#! /usr/bin/env python3

import yaml
import numpy as np
import tensorflow as tf
from model_runner import Runner
from visualizer import Visualizer
from dataset import Dataset


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


def train(dataset, params)
  train_encoder_input, train_decoder_target, val_encoder_input, val_decoder_target = dataset.get_training_data()
  encoder_vocab_size, decoder_vocab_size = dataset.vocab_size
  params['encoder_vocab_size'] = encoder_vocab_size
  params['decoder_vocab_size'] = decoder_vocab_size
  runner = Runner(params, RNN_MAP[params['rnn_type']], dataset.encoder_tokenizer, dataset.decoder_tokenizer)
  train_loss, val_acc = run.train(train_encoder_input, train_decoder_target,
                                   val_encoder_input, val_decoder_target, epochs=params['epochs'])
  print(f"[Without Attention] Training Loss: {train_loss} Validation Accuracy: {val_acc}")
  return runner


def question_4(dataset, runner):
  save_path = os.path.abspath(os.path.realpath(__file__) + "/predictions_vanilla/test_without_attention.csv")
  test_encoder_input, test_decoder_target = dataset.get_testing_data()
  run.test_and_save(test_encoder_input, test_decoder_target, save_path)


def question_6(dataset, runner, num_samples=3):
  test_encoder_input, test_decoder_target = dataset.get_testing_data()
  max_target_len = test_decoder_target.shape[1]
  viz = Visualizer(runner)
  viz.viz_connectivity(test_encoder_input[:num_samples, :], max_target_len)


if __name__ == '__main__':
  dataset = Dataset(DATA_PATH)
  params = read_config(CONFIG_PATH)
  runner = train(dataset, params)

  question_4(dataset, runner)
  question_6(dataset, runner)


