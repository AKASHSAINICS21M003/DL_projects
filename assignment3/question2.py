#! /usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from model_runner import Runner
from dataset import Dataset


RNN_MAP = {
    "lstm": tf.keras.layers.LSTM,
    "gru": tf.keras.layers.GRU,
    "rnn": tf.keras.layers.SimpleRNN
}

DATA_PATH = os.path.abspath(os.path.realpath(__file__) + '/data/dakshina_dataset_v1.0/hi/lexicons')

WANDB_PROJECT = "CS6910_ASSIGNMENT_3"
WANDB_ENTITY = "cs21m003_cs21d406"
WANDB_RUNS = 20

EPOCHS = 10


class WandbRunner(object):
  def __init__(self):
    dataset = Dataset(DATA_PATH)
    self.train_encoder_input, self.train_decoder_target, self.val_encoder_input, self.val_decoder_target = dataset.get_training_data()
    self.encoder_vocab_size, self.decoder_vocab_size = dataset.vocab_size
    self.encoder_tokenizer = dataset.encoder_tokenizer
    self.decoder_tokenizer = dataset.decoder_tokenizer

  def run_wandb(self):
    wandb.init()
    config = wandb.config
    params = {
      "encoder_vocab_size": self.encoder_vocab_size,
      "decoder_vocab_size": self.decoder_vocab_size, 
      "embed_size": config.inp_embed_size,
      "latent_dim": config.latent_dim,
      "num_encoder_layers": config.num_encoder_layers,
      "num_decoder_layers": config.num_decoder_layers,
      "dropout": config.dropout,
      "batch_size": config.batch_size, 
      "use_attention": config.attention
    }
    rnn_class = RNN_MAP[config.rnn_type]
    runner = Runner(params, rnn_class, self.encoder_tokenizer, self.decoder_tokenizer)
    train_loss, valid_accuracy = runner.train(self.train_encoder_input, self.train_decoder_target,
                                              self.val_encoder_input, self.val_decoder_target, epochs=config.epochs)
    wandb.run.name=f"emb_{config.inp_embed_size}_ld_{config.latent_dim}_" +\
                   f"nel_{config.num_encoder_layers}_ndl_{config.num_decoder_layers}_" +\
                   f"dpt_{config.dropout}_at_{config.attention}_bs_{config.batch_size}_cell_{config.rnn_type}"
    for tl, va in zip(train_loss, valid_accuracy):
      wandb.log({"training_loss": tl, "validation_accuracy": va})

  def do_hyperparameter_search(self):
    sweep_config = {
        "name": "Transliteration Search",
        "method": "random",
        "metric": {
            "name": "validation_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "inp_embed_size": {"values": [32, 64, 128]}, 
            "latent_dim": {"values": [32, 64, 128, 256]}, 
            "num_encoder_layers": {"values": [1, 2, 3]},
            "num_decoder_layers": {"values": [1, 2, 3]},
            "dropout": {"values": [0.2, 0.3, 0.4]},
            "batch_size": {"values": [32, 64]},
            "attention": {"values": [False]}, 
            "rnn_type": {"values": ["rnn", "lstm", "gru"]},
            "epochs": {"values": [EPOCHS]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
    wandb.agent(sweep_id, function=self.run_wandb, count=WANDB_RUNS)


if __name__ == '__main__':
  wandb_runner = WandbRunner()
  wandb_runner.do_hyperparameter_search()


