#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)


class BaseModel(tf.keras.Model):
  """
  A super class for both encoder and decoder model.
  It contains all the necessary methods for creating Encoder and Decoder object
  """
  def __init__(self, params, rnn_class):
    super(BaseModel, self).__init__()
    self.set_attributes(params)

  def set_attributes(self, params):
    for k, v in params.items():
      setattr(self, k, v)

  def stacked_layers(self, rnn_class, num_layers):
    """
    It creates a stacked layer of rnn  
    """
    first_rnn = rnn_class(self.latent_dim, return_state=True, return_sequences=True)
    if num_layers <= 1:
      return first_rnn, None
    stacked_input = tf.keras.Input(shape=(None, self.latent_dim))
    stacked_ouput = stacked_input
    for layer in range(1, num_layers):
      stacked_output = tf.keras.layers.Dropout(self.dropout)(stacked_ouput)
      stacked_encoder = rnn_class(self.latent_dim, return_state=True, return_sequences=True)
      x = stacked_encoder(stacked_output)
      stacked_output = x[0]
    stacked_rnn = tf.keras.Model(stacked_input, x)
    return first_rnn, stacked_rnn

  def call(self, *args, **kwargs):
    raise NotImplementedError

  def initialize_hidden_state(self, batch=None):
    if batch == None:
      batch = self.batch_size
    init = [tf.zeros((batch, self.latent_dim))]
    if isinstance(self.first_rnn, tf.keras.layers.LSTM):
      init *= 2
    return init


class Encoder(BaseModel):
  def __init__(self, params, rnn_class):
    super(Encoder, self).__init__(params, rnn_class)
    self.embed = tf.keras.layers.Embedding(self.encoder_vocab_size, self.embed_size, mask_zero=True)
    self.first_rnn, self.stacked_rnn = self.stacked_layers(rnn_class, self.num_encoder_layers)

  def call(self, x, hidden):
    x = self.embed(x)
    x = self.first_rnn(x, initial_state=hidden)
    if self.num_encoder_layers > 1:
      x = self.stacked_rnn(x[0])
    output, state = x[0], x[1:]
    return (output, state)


class Decoder(BaseModel):
  def __init__(self, params, rnn_class):
    super(Decoder, self).__init__(params, rnn_class)
    self.first_rnn, self.stacked_rnn = self.stacked_layers(rnn_class, self.num_decoder_layers)
    self.dense = tf.keras.layers.Dense(self.decoder_vocab_size, activation="softmax")
    if self.use_attention:
      self.attention = Attention(self.latent_dim)

  def call(self, x, hidden, encoder_output=None):
    x = tf.one_hot(x, depth=self.decoder_vocab_size)
    attention_weights = tf.zeros([0, 0])
    if self.use_attention:
      context_vector, attention_weights = self.attention(hidden, encoder_output)
      x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    x = self.first_rnn(x, initial_state=hidden)
    if self.num_decoder_layers > 1:
      x = self.stacked_rnn(x[0])
    output, state = x[0], x[1:]
    output = self.dense(output)
    return (output, state, attention_weights)


class Attention(tf.keras.layers.Layer):
  """
  It implements Bahdanau Attention algorithm (softmax(V*(W1^T*s + W2*h)))
  """
  def __init__(self, latent_dim):
    super(Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(latent_dim)
    self.W2 = tf.keras.layers.Dense(latent_dim)
    self.V = tf.keras.layers.Dense(1)

  def call(self, decoder_state, encoder_output):
    decoder_state = tf.concat(decoder_state, 1)
    decoder_state = tf.expand_dims(decoder_state, 1)
    score = self.V(tf.nn.tanh(self.W1(decoder_state) + self.W2(encoder_output)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * encoder_output
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights
