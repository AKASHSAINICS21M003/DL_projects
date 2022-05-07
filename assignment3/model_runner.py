#! /usr/bin/env python3

import numpy as np
import tensorflow as tf
from model import Encoder, Decoder


class Runner(object):
  def __init__(self, params, rnn_class, encoder_tokenizer, decoder_tokenizer, encoder=None, decoder=None):
    self.params = params
    self.encoder_tokenizer = encoder_tokenizer
    self.decoder_tokenizer = decoder_tokenizer
    self.encoder = Encoder(params, rnn_class) if encoder is None else encoder
    self.decoder = Decoder(params, rnn_class) if decoder is None else decoder
    self.optimizer = tf.keras.optimizers.Adam()
    self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

  @staticmethod
  def index_word(tokenizer, seq):
    result = ''
    for s in seq:
      if s == 0: # generally we should not encounter this id, but it we do then it is just a unrecognized character
        result += '?'
      else:
        result += tokenizer.index_word[s]
      if result[-1] == '\n':
        break
    return result

  @staticmethod
  def word_index(tokenizer, seq, max_length):
    result = []
    for s in seq:
      result.append(tokenizer.word_index[s])
    result = result + [0] *(max_length - len(result))
    return np.array(result)

  def _custom_loss_function(self, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0)) # finds all the dummy characters that were added to make the sequcence length equal across data
    loss = self.loss_obj(real, pred) # returns the cross entropy for each data
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask # removes all the dummy characters from loss calculation
    return tf.reduce_mean(loss)

  @tf.function
  def _train_step(self, encoder_input, decoder_target):
    loss = 0
    encoder_hidden = self.encoder.initialize_hidden_state(batch=encoder_input.shape[0])
    with tf.GradientTape() as tape:
      encoder_output, encoder_hidden = self.encoder(encoder_input, encoder_hidden)
      decoder_hidden = encoder_hidden
      decoder_input = tf.expand_dims(decoder_target[:, 0], 1)
      for t in range(1, decoder_target.shape[1]):  # unfolding in time
        pred_prob, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
        loss += self._custom_loss_function(decoder_target[:, t], pred_prob)
        decoder_input = tf.expand_dims(decoder_target[:, t], 1)
    batch_loss = loss / int(decoder_target.shape[1])  # normalizing in time
    trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    grads = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(grads, trainable_variables))
    return batch_loss

  def train(self, encoder_input, decoder_target, val_encoder_input, val_decoder_target, epochs=5):
    num_train_data = encoder_input.shape[0]
    indx = np.arange(num_train_data)
    np.random.shuffle(indx)
    train_loss, valid_accuracy = [], []
    for epoch in range(epochs):
      total_loss = 0
      step = 0
      start, end = 0, self.params['batch_size']
      while start < num_train_data:
        batch_indx = indx[start:end]
        inp, targ = encoder_input[batch_indx, :], decoder_target[batch_indx, :]
        total_loss += self._train_step(inp, targ)
        start = end
        end += self.params['batch_size']
        step += 1
      val_acc = self.validation_step(val_encoder_input, val_decoder_target)
      train_loss.append(total_loss/step)
      valid_accuracy.append(val_acc)
      # comment this line if you don't want to print loss/acc
      # print(f"Epoch: {epoch+1}, Loss: {total_loss/step}, val_acc: {val_acc}")
    return train_loss, valid_accuracy

  def translate(self, encoder_input, max_target_len):
    batch = encoder_input.shape[0]
    encoder_hidden = self.encoder.initialize_hidden_state(batch)
    encoder_output, decoder_hidden = self.encoder(encoder_input, encoder_hidden)
    attention_weights = np.zeros((batch, encoder_input.shape[1], max_target_len))
    result = np.zeros((batch, max_target_len), dtype=int)
    result[:, 0] = self.decoder_tokenizer.word_index['\t']
    decoder_input = tf.expand_dims(result[:, 0], 1)
    for t in range(1, max_target_len):
      pred_prob, decoder_hidden, attention_w = self.decoder(decoder_input, decoder_hidden, encoder_output)
      pred_id = tf.argmax(pred_prob, -1)
      result[:, t] = pred_id[:, 0]
      if attention_w.shape != (0, 0):
        attention_weights[:, :, t] = tf.squeeze(attention_w)
      decoder_input = pred_id
    return result, attention_weights

  def validation_step(self, encoder_input, decoder_target):
    max_target_len = decoder_target.shape[1]
    results, _ = self.translate(encoder_input, max_target_len)
    val_accuracy = 0
    for r, t in zip(results, decoder_target):
      res_word = self.index_word(self.decoder_tokenizer, r)
      targ_word = self.index_word(self.decoder_tokenizer, t)
      val_accuracy += 1 if res_word == targ_word else 0
    val_accuracy /= decoder_target.shape[0]
    return val_accuracy
  
  def test_and_save(self, encoder_input, decoder_target, save_path):
    results, _ = self.translate(encoder_input, decoder_target.shape[1])
    test_accuracy = 0
    with open(save_path, 'w') as fp:
      fp.write(f"input,actual_transliteration,predicted_transliteration")
      for e, p, t in zip(encoder_input, results, decoder_target):
        enc_word = self.index_word(self.encoder_tokenizer, e)
        pred_word = self.index_word(self.decoder_tokenizer, p)
        targ_word = self.index_word(self.decoder_tokenizer, t)
        fp.write(f"{enc_word[:-1]},{targ_word[1:-1]},{pred_word[1:-1]}")
        test_accuracy += 1 if pred_word == targ_word else 0
    test_accuracy /= decoder_target.shape[0]
    attention_message = "[With Attention]" if self.params['use_attention'] else "[Without Attention]"
    print(f"{attention_message} Test Accuracy: {test_accuracy}")

  def get_embedding_gradient(self, input_word, max_target_len):
    prediction, gradients = [], []
    encoder_hidden = self.encoder.initialize_hidden_state(1)
    inp_embed = self.encoder.embed(input_word)
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(inp_embed)
      x = self.encoder.first_rnn(inp_embed, initial_state=encoder_hidden)
      if self.params['num_encoder_layers'] > 1:
        x = self.encoder.stacked_rnn(x[0])
      encoder_output, decoder_hidden = x[0], x[1:]
      decoder_input = tf.expand_dims([self.decoder_tokenizer.word_index['\t']], 1)
      for t in range(1, max_target_len):
        pred_prob, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
        gradients.append(tape.gradient(pred_prob, inp_embed)[0])
        pred_id = tf.argmax(pred_prob, -1)
        prediction.append(int(pred_id[0, 0]))
        decoder_input = pred_id
        if prediction[-1] == self.decoder_tokenizer.word_index['\n']:
          break
    return prediction, gradients

