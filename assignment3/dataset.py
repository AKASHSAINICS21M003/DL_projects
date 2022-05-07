#! /usr/bin/env python3

import numpy as np
import tensorflow as tf


class Dataset(object):
  def __init__(self, data_path):
    self.train_path = os.path.join(data_path, 'hi.translit.sampled.train.tsv')
    self.validation_path = os.path.join(data_path, 'hi.translit.sampled.dev.tsv')
    self.test_path = os.path.join(data_path, 'hi.translit.sampled.test.tsv')
    self.encoder_tokenizer = None
    self.decoder_tokenizer = None
    self.load_train_data = False

  @staticmethod
  def _read_file(filepath):
    encoder_words, decoder_words = [], []
    with open(filepath, 'r') as fp:
      for line in fp:
        line = line.strip()
        if not line:
          continue
        target, src, _ = [x.strip() for x in line.split('\t')]
        src = src + "\n"  # \n represents end_of_word
        encoder_words.append(src)
        target = "\t" + target + "\n"  # \t represents start_word and \n represents end_of_word
        decoder_words.append(target)
    return encoder_words, decoder_words

  @property
  def vocab_size(self):
    assert self.load_train_data, "Seems like you want to know the vocab size even before loading train data"
    encoder_vocab_size = len(self.encoder_tokenizer.word_index) + 1 # number 0 is reserved for padding
    decoder_vocab_size = len(self.decoder_tokenizer.word_index) + 1 # number 0 is reserved for padding
    return encoder_vocab_size, decoder_vocab_size

  def _reset_tokenizer(self):
    self.load_train_data = False
    self.encoder_tokenizer = None
    self.decoder_tokenizer = None

  def _get_tokenizer(self, encoder_words, decoder_words):
    assert self.load_train_data, "Seems like you are trying to access test data even before accessing train data !!"
    if self.encoder_tokenizer is None:
      self.encoder_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
      self.encoder_tokenizer.fit_on_texts(encoder_words)
    if self.decoder_tokenizer is None:
      self.decoder_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
      self.decoder_tokenizer.fit_on_texts(decoder_words) 
    return self.encoder_tokenizer, self.decoder_tokenizer

  def _get_dataset(self, encoder_words, decoder_words):
    encoder_tokenizer, decoder_tokenizer = self._get_tokenizer(encoder_words, decoder_words)
    encoder_input = encoder_tokenizer.texts_to_sequences(encoder_words)
    encoder_input = tf.keras.preprocessing.sequence.pad_sequences(encoder_input, padding='post') 
    decoder_target = decoder_tokenizer.texts_to_sequences(decoder_words)
    decoder_target = tf.keras.preprocessing.sequence.pad_sequences(decoder_target, padding='post')
    return encoder_input, decoder_target

  def get_training_data(self):
    try:
      self.load_train_data = True
      train_encoder_words, train_decoder_words = self._read_file(self.train_path)
      train_encoder_input, train_decoder_target = self._get_dataset(train_encoder_words, train_decoder_words)
      val_encoder_words, val_decoder_words = self._read_file(self.validation_path)
      val_encoder_input, val_decoder_target = self._get_dataset(val_encoder_words, val_decoder_words)
    except Exception as ex:
      self._reset_tokenizer()
      raise ex
    return train_encoder_input, train_decoder_target, val_encoder_input, val_decoder_target

  def get_testing_data(self):
    test_encoder_words, test_decoder_words = self._read_file(self.test_path)
    test_encoder_input, test_decoder_target = self._get_dataset(test_encoder_words, test_decoder_words)
    return test_encoder_input, test_decoder_target
