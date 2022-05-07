#! /usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
from IPython.display import HTML as html_print
from IPython.display import display


class Visualizer(object):
  def __init__(self, runner):
    self.runner = runner

  @staticmethod
  def cstr(s, color='black'):
    if s == ' ':
      return "<text style=color:#000;padding-left:10px;background-color:{}> </text>".format(color, s)
    else:
      return "<text style=color:#000;background-color:{}>{} </text>".format(color, s)

  @staticmethod
  def print_color(t):
	  display(html_print(''.join([Visualizer.cstr(ti, color=ci) for ti,ci in t])))
   
  @staticmethod
  def get_clr(value):
    colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8'
              '#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
              '#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
              '#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
    value = int(value * 100 / 5)
    return colors[value]

  @staticmethod
  def scale(x):
    z = 1/(1 + np.exp(-x))
    # min, max = np.min(x), np.max(x)
    # z = (x - min) / (max - min) 
    return z

  def viz_connectivity(self, encoder_input, max_target_len=21):
    runner = self.runner
    for inp in encoder_input:
      pred, grads = runner.get_embedding_gradient(tf.expand_dims(inp, 0), max_target_len)
      orig_word = runner.index_word(runner.encoder_tokenizer, inp)[:-1]
      pred_word = runner.index_word(runner.decoder_tokenizer, pred)[:-1]
      print(f"\nVisualizing tranliteration of word {orig_word}")
      for gd in grads:
        norm = tf.norm(gd, axis=1)[:len(orig_word)]
        scaled = self.scale(norm)
        for pw in pred_word:
          print(f"For predicting char {pw}: ", end=" ")
          word_color = []
          for i in range(len(orig_word)):
            word_color.append((orig_word[i], self.get_clr(scaled[i])))
          self.print_color(word_color)

  def viz_attention(self, encoder_input, max_target_len):
    runner = self.runner
    font_path = "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"
    results, attention_weights = runner.translate(encoder_input, max_target_len)
    fig = plt.figure(figsize=(20,20))
    for i, (e, p) in enumerate(zip(encoder_input, results)):
      enc_word = runner.index_word(runner.encoder_tokenizer, e)
      pred_word = runner.index_word(runner.decoder_tokenizer, p)
      enc_size, pred_size = len(enc_word), len(pred_word)
      attention = attention_weights[i, :enc_size, 1:pred_size]
      attention /= attention.sum(axis=0, keepdims=True)

      ax = fig.add_subplot(3, 3, i+1)
      ax.imshow(attention)

      hindi_font = FontProperties(fname=font_path, size=10)  
      pred_chars = [str(x) for x in pred_word[1:]]
      ax.set_xticks(np.arange(pred_size-1))
      ax.set_xticklabels(pred_chars, fontproperties=hindi_font)

      enc_chars = [str(x) for x in enc_word]
      ax.set_yticks(np.arange(enc_size))
      ax.set_yticklabels(enc_chars)
    plt.show()

