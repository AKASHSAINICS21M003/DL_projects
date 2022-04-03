#! /usr/bin/env python3

import sys
import tensorflow as tf
from tf.keras.models import load_model, Model

import wandb
import matplotlib.pyplot as plt


@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,tf.float32)  * tf.cast(x>0,tf.float32) * dy
  return tf.nn.relu(x), grad


def normalize_image(img):
  grads_norm = img[:,:,0] + img[:,:,1] + img[:,:,2]
  grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
  return grads_norm


def guided_backpropagation_model(model, conv_layer):
  gb_model = Model(inputs = [model.inputs], outputs = [model.layers[conv_layer].output])
  for layer in gb_model.layers:
    if hasattr(layer, "activation") and (layer.activation == tf.keras.activations.relu):
      layer.activation = guidedRelu
  return gb_model


def main(model_path, image_path):
  model = load_model(model_path)
  conv_layer_5 = 16
  output_shape = model.layers[conv_layer_5].output.shape[1:]

  input_img = cv2.imread(image_path)
  input_img = cv2.resize(input_img, (256, 256), interpolation=cv2.INTER_NEAREST) # while training we resized using nearest interpolation

  plt.imshow(input_img)
  plt.axis("off")
  wandb.init(project="guided_backprop", entity="cs21m003_cs21d406")
  wandb.log({"true_image":plt})

  plt.figure(figsize=(30,30))
  input_img /= 255
  gb_model = guided_backpropagation_model(model, conv_layer_5)
  for pt in range(10):
    idx = np.random.randint(0, output_shape[0])
    idy = np.random.randint(0, output_shape[1])
    idz = np.random.randint(0, output_shape[2])

    mask = np.zeros((1, *output_shape), dtype="float")
    mask[0, idx, idy, idz] = 1.0

    with tf.GradientTape() as tape:
      tape.watch(input_img)
      result = gb_model(input_img) * mask
    grads = tape.gradient(result, input_img)[0]

    normalized_grads = normalize_image(grads)
    plt.subplot(10, 1, pt+1)
    plt.imshow(normalized_grads, vmin=0.3, vmax=0.7, cmap="gray")
    plt.axis("off")
  wandb.log({"neurons":plt})  


if __name__ == '__main__':
  model_path, image_path = sys.argv[1], sys.argv[2]
  main(model_path, image_path)

