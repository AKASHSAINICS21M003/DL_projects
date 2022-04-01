#! /usr/bin/env python3

import sys
import tensorflow as tf
from tf.keras.models import load_model, Model


np.random.seed(0)


@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,tf.float32)  * tf.cast(x>0,tf.float32) * dy
  return tf.nn.relu(x), grad


def norm_flat_image(img):
  grads_norm = img[:,:,0]+ img[:,:,1]+ img[:,:,2]
  grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
  return grads_norm


def guided_backpropagation_model(conv_layer_5):
  gb_model = Model(inputs = [model.inputs], outputs = [model.layers[conv_layer_5].output])
  layer_dict = [layer for layer in gb_model.layers if hasattr(layer,'activation')]
  for layer in layer_dict:
    if layer.activation == tf.keras.activations.relu:
      layer.activation = guidedRelu
  return gb_model


def main(model_path, image_path):
  model = load_model(model_path)
  conv_layer_5 = 16
  output_size = model.layers[conv_layer_5].output.shape[1:]
  
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

    normalized_grads = norm_flat_image(grads)
    plt.figure(figsize=(15,45))
    plt.subplot(10, 1, pt+1)
    plt.imshow(normalized_grads, vmin=0.3, vmax=0.7, cmap="gray")
    plt.axis("off")


if __name__ == '__main__':
  model_path, image_path = sys.argv[1], sys.argv[2]
  main(model_path, image_path)

