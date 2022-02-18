#!/usr/env/bin python

from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


MAPPINGS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


def plot_sample_images(images, labels, mappings):
  image_map = {}
  map_cnt = 0
  for img, label in zip(images, labels):
    if map_cnt == 10:
      break
    if label not in image_map:
      image_map[label] = img
      map_cnt += 1
  
  fig = plt.figure(figsize=(10,5))
    
  for label, img in image_map.items():
    ax = fig.add_subplot(2,5,label+1)
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(mappings[label])


if __name__ == '__main__':
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  plot_sample_images(x_train, y_train, MAPPINGS)

