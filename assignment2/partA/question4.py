#! /usr/bin/env python3


import os
import cv2
import numpy as np
import wandb
import matplotlib.pyplot as plt
from tf.keras.models import load_model, Model


DATA_PATH = os.path.abspath(os.path.realpath(__file__) + "/../data")

# This input size does not refer to the size of the actual image 
# but rather the desired input size that we want to feed to the network
INPUT_SIZE = (256, 256, 3)


def test_4b(model, test_path, input_size, classes):
  fig = plt.figure(figsize=(20, 30))
  plots_cnt = 1
  for cls in classes:
    img_path = os.path.join(test_path, cls)
    image_files = os.listdir(img_path)[:3]
    for img_file in image_files:
      try:
        img = cv2.imread(os.path.join(img_path, img_file))
        img = cv2.resize(img, input_size[:2], interpolation=cv2.INTER_NEAREST) # while training we resized using nearest interpolation
        fig.add_subplot(10, 3, plots_cnt)
        plt.imshow(image)
        plt.axis("off")

        image = image/255
        pred = model.predict(image.reshape(1,image_size,image_size,3))
        class_name = classes[pred.argmax()]
        plt.title("True_label-"+cls+"\n"+"pred_label-"+class_name)
        plots_cnt += 1
      except Exception as e:
        raise e
  wandb.init(project="CS6910_ASSIGNMENT_2", entity="cs21m003_cs21d406")
  wandb.log({"true_image": plt})


def test_4c(model, image_path, input_size, layer=1):
  inputs = model.inputs
  outputs = model.layers[layer].output
  filter_maps_output = Model(inputs=inputs, outputs=outputs)

  image = cv2.imread(image_path)
  image = cv2.resize(image, input_size[:2], interpolation=cv2.INTER_NEAREST) # while training we resized using nearest interpolation

  plt.imshow(image)
  plt.axis("off")
  wandb.init(project="CS6910_ASSIGNMENT_2", entity="cs21m003_cs21d406")
  wandb.log({"filter_img": plt})

  image = image / 255
  filter_1 = filter_maps_output(np.expand_dim(image, axis=0))
  num_filters = filter_1.shape[3]
  fig = plt.figure(figsize=(30, 30))
  rows, columns = 8, 4  # Our best model has 32 filters
  for i in range(num_filters):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(filter_1[0,:,:,i])
    plt.title(str(i+1)+"_filter")
    plt.axis("off")
  wandb.log({"filters":plt})


if __name__ == '__main__':
  model_path, image_path = sys.argv[1], sys.argv[2]
  model = load_model(model_path)

  classes = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"] 
  test_path = os.path.join(DATA_PATH, 'val')
  
  test_4b(model, test_path, INPUT_SIZE, classes)

  test_4c(model, image_path, INPUT_SIZE, layer=1)

