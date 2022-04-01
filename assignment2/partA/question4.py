#! /usr/bin/env python3


import os
import cv2
import numpy as np
import wandb
import matplotlib.pyplot as plt
from tf.keras.models import load_model


def test_4b(model, test_path, classes):
  fig = plt.figure(figsize=(20, 30))
  plots_cnt = 1
  for cls in classes:
    img_path = os.path.join(test_path, cls)
    image_files = os.listdir(img_path)[:3]
    for img_file in image_files:
      try:
        img = cv2.imread(os.path.join(img_path, img_file))
        img = cv2.resize(img, (256, 256))
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


def test_4c(test_data, model, layer='conv2d'):
  outputs = model.get_layer(layer).output
  inputs = model.inputs
  filter_maps_output = Model(inputs=inputs,outputs=outputs)
  n_2 = random.randrange(test_data[0][0].shape[0])
  n_3 = random.randrange(test_data[0][0].shape[0])
  filter_1 = filter_maps_output(test_data[n_2][0])
  plt.imshow(test_data[n_2][0][n_3])
  plt.axis("off")
  wandb.init(project="CS6910_ASSIGNMENT_2", entity="cs21m003_cs21d406")
  wandb.log({"filter_img": plt})
  num_filters = filter_1.shape[3]
  fig = plt.figure(figsize=(30, 30))
  rows, columns = 8, 4
  for i in range(num_filters):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(filter_1[n_3,:,:,i])
    plt.title(str(i+1)+"_filter")
    plt.axis("off")
  wandb.log({"filters":plt})


if __name__ == '__main__':
  model_path, data_path = sys.argv[1], sys.argv[2]
  classes = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]
 
  test_path = os.path.join(data_path, 'val')
  _, _, test_data = get_data(data_path, len(classes)) 
  
  model = load_model(model_path)

  test_4b(model, test_path, classes)

  test_4c(test_data, model)

