## Assignment 2 : Part A

### Description
- **model.py**: contains CNN model code
- **utils.py**: contains function to get_data (train, validate, test data) and Config class to read parameters.yaml file
- **run_single_config.py**: is used to run a single configuration defined in parameters.yaml file.
- **question2.py**: solution for question 2
- **question4.py**: solution for question 4
- **question5.py**: solution for question 5

### Setup
- install wandb and set up login
- install the following packages
  ```bash
  pip install numpy matplotlib opencv-python tensorflow
  ```
  
### Run single configuration
- Set the desired parameters in  parameters.yaml file
- Choose a location where you want to save your model
- Then run the following command
  ```bash
  python3 run_single_config.py ./parameters.yaml {location_to_save_your_model}
  ```

### Run question 2:
```bash
python3 question2.py
```

### Run question 4:
```bash
python3 question4.py {saved_model_path} {image_path_for_which_you_want_to_visualize_first_conv_layer}
```

### Run question 5:
```bash
python3 question5.py {saved_model_path} {image_path_for_which_you_want_to_visualize_guided_backpropagation}
```


