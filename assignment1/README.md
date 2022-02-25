## Assignment1

### Description
+ Model codes are in
  + feed_forward_nn.py
  + activation_func.py
  + loss_func.py
  + measure.py
  + optimizer.py
+ Helper codes :
  + run_single_config.py
  + run_wandb.py
+ Question codes:
  + question1.py
  + question4.py
  + question7.py
  + question8.py
  + question10.py

### Set up
+ Install wandb and set up login
+ Install following pacakges
```bash
pip install numpy, matplotlib, PyYAML, sklearn
```

### Run single configuration
+ Set parameters in parameters.yaml file.
+ Then run the following python file as follows
  + Takes 2 arguments
     + Parameters config file
     + true/false: whether to run the model to print test accuracy or not 
```bash
python run_single_config.py ./parameters.yaml false
```

### Run question1
```bash
python question1.py
```

### Run question4
```bash
python question4.py
```

### Run question7
```bash
python question7.py
```

### Run question8
```bash
python question8.py
```

### Run question10
```bash
python question10.py
```
