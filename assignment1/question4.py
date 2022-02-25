#! /usr/bin/env python

# Search models of cross entropy loss function


from run_wandb import do_hyperparameter_search


WANDB_PROJECT = "CS6910_ASSIGNMENT_1"
WANDB_ENTITY  = "cs21m003_cs21d406"


def explore_hyperparameters(data_set, search_type, loss_func, runs=5):
  sweep_config = {
    "name": f"{search_type} sweep",
    "method": search_type,
    "metric":{
      "name": "ValidationAccuracy",
      "goal": "maximize"
    },
    "parameters":{
      "epochs": {"values": [5, 10]}, 
      "batch_size": {"values": [16, 32, 64]}, 
      "hidden_layers": {"values": [3, 4, 5]}, 
      "hidden_nodes": {"values": [32, 64, 128]},
      "reg": {"values": [0, 0.0005, 0.5]},
      "weight_init": {"values": ['random', 'xavier']} , 
      "act_func": {"values": ["sigmoid", "tanh", "relu"]}, 
      "loss_func": {"values": [loss_func]}, 
      "learning_rate": {"values": [1e-3, 1e-4]},   
      "optimizer": {"values": ["sgd", "momentum_gd", "nesterov_gd", "rmsprop", "adam", "nadam"]},
      "search_type": {"values": [search_type]},
      "data_set": {"values": [data_set]}
    }
  }
  do_hyperparameter_search(sweep_config, WANDB_PROJECT, WANDB_ENTITY, runs=runs)


if __name__ == '__main__':
  explore_hyperparameters("fashion_mnist", "bayes", "cross_entropy", runs=40)
  explore_hyperparameters("fashion_mnist", "random", "cross_entropy", runs=40)


