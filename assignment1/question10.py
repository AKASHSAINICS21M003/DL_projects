#! /usr/bin/env python

# Run 3 best estimated configurations


from run_wandb import do_plot


WANDB_PROJECT = "CS6910_ASSIGNMENT_1"
WANDB_ENTITY  = "cs21m003_cs21d406"


def set_name(params):
  ls = "ce" if params["loss_func"] == "cross_entropy" else "mse"
  name = "e_{}_bs_{}_hl_{}_hn_{}_init_{}_ac_{}_reg_{}_ls_{}_opt_{}_lr_{}".format(
    params["epochs"],
    params["batch_size"],
    params["hidden_layers"],
    params["hidden_nodes"],
    params["weight_init"],
    params["act_func"],
    params["reg"],
    ls,
    params["optimizer"],
    params["learning_rate"]
  )
  params["name"] = name


def main(data_set):
  # Configuration 1:
  params1 = {
    "project"           : WANDB_PROJECT,
    "entity"            : WANDB_ENTITY,
    "epochs"            : 5,
    "batch_size"        : 32,
    "hidden_nodes"      : 128,
    "hidden_layers"     : 3,
    "weight_init"       : "xavier",
    "act_func"          : "tanh",
    "reg"               : 0.0005,
    "loss_func"         : "cross_entropy",
    "optimizer"         : "nadam",
    "learning_rate"     : 1e-3,
    "data_set"          : data_set,
  }
  set_name(params1)
  do_plot(params1)

  # Configuration 2:
  params2 = {
    "project"           : WANDB_PROJECT,
    "entity"            : WANDB_ENTITY, 
    "epochs"            : 5,
    "batch_size"        : 32,
    "hidden_nodes"      : 64,
    "hidden_layers"     : 3,
    "weight_init"       : "xavier",
    "act_func"          : "tanh",
    "reg"               : 0.0005,
    "loss_func"         : "cross_entropy",
    "optimizer"         : "adam",
    "learning_rate"     : 1e-3,
    "data_set"          : data_set,
  }
  set_name(params2)
  do_plot(params2)

  # Configuration 3:
  params3 = {
    "project"           : WANDB_PROJECT,
    "entity"            : WANDB_ENTITY,
    "epochs"            : 5,
    "batch_size"        : 32,
    "hidden_nodes"      : 32,
    "hidden_layers"     : 4,
    "weight_init"       : "xavier",
    "act_func"          : "relu",
    "reg"               : 0.0,
    "loss_func"         : "cross_entropy",
    "optimizer"         : "nadam",
    "learning_rate"     : 1e-3,
    "data_set"          : data_set,
  }
  set_name(params3)
  do_plot(params3)


if __name__ == '__main__':
  main("mnist")


