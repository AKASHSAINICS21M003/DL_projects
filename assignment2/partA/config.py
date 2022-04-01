#! /usr/bin/env python3

import yaml


class Config:
  @staticmethod
  def from_yaml(filepath):
    with open(filepath, 'r') as fp:
      dataMap = yaml.safe_load(fp)
    return Config.from_dict(dataMap)

  @staticmethod
  def from_dict(dataMap):
    config = Config()
    for name, value in dataMap.items():
      setattr(config, name, value)

