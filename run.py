from experiments.base import BaseExperimentNature
import wandb
import os
import sys
import math

from argparse import ArgumentParser

if __name__ == "__main__":

  parser = ArgumentParser()
  
  parser.add_argument('--name', type=str, default="Default Name")
  parser.add_argument('--num_levels', type=int, default=200)
  parser.add_argument('--batchNorm', type=bool, default=False)
  parser.add_argument('--dropout', type=float, default=0.0)
  parser.add_argument('--l2_penalty', type=float, default=0)
  parser.add_argument('--entropy_bonus', type=float, default=0.01)
  parser.add_argument('--augment_obs', type=str, default=None)
  parser.add_argument('--attention', type=str, default=None)
  parser.add_argument('--num_steps', type=int, default=25_000_000)
  parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[512])
  parser.add_argument('--max_pooling', type=bool, default=False)
  parser.add_argument('--arch', type=str, default="original")
  parser.add_argument('--env', type=str, default="coinrun")
  
  args = parser.parse_args()
  experiment = BaseExperimentNature(**vars(args))
  config = experiment.getConfig()
  run = wandb.init(name=config['name'], config=config, reinit=False, sync_tensorboard=True)
  experiment.run()

  sys.exit(0)
        
        
