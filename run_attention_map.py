from experiments.attention_maps import AttentionMaps
import wandb
import os
import sys
import math

from argparse import ArgumentParser

if __name__ == "__main__":

  parser = ArgumentParser()
  
  parser.add_argument('--name', type=str, default="default")
  parser.add_argument('--batchNorm', type=bool, default=False)
  parser.add_argument('--dropout', type=float, default=0.0)
  parser.add_argument('--augment_obs', type=str, default=None)
  parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[512])
  parser.add_argument('--max_pooling', type=bool, default=False)
  parser.add_argument('--arch', type=str, default="original")
  parser.add_argument('--path', type=str)
  parser.add_argument('--obs', type=str)
  
  args = parser.parse_args()
  maps = AttentionMaps(**vars(args))
  maps.run()

  sys.exit(0)
          
