import wandb
import os
import sys
from evaluation.evaluate import evaluate_generalization

from argparse import ArgumentParser

if __name__ == "__main__":
  parser = ArgumentParser()

  parser.add_argument('--name', type=str, default="Default Name")
  parser.add_argument('--path', type=str)
  parser.add_argument('--num_levels', type=int, default=200)
  parser.add_argument('--batchNorm', type=bool, default=False)
  parser.add_argument('--dropout', type=float, default=0.0)
  parser.add_argument('--augment_obs', type=str, default=None)
  parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[512])
  parser.add_argument('--max_pooling', type=bool, default=False)
  parser.add_argument('--arch', type=str, default="original")
  parser.add_argument('--env', type=str, default="coinrun")

  args = parser.parse_args()

  model_to_evaluate = vars(args)

  evaluate_generalization(model_to_evaluate)

  sys.exit(0)
        
        