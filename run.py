from experiments.base import BaseExperimentNature
import wandb
import os
import sys
import math

if __name__ == "__main__":

  experiments = [
    BaseExperimentNature(name="CoinRun-Attention", num_levels=500, attention='normal'),
    BaseExperimentNature(name="CoinRun-SelfAttention", num_levels=500, attention='self'),
  ]

  for experiment in experiments:
    run = wandb.init(config=experiment.getConfig(), reinit=True, sync_tensorboard=True)
    with run:
      experiment.run()
        
  sys.exit(0)
        
        
