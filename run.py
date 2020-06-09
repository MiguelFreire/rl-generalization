from experiments.base import BaseExperimentNature
import wandb
import os
import sys
import math

if __name__ == "__main__":

  experiments = [
    BaseExperimentNature(name="CoinRunColorJitter", num_levels=500, augment_obs='jitter'),
    BaseExperimentNature(name="CoinRunRandomConv", num_levels=500, augment_obs='rand_conv'),
    BaseExperimentNature(name="CoinRunCutout", num_levels=500, augment_obs='cutout'),
  ]

  for experiment in experiments:
    run = wandb.init(config=experiment.getConfig(), reinit=True, sync_tensorboard=True)
    with run:
      experiment.run()
        
  sys.exit(0)
        
        
