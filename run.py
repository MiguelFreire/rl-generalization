from experiments.base import BaseExperimentNature
import wandb
import os
import sys
import math

if __name__ == "__main__":

  experiments = [
    BaseExperimentNature(name="NatureCNN-RandConv", num_levels=200, augment_obs='rand_conv'),
    BaseExperimentNature(name="NatureCNN-ColorJitter", num_levels=200, augment_obs='jitter'),
    BaseExperimentNature(name="NatureCNN-Cutout", num_levels=200, augment_obs='cutout'),
  ]

  for i,experiment in enumerate(experiments):
    run = wandb.init(config=experiment.getConfig(), reinit=True, sync_tensorboard=True)
    with run:
      experiment.run(run_ID=i)
        
  sys.exit(0)
        
        
