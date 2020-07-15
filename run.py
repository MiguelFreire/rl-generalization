from experiments.base import BaseExperimentNature
import wandb
import os
import sys
import math

if __name__ == "__main__":

  experiments = [
    BaseExperimentNature(name="NatureCNN-200", num_levels=200),
    BaseExperimentNature(name="NatureCNN-Arch-Depth+1", num_levels=200, arch="depth+1"),
    BaseExperimentNature(name="NatureCNN-Arch-Depth+2", num_levels=200, arch="depth+2"),
    BaseExperimentNature(name="NatureCNN-Arch-Channels/2", num_levels=200, arch="channels/2"),
    BaseExperimentNature(name="NatureCNN-Arch-Channels*2", num_levels=200, arch="channels*2"),
  ]

  for i,experiment in enumerate(experiments):
    run = wandb.init(config=experiment.getConfig(), reinit=True, sync_tensorboard=True)
    with run:
      experiment.run(run_ID=i)

  sys.exit(0)
        
        
