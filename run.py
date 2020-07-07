from experiments.base import BaseExperimentNature
import wandb
import os
import sys
import math

if __name__ == "__main__":

  experiments = [
    BaseExperimentNature(name="NatureCNN-HiddenSize-256", num_levels=200, hidden_sizes=256),
    BaseExperimentNature(name="NatureCNN-HiddenSize-1024", num_levels=200, hidden_sizes=1024),
    BaseExperimentNature(name="NatureCNN-MaxPooling", num_levels=200, max_pooling=True),
  ]

  for i,experiment in enumerate(experiments):
    run = wandb.init(config=experiment.getConfig(), reinit=True, sync_tensorboard=True)
    with run:
      experiment.run(run_ID=i)
        
  sys.exit(0)
        
        
