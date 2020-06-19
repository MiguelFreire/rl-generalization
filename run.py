from experiments.base import BaseExperimentNature
import wandb
import os
import sys
import math

if __name__ == "__main__":

  experiments = [
    BaseExperimentNature(name="NatureCNN-Levels200", num_levels=200),
    BaseExperimentNature(name="NatureCNN-BatchNorm", num_levels=200, batchNorm=True),
    BaseExperimentNature(name="NatureCNN-L2-0.1", num_levels=200, l2_penalty=0.1e-4),
    BaseExperimentNature(name="NatureCNN-L2-0.25", num_levels=200, l2_penalty=0.25e-4),
    BaseExperimentNature(name="NatureCNN-L2-0.5", num_levels=200, l2_penalty=0.5e-4),
    BaseExperimentNature(name="NatureCNN-L2-1.0", num_levels=200, l2_penalty=1.0e-4),
    BaseExperimentNature(name="NatureCNN-L2-2.5", num_levels=200, l2_penalty=2.5e-4),
    BaseExperimentNature(name="NatureCNN-Entropy-0", num_levels=200, entropy_bonus=0),
    BaseExperimentNature(name="NatureCNN-Entropy-0.02", num_levels=200, entropy_bonus=0.02),
    BaseExperimentNature(name="NatureCNN-Entropy-0.05", num_levels=200, entropy_bonus=0.05),
    BaseExperimentNature(name="NatureCNN-Entropy-0.07", num_levels=200, entropy_bonus=0.07, num_steps=35_000_000),
    BaseExperimentNature(name="NatureCNN-Entropy-0.1", num_levels=200, entropy_bonus=0.1, num_steps=35_000_000),
    BaseExperimentNature(name="NatureCNN-Dropout-0.05", num_levels=200, dropout=0.05),
    BaseExperimentNature(name="NatureCNN-Dropout-0.10", num_levels=200, dropout=0.1),
    BaseExperimentNature(name="NatureCNN-Dropout-0.15", num_levels=200, dropout=0.15),
    BaseExperimentNature(name="NatureCNN-Dropout-0.20", num_levels=200, dropout=0.20, num_steps=35_000_000),
    BaseExperimentNature(name="NatureCNN-Dropout-0.25", num_levels=200, dropout=0.25, num_steps=35_000_000),
  ]

  for i,experiment in enumerate(experiments):
    run = wandb.init(config=experiment.getConfig(), reinit=True, sync_tensorboard=True)
    with run:
      experiment.run(run_ID=i)
        
  sys.exit(0)
        
        
