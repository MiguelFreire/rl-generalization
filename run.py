from experiments.base import BaseExperimentNature
import wandb
import os
import sys
import math

if __name__ == "__main__":

  experiments = [
    BaseExperimentNature(name="CoinRunDropout-0.2", num_levels=500, dropout=0.2),
    BaseExperimentNature(name="CoinRunDropout-0.25", num_levels=500, dropout=0.25),
    BaseExperimentNature(name="CoinRunL2-1.0", num_levels=500, l2_penalty=1e-4),
    BaseExperimentNature(name="CoinRunL2-2.5", num_levels=500, l2_penalty=2.5e-4),
    BaseExperimentNature(name="CoinRunEntropy-0.07", num_levels=500, entropy_bonus=0.07),
    BaseExperimentNature(name="CoinRunEntropy-0.1", num_levels=500, entropy_bonus=0.1),
  ]

  for experiment in experiments:
    run = wandb.init(config=experiment.getConfig(), reinit=True, sync_tensorboard=True)
    with run:
      experiment.run()
        
  sys.exit(0)
        
        
