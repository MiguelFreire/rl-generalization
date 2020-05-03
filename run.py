from experiments.base import BaseExperimentNature
import wandb
import os
import sys

if __name__ == "__main__":

  experiments = [
    BaseExperimentNature(name="CoinRunNatureLevels-100", num_levels=100),
    BaseExperimentNature(name="CoinRunNatureLevels-500",num_levels=500),
    BaseExperimentNature(name="CoinRunNatureLevels-1000",num_levels=1000),
    BaseExperimentNature(name="CoinRunNatureLevels-10000",num_levels=10000),
    BaseExperimentNature(name="CoinRunNatureLevels-15000",num_levels=15000),
    BaseExperimentNature(name="CoinRunNatureLevels-batchnorm",num_levels=500, batchNorm=True),
  ]

  for experiment in experiments:
    run = wandb.init(reinit=True)
    with run:
      experiment.run()
        
  sys.exit(0)
        
        