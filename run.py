from experiments.base import BaseExperimentNature
import wandb
import os
import sys

if __name__ == "__main__":
  experiments = [
    BaseExperimentNature(name="CoinRunNatureLevels-L2-0.1",num_levels=500, l2_penalty=0.1E-4),
    BaseExperimentNature(name="CoinRunNatureLevels-L2-0.25",num_levels=500, l2_penalty=0.25E-4),
    BaseExperimentNature(name="CoinRunNatureLevels-L2-0.5",num_levels=500, l2_penalty=0.5E-4),
    BaseExperimentNature(name="CoinRunNatureLevels-L2-1.0",num_levels=500, l2_penalty=1E-4),
    BaseExperimentNature(name="CoinRunNatureLevels-L2-2.5",num_levels=500, l2_penalty=2.5E-4),
    BaseExperimentNature(name="CoinRunNatureLevels-entropy-0",num_levels=500, entropy_bonus=0),
    BaseExperimentNature(name="CoinRunNatureLevels-entropy-0.02",num_levels=500, entropy_bonus=0.02),
    BaseExperimentNature(name="CoinRunNatureLevels-entropy-0.05",num_levels=500, entropy_bonus=0.05),
    BaseExperimentNature(name="CoinRunNatureLevels-entropy-0.07",num_levels=500, entropy_bonus=0.07),
    BaseExperimentNature(name="CoinRunNatureLevels-entropy-0.1",num_levels=500, entropy_bonus=0.10),
  ]

  for experiment in experiments:
    run = wandb.init(config=experiment.getConfig(), reinit=True, sync_tensorboard=True)
    with run:
      experiment.run()
        
  sys.exit(0)
        
        
