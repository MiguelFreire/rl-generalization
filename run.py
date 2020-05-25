from experiments.impala import ExperimentImpala
import wandb
import os
import sys
import math

if __name__ == "__main__":

  experiments = [
    ExperimentImpala(name="CoinRunImpala-16", num_levels=500, in_channels=[3], out_channels=[16], learning_rate=5e-4),
    ExperimentImpala(name="CoinRunImpala-16-32", num_levels=500, in_channels=[3,16], out_channels=[16,32], learning_rate=(1/math.sqrt(3))*5e-4),
    ExperimentImpala(name="CoinRunImpala-16-32-32", num_levels=500, in_channels=[3,16,32], out_channels=[16,32,32], learning_rate=(1/math.sqrt(5))*5e-4),
  ]

  for experiment in experiments:
    run = wandb.init(config=experiment.getConfig(), reinit=True, sync_tensorboard=True)
    with run:
      experiment.run()
        
  sys.exit(0)
        
        
