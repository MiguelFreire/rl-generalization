from experiments.experiment import Experiment

class BaseExperimentNature(Experiment):
    def __init__(self, name="CoinRunNatureLevels-default", num_levels=500, batchNorm=False, dropout=0.0):
      self.num_levels = num_levels
      self.batchNorm = batchNorm
      self.dropout = dropout
      self.name = name
    def getConfig(self):
        return {
            "name": self.name,
            "discount": 0.999,
            "lambda": 0.95,
            "timesteps_per_rollout": 256,
            "epochs_per_rollout": 3,
            "minibatches_per_epoch": 8,
            "entropy_bonus": 0.01,
            "ppo_clip": 0.2,
            "learning_rate": 5e-4,
            "workers": 8,
            "envs_per_worker": 64,
            "total_timesteps": 25_000_000,
          
            "dropout": self.dropout,
            "batchNorm": self.batchNorm,
            "num_levels": self.num_levels,
            "model": "nature"
        }
    
