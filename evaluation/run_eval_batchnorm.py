import wandb
import os
import sys
from evaluation import evaluate_generalization

if __name__ == "__main__":

  models_to_evaluate = [
    {"model_name":"CoinRunNature-BatchNorm", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-BatchNorm.pt", "batchNorm": True},
  ]

  evaluate_generalization(models_to_evaluate, "CoinRun-Evaluation-BatchNorm")

  sys.exit(0)
        
        
