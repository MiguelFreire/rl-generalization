import wandb
import os
import sys
from evaluation import evaluate_generalization

if __name__ == "__main__":

  models_to_evaluate = [
    {"model_name":"CoinRunNature-Levels-100", "num_levels": 100, "model_path": "../saved_models/CoinRunNature-Levels-100.pt"},
    {"model_name":"CoinRunNature-Levels-500", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Levels-500.pt"},
    {"model_name":"CoinRunNature-Levels-1000", "num_levels": 1000, "model_path": "../saved_models/CoinRunNature-Levels-1000.pt"},
    {"model_name":"CoinRunNature-Levels-10000", "num_levels": 10000, "model_path": "../saved_models/CoinRunNature-Levels-10000.pt"},
    {"model_name":"CoinRunNature-Levels-15000", "num_levels": 15000, "model_path": "../saved_models/CoinRunNature-Levels-15000.pt"},
  ]

  evaluate_generalization(models_to_evaluate, "CoinRun-Evaluation-Levels")

  sys.exit(0)
        
        
