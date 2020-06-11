import wandb
import os
import sys
from evaluation import evaluate_generalization

if __name__ == "__main__":

  models_to_evaluate = [
    {"model_name":"CoinRunNature-Dropout-5", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Dropout-5.pt", "dropout": 0.05},
    {"model_name":"CoinRunNature-Dropout-10", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Dropout-10.pt", "dropout": 0.10},
    {"model_name":"CoinRunNature-Dropout-15", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Dropout-15.pt", "dropout": 0.15},
    {"model_name":"CoinRunNature-Dropout-20", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Dropout-20.pt", "dropout": 0.20},
    {"model_name":"CoinRunNature-Dropout-25", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Dropout-25.pt", "dropout": 0.25},
  ]

  evaluate_generalization(models_to_evaluate, "CoinRun-Evaluation-Dropout")

  sys.exit(0)
        
        
