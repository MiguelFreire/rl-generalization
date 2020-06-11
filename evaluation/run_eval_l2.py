import wandb
import os
import sys
from evaluation import evaluate_generalization

if __name__ == "__main__":

  models_to_evaluate = [
    {"model_name":"CoinRunNature-l2-0.1", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-L2-0.1.pt"},
    {"model_name":"CoinRunNature-l2-0.25", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-L2-0.25.pt"},
    {"model_name":"CoinRunNature-l2-0.5", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-L2-0.5.pt"},
    {"model_name":"CoinRunNature-l2-1.0", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-L2-1.0.pt"},
    {"model_name":"CoinRunNature-l2-2.5", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-L2-2.5.pt"},
  ]

  evaluate_generalization(models_to_evaluate, "CoinRun-Evaluation-L2")

  sys.exit(0)
        
        
