import wandb
import os
import sys
from evaluation import evaluate_generalization

if __name__ == "__main__":

  models_to_evaluate = [
    {"model_name":"CoinRunNature-entropy-0", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Entropy-0.pt"},
    {"model_name":"CoinRunNature-entropy-0.02", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Entropy-0.02.pt"},
    {"model_name":"CoinRunNature-entropy-0.05", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Entropy-0.05.pt"},
    {"model_name":"CoinRunNature-entropy-0.07", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Entropy-0.07.pt"},
    {"model_name":"CoinRunNature-entropy-0.1", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Entropy-0.1.pt"},
  ]

  evaluate_generalization(models_to_evaluate, "CoinRun-Evaluation-Entropy")

  sys.exit(0)
        
        
