import wandb
import os
import sys
from evaluation import evaluate_generalization

if __name__ == "__main__":

  models_to_evaluate = [
    {"model_name":"CoinRunImpala-16", "num_levels": 500, "model_path": "../saved_models/CoinRunImpala-16.pt"},
    {"model_name":"CoinRunImpala-16-32", "num_levels": 500, "model_path": "../saved_models/CoinRunImpala-16-32.pt"},
    {"model_name":"CoinRunImpala-16-32-32", "num_levels": 500, "model_path": "../saved_models/CoinRunImpala-16-32-32.pt"},
  ]

  evaluate_generalization(models_to_evaluate, "CoinRun-Evaluation-Architecture", impala=True)

  sys.exit(0)
        
        
