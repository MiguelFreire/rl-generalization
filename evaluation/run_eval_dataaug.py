import wandb
import os
import sys
from evaluation import evaluate_generalization

if __name__ == "__main__":

  models_to_evaluate = [
    {"model_name":"CoinRunNature-ColorJitter", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-ColorJitter.pt" , "data_aug": "jitter"},
    {"model_name":"CoinRunNature-Cutout", "num_levels": 500, "model_path": "../saved_models/CoinRunNature-Cutout.pt", "data_aug": "cutout"},
  ]

  evaluate_generalization(models_to_evaluate, "CoinRun-Evaluation-DataAug")

  sys.exit(0)
        
        
