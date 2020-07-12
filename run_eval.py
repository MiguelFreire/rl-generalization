import wandb
import os
import sys
from evaluation import evaluate_generalization

if __name__ == "__main__":
  os.environ["WANDB_API_KEY"]="35f237a04eb1653cbff636ef3fab002be9f2e624"
  os.environ["WANDB_ENTITY"]="miguelfreire"
  os.environ["WANDB_PROJECT"]="generalization-rl-torch"
  models_to_evaluate = [
    {"model_name":"CoinRunNature-Arch-HiddenSizes-256", "num_levels": 200, "model_path": "/fixing_models/CoinRunNature-Arch-HiddenSize-256/NatureCNN-HiddenSize-256.pt", "hidden_sizes": 256},
    {"model_name":"CoinRunNature-Arch-HiddenSizes-1024", "num_levels": 200, "model_path": "/fixing_models/CoinRunNature-Arch-HiddenSize-1024/NatureCNN-HiddenSize-1024.pt", "hidden_sizes": 1024},
    {"model_name":"CoinRunNature-Arch-MaxPooling", "num_levels": 200, "model_path": "/fixing_models/CoinRunNature-Arch-MaxPooling/NatureCNN-MaxPooling.pt", "max_pooling": True},
  ]

  evaluate_generalization(models_to_evaluate, "CoinRun-Evaluation-Dropout")

  sys.exit(0)
        
        