import torch
import os
import wandb
from agents.nature import OriginalNatureAgent
from agents.impala import ImpalaAgent
from experiments.experiment import make_env
from rlpyt.utils.prog_bar import ProgBarCounter
import pandas
import numpy as np
import multiprocessing as mp


def calculateWinRate(levels_outcome):
    wins = 0
    for level in levels_outcome:
        if level:
            wins += 1
    return wins / len(levels_outcome) * 100


def evaluate_in_training(agent, num_levels=500, seed=42069, env_name='procgen'):
    print("Evaluating in training")
    env = make_env(num_levels=1, start_level=0, seed=seed, env=env_name)
    agent.initialize(env.spaces)
    agent.eval_mode(0)
    levels = [False for i in range(num_levels)]

    progress = ProgBarCounter(num_levels)
    prev_action = torch.tensor(0.0, dtype=torch.float) #None
    prev_reward = torch.tensor(0.0, dtype=torch.float) #None
    for j in range(num_levels):
        env = make_env(num_levels=1, start_level=j, seed=seed, env=env_name)
        done = False
        obs, _, _, info = env.step(-1)
        obs = torch.from_numpy(obs).unsqueeze(0)
        
        while True:
          if done:
            if info.prev_level_complete:
              levels[j] = True
            break
          step = agent.step(obs, prev_action, prev_reward)
          obs, rewards, done, info = env.step(step.action)
          obs = torch.from_numpy(obs).unsqueeze(0)
        
        progress.update(j)
    progress.stop()
    
    return calculateWinRate(levels)

def evaluate_in_testing(agent, num_levels=5000, start_level=400000, seed=42069, env_name='procgen'):
    print("Evaluating in testing")
    env = make_env(num_levels=1, start_level=start_level, seed=seed, env=env_name)
    agent.initialize(env.spaces)
    agent.eval_mode(0)
    levels = [False for i in range(num_levels)]

    progress = ProgBarCounter(num_levels)
    prev_action = torch.tensor(0.0, dtype=torch.float) #None
    prev_reward = torch.tensor(0.0, dtype=torch.float) #None
    for j in range(num_levels):
        env = make_env(num_levels=1, start_level=start_level+j, seed=seed, env=env_name)
        done = False
        obs, _, _, info = env.step(-1)
        obs = torch.from_numpy(obs).unsqueeze(0)

        while True:
          if done:
            if info.prev_level_complete:
              levels[j] = True
            break
          step = agent.step(obs, prev_action, prev_reward)
          obs, rewards, done, info = env.step(step.action)
          obs = torch.from_numpy(obs).unsqueeze(0)

        progress.update(j)
    progress.stop()

    return calculateWinRate(levels)
  
  
def evaluate(i, num_levels=200, start_level=0, env_name="procgen", saved_params={}, model_kwargs={}, impala=False):
  agent = ImpalaAgent(initial_model_state_dict=saved_params, model_kwargs=model_kwargs) if impala else OriginalNatureAgent(initial_model_state_dict=saved_params, model_kwargs=model_kwargs
  if start_level == 0: #evaluate training
    return (i, evaluate_in_training(agent, num_levels, env_name=env_name))
  else:
    return (i, evaluate_in_testing(agent, start_level=start_level, num_levels=num_levels,  env_name=env_name))
    
def evaluate_generalization(m, impala=False):
    data = []
    wandb.init(name=m['name'])

    
    print("Evaluation " + m['name'] + "\n")
    saved_params = torch.load(os.getcwd() + m['path'])
    
    batchNorm = m['batchNorm'] if "batchNorm" in m else False
    dropout = m['dropout'] if "dropout" in m else 0.0
    data_aug= m['data_aug'] if "data_aug" in m else None
    hidden_sizes = m['hidden_sizes'] if "hidden_sizes" in m else [512]
    max_pooling = m['max_pooling'] if "max_pooling" in m else False
    arch = m['arch'] if "arch" in m else "original"
    env = m['env'] if "env" in m else "coinrun"

    model_kwargs = {
        "batchNorm": batchNorm,
        "dropout": dropout,
        "augment_obs": data_aug,
        "hidden_sizes": hidden_sizes,
        "use_maxpool": max_pooling,
        "arch": arch,
    } if not impala else {
        "in_channels": [3, 16, 32],
        "out_channels": [16, 32, 32],
        "hidden_sizes": 256,
    }

    num_levels = m['num_levels']
    
    with mp.Pool(mp.cpu_count()) as pool:
        params = [
          (0, num_levels, 0, env, saved_params, model_kwargs, impala),
          (1, 5000, 40000, env, saved_params, model_kwargs, impala),
          (2, 5000, 50000, env, saved_params, model_kwargs, impala),
          (3, 5000, 60000, env, saved_params, model_kwargs, impala)
        ]
    
        results = [pool.apply_async(evaluate, p) for p in params]

        r = [res.get() for res in results]
        r.sort(key=lambda x: x[0]) #sort just to be sure
        train_winrate = r[0]
        test_winrate = np.array([r[1], r[2], r[3]])
        std = np.std(test_winrate)
        avg = np.average(test_winrate)   
        wandb.log({
          "Train": train_winrate,
          "Test 1": test_winrate1,
          "Test 2": test_winrate2,
          "Test 3": test_winrate3,
          "Test Std": std,
          "Test Avg": avg,
        })
    
    return
