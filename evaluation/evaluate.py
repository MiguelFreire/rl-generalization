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
    levels = [False for i in range(num_levels)]

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
    
    return calculateWinRate(levels)

def evaluate_in_testing(agent, num_levels=5000, start_level=400000, seed=42069, env_name='procgen'):
    print("Evaluating in testing")
    env = make_env(num_levels=1, start_level=start_level, seed=seed, env=env_name)
    levels = [False for i in range(num_levels)]

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

    return calculateWinRate(levels)
  
  
def evaluate(i, agent, num_levels=200, start_level=0, env_name="procgen", q=None):
  if start_level == 0: #evaluate training
    result = (i, evaluate_in_training(agent, num_levels, env_name=env_name))
    if q is not None:
      q.put(result)
    return result
  else:
    result = (i, evaluate_in_testing(agent, start_level=start_level, num_levels=num_levels,  env_name=env_name))
    if q is not None:
      q.put(result)
    return result
    
def evaluate_generalization(m):
    data = []
    wandb.init(name=m['name'])
    
    print("Evaluation " + m['name'] + "\n")
    saved_params = torch.load(os.getcwd() + m['path'])
    
    batchNorm = m['batchNorm'] if "batchNorm" in m else False
    dropout = m['dropout'] if "dropout" in m else 0.0
    data_aug= m['augment_obs'] if "augment_obs" in m else None
    hidden_sizes = m['hidden_sizes'] if "hidden_sizes" in m else [512]
    max_pooling = m['max_pooling'] if "max_pooling" in m else False
    arch = m['arch'] if "arch" in m else "original"
    env = m['env'] if "env" in m else "coinrun"
    impala = arch == 'impala'
    
    model_kwargs = {
        "batchNorm": batchNorm,
        "dropout": dropout,
        "augment_obs": data_aug,
        "hidden_sizes": hidden_sizes,
        "use_maxpool": max_pooling,
        "arch": arch,
    }

    impala_kwargs = {
        "in_channels": [3, 16, 32],
        "out_channels": [16, 32, 32],
        "hidden_size": 512,
    }

    if impala:
        agent = ImpalaAgent(initial_model_state_dict=saved_params, model_kwargs=impala_kwargs)
    else:
        agent = OriginalNatureAgent(initial_model_state_dict=saved_params, model_kwargs=model_kwargs)
    num_levels = m['num_levels']

    dummy_env = make_env(num_levels=1, start_level=0, env=env)
    agent.initialize(dummy_env.spaces, share_memory=True)
    agent.eval_mode(0)

    mp.set_start_method('spawn', force=True)
    q = mp.Queue()

    params = [
          (0, agent, num_levels, 0, env, q),
          (1, agent, 500, 40000, env, q),
          (2, agent, 500, 50000, env, q),
          (3, agent, 500, 60000, env, q)
        ]

    processes = []
    results = []
    for param in params:
      p = mp.Process(target=evaluate, args=param)
      p.start()
      processes.append(p)

    for p in processes:
      p.join()

    for i in params:
      results.append(q.get())
    
    results.sort(key=lambda x: x[0])

    train_winrate = results[0][1]
    test_winrate = np.array([results[1][1], results[2][1], results[3][1]])
    std = np.std(test_winrate)
    avg = np.average(test_winrate)   
    wandb.log({
      "Train": train_winrate,
      "Test 1": test_winrate[0],
      "Test 2": test_winrate[1],
      "Test 3": test_winrate[2],
      "Test Std": std,
      "Test Avg": avg,
    })
    
    return
