import torch
import os
import wandb
from agents.nature import OriginalNatureAgent
from agents.impala import ImpalaAgent
from experiments.experiment import make_env
from rlpyt.utils.prog_bar import ProgBarCounter
import pandas
import numpy as np

def calculateWinRate(levels_outcome):
    wins = 0
    for level in levels_outcome:
        if level:
            wins += 1
    return wins / len(levels_outcome) * 100


def evaluate_in_training(agent, num_levels=500, seed=42069, env='procgen'):
    env = make_env(num_levels=1, start_level=0, seed=seed, env=env)
    agent.initialize(env.spaces)
    agent.eval_mode(0)
    levels = [False for i in range(num_levels)]
    obs = env.reset()
    obs = torch.from_numpy(obs).unsqueeze(0)
    progress = ProgBarCounter(num_levels)
    prev_action = torch.tensor(0.0, dtype=torch.float) #None
    prev_reward = torch.tensor(0.0, dtype=torch.float) #None
    for j in range(num_levels):
        done = False
        while not done:
            step = agent.step(obs, prev_action, prev_reward)
            obs, rewards, done, info = env.step(step.action)
            obs = torch.from_numpy(obs).unsqueeze(0)
            if done:
                if info.level_complete:
                    levels[j] = True
                env = make_env(num_levels=1, start_level=j, seed=seed)
                obs = env.reset()
                obs = torch.from_numpy(obs).unsqueeze(0)
        progress.update(j)
    progress.stop()
    
    return calculateWinRate(levels)

def evaluate_in_testing(agent, num_levels=5000, start_level=400000, seed=42069, env='procgen'):
    env = make_env(num_levels=1, start_level=start_level, seed=seed, env=env)
    agent.initialize(env.spaces)
    agent.eval_mode(0)
    levels = [False for i in range(num_levels)]
    obs = env.reset()
    obs = torch.from_numpy(obs).unsqueeze(0)
    progress = ProgBarCounter(num_levels)

    prev_action = torch.tensor(0.0, dtype=torch.float) #None
    prev_reward = torch.tensor(0.0, dtype=torch.float) #None
    for j in range(num_levels):
        done = False
        while not done:
            step = agent.step(obs, prev_action, prev_reward)
            obs, rewards, done, info = env.step(step.action)
            obs = torch.from_numpy(obs).unsqueeze(0)
            if done:
                if info.level_complete:
                    levels[j] = True
                env = make_env(num_levels=1, start_level=start_level+j, seed=seed)
                obs = env.reset()
                obs = torch.from_numpy(obs).unsqueeze(0)
        progress.update(j)
    progress.stop()
    
    return calculateWinRate(levels)

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

    model_kwargs = {
        "batchNorm": batchNorm,
        "dropout": dropout,
        "augment_obs": data_aug,
        "hidden_sizes": hidden_sizes,
        "max_pooling": max_pooling,
        "arch": arch,
    }

    if impala:
        agent = ImpalaAgent(initial_model_state_dict=saved_params)
    else:
        agent = OriginalNatureAgent(initial_model_state_dict=saved_params, model_kwargs=model_kwargs)
    num_levels = m['num_levels']
    print("Evaluating Training - " + str(num_levels) + "Levels \n")
    train_winrate = evaluate_in_training(agent, num_levels, env=m['env'])
    print("Evaluating Testing 1 \n")
    test_winrate1 = evaluate_in_testing(agent, start_level=40000, num_levels=500,  env=m['env'])
    print("Evaluating Testing 2 \n")
    test_winrate2 = evaluate_in_testing(agent, start_level=50000, num_levels=500, env=m['env'])
    print("Evaluating Testing 3 \n")
    test_winrate3 = evaluate_in_testing(agent, start_level=60000, num_levels=500, env=m['env'])
    
    test_winrate = np.array([test_winrate1, test_winrate2, test_winrate3])

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

    #return pandas.DataFrame(data=data, index=[n['model_name'] for n in models_to_evaluate], columns=['Train', 'Test_1', 'Test_2', 'Test_3', 'Test_Std', 'Test_Avg'])
    return
