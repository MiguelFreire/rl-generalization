import torch
import os
import wandb
from agents.nature import OriginalNatureAgent
from environments.procgen import make_env
from rlpyt.utils.prog_bar import ProgBarCounter
import pandas

def calculateWinRate(levels_outcome):
    wins = 0
    for level in levels_outcome:
        if level:
            wins += 1
    return wins / len(levels_outcome) * 100


def evaluate_in_training(agent, num_levels=500, seed=42069):
    env = make_env(num_levels=1, start_level=0, seed=seed)
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

def evaluate_in_testing(agent, num_levels=5000, start_level=400000, seed=42069):
    env = make_env(num_levels=1, start_level=start_level, seed=seed)
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

def evaluate_generalization(models_to_evaluate):
    data = []
    for m in models_to_evaluate:
        saved_params = torch.load(os.getcwd() + m['model_path'], map_location=torch.device('cpu'));
        agent = OriginalNatureAgent(initial_model_state_dict=saved_params)
        num_levels = m['num_levels']
        print("Evaluating Training - " + str(num_levels) + "Levels")
        train_winrate = evaluate_in_training(agent, num_levels)
        print("Evaluating Testing - " + str(num_levels) + "Levels")
        test_winrate = evaluate_in_testing(agent)
        data.append([train_winrate, test_winrate])

    return pandas.DataFrame(data=data, index=[n['model_name'] for n in models_to_evaluate], columns=['Train', 'Test'])
