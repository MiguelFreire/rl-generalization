import torch

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from agents.impala import ImpalaAgent
from rlpyt.algos.pg.ppo import PPO
from gym import Wrapper
from environments.procgen import ProcgenWrapper

import gym
import wandb

def make_env(*args, **kwargs):
    num_levels = kwargs['num_levels'] 
    difficulty="easy"
    paint_vel_info=True
    seed=42069
    env = gym.make("procgen:procgen-coinrun-v0", num_levels=num_levels, distribution_mode=difficulty, rand_seed=seed, paint_vel_info=paint_vel_info)
    return ProcgenWrapper(env)

class ImpalaExperiment:
    def __init__(self, name="CoinRunImpala-Default", num_levels=500, learning_rate=5e-4, in_channels=[3,16,32], out_channels=[16,32,32]):
      self.name = name
      self.learning_rate = learning_rate
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.num_levels = num_levels
    
    def getConfig(self):
        return {
            "name": self.name,
            "discount": 0.999,
            "lambda": 0.95,
            "timesteps_per_rollout": 256,
            "epochs_per_rollout": 3,
            "minibatches_per_epoch": 8,
            "entropy_bonus": 0.01,
            "ppo_clip": 0.2,
            "learning_rate": self.learning_rate,
            "workers": 8,
            "envs_per_worker": 64,
            "total_timesteps": 25_000_000,
            "dropout": 0.0,
            "batchNorm": False,
            "num_levels": self.num_levels,
            "model": "impala"
            "out_channels": self.out_channels,
            "in_channels": self.in_channels,
            "hidden_size": 512,
        }
    

    def run(self):
        config = self.getConfig()
        sampler = CpuSampler(
                      EnvCls=make_env,
                      env_kwargs={"num_levels": config["num_levels"]},
                      batch_T=256,
                      batch_B=8,
                      max_decorrelation_steps=0)
        
        optim_args = dict(weight_decay=config["l2_penalty"]) if "l2_penalty" in config else None

        algo = PPO(discount=config["discount"], entropy_loss_coeff=config["entropy_bonus"],
            gae_lambda=config["lambda"], minibatches=config["minibatches_per_epoch"],
            epochs=config["epochs_per_rollout"], ratio_clip=config["ppo_clip"],
            learning_rate=config["learning_rate"], normalize_advantage=True, optim_kwargs=optim_args)
        agent = ImpalaAgent(model_kwargs={"in_channels": config["in_channels"], "out_channels": config["out_channels"], "hidden_size": config["hidden_size"]})
        
        affinity = dict(cuda_idx=0, workers_cpus=list(range(config["workers"])))

        runner = MinibatchRl(
                  algo=algo,
                  agent=agent,
                  sampler=sampler,
                  n_steps=25e6,
                  log_interval_steps=500,
                  affinity=affinity,
                  seed=42069)
        log_dir="./logs"
        name=config["name"]
        run_ID=name
        with logger_context(log_dir, run_ID, name, config, use_summary_writer=True):
            runner.train()
        torch.save(agent.state_dict(), "./" + name + ".pt")
        wandb.save("./" + name + ".pt")
      
  
