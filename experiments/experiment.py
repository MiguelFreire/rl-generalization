import torch

#from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from agents.nature import OriginalNatureAgent, AttentionNatureAgent, SelfAttentionNatureAgent, NatureRecurrentAgent
from rlpyt.samplers.parallel.gpu.collectors import GpuResetCollector
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
    
    if "env" in kwargs:
        env_name = "procgen:procgen-"+kwargs['env']+"-v0"
    else:
        env_name = "procgen:procgen-coinrun-v0"
    
    start_level = kwargs["start_level"] if "start_level" in kwargs else 0

    env = gym.make(env_name, num_levels=num_levels, start_level=start_level, distribution_mode=difficulty, paint_vel_info=paint_vel_info)
    return ProcgenWrapper(env)

class Experiment:
    def getConfig(self):
        return {
            "name": "CoinRunNature500-default",
            "discount": 0.999,
            "lambda": 0.95,
            "timesteps_per_rollout": 256,
            "epochs_per_rollout": 3,
            "minibatches_per_epoch": 8,
            "entropy_bonus": 0.01,
            "ppo_clip": 0.2,
            "learning_rate": 5e-4,
            "workers": 8,
            "envs_per_worker": 64,
            "total_timesteps": 30_000_000,
            "dropout": 0.0,
            "batchNorm": False,
            "num_levels": 500,
            "model": "nature",
            "augment_obs": "",
            "attention": None,
            "maxpool": False,
            "hidden_sizes": 512,
            "arch": "original",
            "env": "coinrun",
        }
    

    def run(self, run_ID=0):
        config = self.getConfig()
        sampler = GpuSampler(
                      EnvCls=make_env,
                      env_kwargs={"num_levels": config["num_levels"], "env": config['env']},
                      CollectorCls=GpuResetCollector,
                      batch_T=256,
                      batch_B=config["envs_per_worker"],
                      max_decorrelation_steps=1000)
        
        optim_args = dict(weight_decay=config["l2_penalty"]) if "l2_penalty" in config else None

        algo = PPO(
                value_loss_coeff=0.5,
                clip_grad_norm=0.5,
                discount=config["discount"], 
                entropy_loss_coeff=config["entropy_bonus"],
                gae_lambda=config["lambda"], 
                minibatches=config["minibatches_per_epoch"],
                epochs=config["epochs_per_rollout"], 
                ratio_clip=config["ppo_clip"],
                learning_rate=config["learning_rate"], 
                normalize_advantage=True, 
                optim_kwargs=optim_args)
        
        if config["arch"] == 'impala':
            agent = ImpalaAgent(model_kwargs={"in_channels": [3,16,32], "out_channels": [16,32,32], "hidden_size": 256})
        elif config["arch"] == 'lstm':
            agent = NatureRecurrentAgent(model_kwargs={"hidden_sizes": [512], "lstm_size": 256})
        else:
            agent = OriginalNatureAgent(model_kwargs={"batchNorm": config["batchNorm"], "dropout": config["dropout"], "augment_obs": config["augment_obs"], "use_maxpool": config["maxpool"], "hidden_sizes": config["hidden_sizes"], "arch": config["arch"]})
        
        affinity = dict(cuda_idx=0, workers_cpus=list(range(8)))

        runner = MinibatchRl(
                  algo=algo,
                  agent=agent,
                  sampler=sampler,
                  n_steps=config["total_timesteps"],
                  log_interval_steps=500,
                  affinity=affinity)
        log_dir="./logs"
        name=config["name"]
        with logger_context(log_dir, run_ID, name, config, use_summary_writer=True, override_prefix=False):
            runner.train()
        torch.save(agent.state_dict(), "./" + name + ".pt")
        wandb.save("./" + name + ".pt")
      
  
