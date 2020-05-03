import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.pg.atari_ff_model import AtariFfModel
import numpy as np
import torch.nn.functional as F
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.agents.pg.atari import AtariMixin
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.algos.pg.ppo import PPO
from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit
from collections import namedtuple

from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.collections import  is_namedtuple_class
from rlpyt.envs.gym import build_info_tuples, info_to_nt
import gym
import os
import wandb


os.environ["WANDB_API_KEY"]="35f237a04eb1653cbff636ef3fab002be9f2e624"
os.environ["WANDB_ENTITY"]="miguelfreire"
os.environ["WANDB_PROJECT"]="generalization-rl-torch"


class ProcgenWrapper(GymEnvWrapper):
    def __init__(self, env,
          act_null_value=0, obs_null_value=0, force_float32=True):
      super().__init__(env)
      o = self.env.reset()
      o, r, d, info = self.env.step(self.env.action_space.sample())
      env_ = self.env
      time_limit = isinstance(self.env, TimeLimit)
      while not time_limit and hasattr(env_, "env"):
          env_ = env_.env
          time_limit = isinstance(self.env, TimeLimit)
      if time_limit:
          info["timeout"] = False  # gym's TimeLimit.truncated invalid name.
      self._time_limit = time_limit
      self.action_space = GymSpaceWrapper(
          space=self.env.action_space,
          name="act",
          null_value=act_null_value,
          force_float32=force_float32,
      )
      self.observation_space = GymSpaceWrapper(
          space=self.env.observation_space,
          name="obs",
          null_value=obs_null_value,
          force_float32=force_float32,
      )
      w = self.observation_space.space.shape[1]
      h = self.observation_space.space.shape[0]
      c = self.observation_space.space.shape[2]
      self.observation_space.space.shape = (c, h, w)
      build_info_tuples(info)

    
    def step(self, action):
        """Reverts the action from rlpyt format to gym format (i.e. if composite-to-
        dictionary spaces), steps the gym environment, converts the observation
        from gym to rlpyt format (i.e. if dict-to-composite), and converts the
        env_info from dictionary into namedtuple."""
        a = self.action_space.revert(action)
        o, r, d, info = self.env.step(a)
        obs = self.observation_space.convert(o.transpose((2, 0, 1)))
        if self._time_limit:
            if "TimeLimit.truncated" in info:
                info["timeout"] = info.pop("TimeLimit.truncated")
            else:
                info["timeout"] = False
        info = info_to_nt(info)
        if isinstance(r, float):
            r = np.dtype("float32").type(r)  # Scalar float32.
        return EnvStep(obs, r, d, info)
    def reset(self):
        """Returns converted observation from gym env reset."""
        return self.observation_space.convert(self.env.reset().transpose((2, 0, 1)))

class OriginalNatureCNNModel(torch.nn.Module):
    def __init__(self, image_shape, output_size):
        super().__init__()
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=[32,64,64],
            kernel_sizes=[8,4,3],
            strides=[4,3,1],
            paddings=[0,0,1],
            use_maxpool=False,
            hidden_sizes=512,
        )
        #policy head
        
        self.pi = torch.nn.Linear(self.conv.output_size, output_size) 
        #value function head 
        self.value = torch.nn.Linear(self.conv.output_size, 1)
        #reset weights just like nature paper
        self.init_weights()
        
    def init_weights(self):
        #orthogonal initialization with gain of sqrt(2)
        def weights_initializer(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
        
        self.apply(weights_initializer)
            
            
    def forward(self, image, prev_action, prev_reward):
        #input normalization, cast to float then grayscale it
        img = image.type(torch.float)
        img = img.mul_(1. / 255)
        
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        
        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        #T -> transition
        #B -> batch_size?
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v
  
  #Agent that uses CNNNature with 2 heads to parametrize policy and value functions
class OriginalNatureAgent(AtariMixin, CategoricalPgAgent):
    def __init__(self, ModelCls=AtariFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


def make_env(*args, **kwargs):
    num_levels = kwargs['num_levels'] 
    difficulty="easy"
    paint_vel_info=True
    seed=42069
    env = gym.make("procgen:procgen-coinrun-v0", num_levels=num_levels, distribution_mode=difficulty, rand_seed=seed, paint_vel_info=paint_vel_info)
    return ProcgenWrapper(env)
  
def makeDefaultPPOExperiment(config):
    return PPO(discount=config["discount"], entropy_loss_coeff=config["entropy_bonus"],
            gae_lambda=config["lambda"], minibatches=config["minibatches_per_epoch"],
            epochs=config["epochs_per_rollout"], ratio_clip=config["ppo_clip"],
            learning_rate=config["learning_rate"], normalize_advantage=True)

if __name__ == "__main__":

  config = {
      "discount": 0.999,
      "lambda": 0.95,
      "timesteps_per_rollout": 256,
      "epochs_per_rollout": 3,
      "minibatches_per_epoch": 8,
      "entropy_bonus": 0.01,
      "ppo_clip": 0.2,
      "learning_rate": 5e-4,
      "workers": 4,
      "envs_per_worker": 64,
      "total_timesteps": 25_000_000,
  }


  sampler = CpuSampler(
      EnvCls=make_env,
      env_kwargs=dict(),
      batch_T=256,
      batch_B=8,
      max_decorrelation_steps=0)

  algo = makeDefaultPPOExperiment()
  agent = OriginalNatureAgent()


  affinity = dict(cuda_idx=0, workers_cpus=list(range(8)))

  runner = MinibatchRl(
          algo=algo,
          agent=agent,
          sampler=sampler,
          n_steps=25e6,
          log_interval_steps=1e5,
          affinity=affinity,
          seed=42069,
      )


  wandb.init(config=config, sync_tensorboard=True)

  log_dir="./logs"
  run_ID=1
  name="test"
  with logger_context(log_dir, run_ID, name, config, use_summary_writer=True):
          runner.train()
      
  torch.save(agent.state_dict(), "./test_run.pt")
  wandb.save("./test_run.pt")
  sys.exit(0)