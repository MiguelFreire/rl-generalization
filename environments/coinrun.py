from rlpyt.envs.gym import GymEnvWrapper
import gym

class CoinRun:
  def __init__(self, num_levels=500, difficulty="easy", paint_vel_info=True, seed=42069):
    self.env = gym.make("procgen:procgen-coinrun-v0", num_levels=num_levels, distribuition_mode=difficulty, rand_seed=seed, paint_vel_info=paint_vel_info)
    
  def make(self):
    return GymEnvWrapper(self.env)