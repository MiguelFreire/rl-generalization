import gym
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.collections import  is_namedtuple_class
from rlpyt.envs.gym import build_info_tuples, info_to_nt
from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit
from collections import namedtuple

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

      
def make_env(*args, **kwargs):
    num_levels = kwargs['num_levels']
    difficulty="easy"
    paint_vel_info=True
    seed=42069
    env = gym.make("procgen:procgen-coinrun-v0", num_levels=num_levels, distribution_mode=difficulty, rand_seed=seed, paint_vel_info=paint_vel_info)
    return ProcgenWrapper(env)