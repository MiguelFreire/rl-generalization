import torch
from rlpyt.agents.pg.categorical import (CategoricalPgAgent, RecurrentCategoricalPgAgent)
from models.nature import (NatureCNNModel, NatureLSTMModel)
from models.nature_attention import NatureAttention, NatureSelfAttention
from rlpyt.agents.pg.atari import AtariMixin
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.agents.base import (AgentStep, BaseAgent, RecurrentAgentMixin,
    AlternatingRecurrentAgentMixin)
from rlpyt.agents.pg.base import AgentInfo, AgentInfoRnn
#Agent that uses CNNNature with 2 heads to parametrize policy and value functions
class OriginalNatureAgent(AtariMixin, CategoricalPgAgent):
      def __init__(self, **kwargs):
        super().__init__(ModelCls=NatureCNNModel, **kwargs)
      
      @torch.no_grad()
      def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
              device=self.device)
        # if is eval mode and rand_conv do monte carlo evaluation
        if not self.model.training and hasattr(self.model, "augment_obs") and self.model.augment_obs == 'rand_conv':
          pi_vector = []
          value_vector = []
          for i in range(10):
            pi, value = self.model(*model_inputs)
            pi_vector.append(pi)
            value_vector.append(value)
          pi_t = torch.stack(pi_vector)
          value_t = torch.stack(value_vector)

          pi = pi_t.sum(axis=0) / pi_t.shape[1]
          value = value_t.sum(axis=0) / value_t.shape[1]
        else:
          pi, value = self.model(*model_inputs) 
          
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

class AttentionNatureAgent(AtariMixin, CategoricalPgAgent):
      def __init__(self, **kwargs):
        super().__init__(ModelCls=NatureAttention, **kwargs)

class SelfAttentionNatureAgent(AtariMixin, CategoricalPgAgent):
      def __init__(self, **kwargs):
        super().__init__(ModelCls=NatureSelfAttention, **kwargs)

class NatureRecurrentAgent(AtariMixin, RecurrentCategoricalPgAgent):
    def __init__(self, **kwargs):
      super().__init__(ModelCls=NatureLSTMModel, **kwargs)