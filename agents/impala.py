from rlpyt.agents.pg.categorical import CategoricalPgAgent
from models.impala import ImpalaModel
from rlpyt.agents.pg.atari import AtariMixin
#Agent that uses CNNNature with 2 heads to parametrize policy and value functions
class ImpalaAgent(AtariMixin, CategoricalPgAgent):
      def __init__(self, **kwargs):
        super().__init__(ModelCls=ImpalaModel, **kwargs)