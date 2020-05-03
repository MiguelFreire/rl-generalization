from rlpyt.agents.pg.categorical import CategoricalPgAgent
from models.nature import NatureCNNModel
from rlpyt.agents.pg.atari import AtariMixin
#Agent that uses CNNNature with 2 heads to parametrize policy and value functions
class OriginalNatureAgent(AtariMixin, CategoricalPgAgent):
      def __init__(self, **kwargs):
        super().__init__(ModelCls=NatureCNNModel, **kwargs)