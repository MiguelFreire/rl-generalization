from rlpyt.agents.pg.categorical import CategoricalPgAgent
from models.nature import NatureCNNModel
from models.nature_attention import NatureAttention, NatureSelfAttention
from rlpyt.agents.pg.atari import AtariMixin
#Agent that uses CNNNature with 2 heads to parametrize policy and value functions
class OriginalNatureAgent(AtariMixin, CategoricalPgAgent):
      def __init__(self, **kwargs):
        super().__init__(ModelCls=NatureCNNModel, **kwargs)

class AttentionNatureAgent(AtariMixin, CategoricalPgAgent):
      def __init__(self, **kwargs):
        super().__init__(ModelCls=NatureAttention, **kwargs)

class SelfAttentionNatureAgent(AtariMixin, CategoricalPgAgent):
      def __init__(self, **kwargs):
        super().__init__(ModelCls=NatureSelfAttention, **kwargs)