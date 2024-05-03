from typing import Optional

from torch import Tensor, nn

from .dataclass_config import (
    CrossAttentionEncoderConfig,
    InputEncoderConfig,
    PropagatorDecoderConfig,
    QueryEncoderConfig,
    _unpack,
)
from .decoder import PropagatorDecoder
from .encoder import CrossAttentionEncoder, InputEncoder, QueryEncoder


__all__ = ["OperatorFormer"]


class OperatorFormer(nn.Module):
    def __init__(
        self,
        input_encoder_config: InputEncoderConfig = InputEncoderConfig(),
        query_encoder_config: QueryEncoderConfig = QueryEncoderConfig(),
        crossattention_encoder_config: CrossAttentionEncoderConfig = CrossAttentionEncoderConfig(),
        propagator_decoder_config: PropagatorDecoderConfig = PropagatorDecoderConfig(),
    ) -> None:
        super(OperatorFormer, self).__init__()

        self.input_encoder = InputEncoder(*_unpack(input_encoder_config))
        self.query_encoder = QueryEncoder(*_unpack(query_encoder_config))
        self.crossattention_encoder = CrossAttentionEncoder(*_unpack(crossattention_encoder_config))
        self.decoder = PropagatorDecoder(*_unpack(propagator_decoder_config))

    def forward(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        input_emb: Tensor = self.input_encoder(x, input_pos, mask)
        query_emb: Tensor = self.query_encoder(query_pos)
        z: Tensor = self.crossattention_encoder(query_emb, input_emb, query_pos, input_pos, mask)

        return self.decoder(z)
