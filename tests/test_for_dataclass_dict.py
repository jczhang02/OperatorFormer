import torch
from torch import Tensor, nn
from rich import print

from dataclasses import asdict, fields

from models.components.encoder import CrossAttentionEncoder, InputEncoder, QueryEncoder
from models.components.dataclass_config import InputEncoderConfig, QueryEncoderConfig, CrossAttentionEncoderConfig
from models.components.pdeformer import PDEFormer

input_encoder_config = InputEncoderConfig()


input_encoder_config_dict = asdict(input_encoder_config)

model = PDEFormer(input_encoder_config)
