from typing import Dict, Optional
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from autonomous_driving_predictor.model_components.trajectory_decoder import TrajectoryDecoder
from autonomous_driving_predictor.model_components.map_context_decoder import MapContextDecoder


class MotionDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_token: Dict,
                 token_data: Dict,
                 use_intention=False,
                 token_size=512,
                 use_lane_tokens=False,
                 use_relational_attention=False,
                 lane_token_embedding_dim=64) -> None:
        super(MotionDecoder, self).__init__()
        
        self.use_lane_tokens = use_lane_tokens
        self.use_relational_attention = use_relational_attention
        self.lane_token_embedding_dim = lane_token_embedding_dim
        
        self.map_encoder = MapContextDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            map_token=map_token
        )
        self.agent_encoder = TrajectoryDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            token_size=token_size,
            token_data=token_data
        )
        self.map_enc = None

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder(data, map_enc)
        return {**map_enc, **agent_enc}

    def inference(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder.inference(data, map_enc)
        return {**map_enc, **agent_enc}

    def inference_no_map(self, data: HeteroData, map_enc) -> Dict[str, torch.Tensor]:
        agent_enc = self.agent_encoder.inference(data, map_enc)
        return {**map_enc, **agent_enc}
