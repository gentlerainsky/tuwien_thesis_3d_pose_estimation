import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
from einops import rearrange
from src.modules.lifter_2d_3d.model.graph_mlp.block.graph_frames import Graph
from src.modules.lifter_2d_3d.model.graph_mlp.block.mlp_gcn import Mlp_gcn

class Model(nn.Module):
    def __init__(
            self,
            frames,
            channel,
            depth,
            d_hid,
            token_dim,
            n_joints    
        ):
        super().__init__()
        # self.graph = Graph('hm36_gt', 'spatial', pad=1)
        self.graph = Graph('coco_upperbody', 'spatial', pad=1)
        self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False)
        
        self.embedding = nn.Linear(2*frames, channel)
        self.mlp_gcn = Mlp_gcn(
            depth,
            embed_dim=channel,
            channels_dim=d_hid,
            tokens_dim=token_dim,
            adj=self.A,
            drop_rate=0.10,
            length=n_joints,
            frames=frames
        )
        self.head = nn.Linear(channel, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> b j (c f)').contiguous() # B 17 (2f)

        x = self.embedding(x)       # B 17 512
        x = self.mlp_gcn(x)         # B 17 512
        x = self.head(x)            # B 17 3

        x = rearrange(x, 'b j c -> b 1 j c').contiguous() # B, 1, 17, 3

        return x

