import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from models.simple_vit import Transformer as SimpleTransfomer
from models.simple_vit import posemb_sincos_2d
from models.vit import Transformer


class MilTransformer(nn.Module):
    def __init__(
        self,
        bag_size,
        input_dim,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.2,
        variant="simple",
    ):
        super().__init__()

        self.bag_size = bag_size

        self.pos_embedding = nn.Parameter(torch.randn(1, bag_size, dim))
        self.to_token_embedding = nn.Linear(input_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        if variant == "simple":
            self.transformer = SimpleTransfomer(dim, depth, heads, dim_head, mlp_dim)
        else:
            self.transformer = Transformer(
                dim, depth, heads, dim_head, mlp_dim, dropout=dropout
            )

        self.linear_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        x, patch_positions = x

        x = self.to_token_embedding(x)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.linear_head(x)
