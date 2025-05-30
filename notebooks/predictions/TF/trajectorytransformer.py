import torch
import torch.nn as nn

class TrajectoryTransformer(nn.Module):
    def __init__(self, n_entities=22, input_dim=2, model_dim=128, num_heads=4, num_layers=2, output_dim=2, seq_out=5):
        super().__init__()
        self.seq_out = seq_out
        self.n_entities = n_entities

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 10, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, n_entities, 2)
        B, T, E, D = x.shape
        x = self.input_proj(x)  # (B, T, E, model_dim)
        x = x.view(B, T * E, -1)  # flatten time and entity
        x = x + self.pos_embedding.repeat(1, self.n_entities, 1)[:, :x.shape[1], :]
        x = self.transformer_encoder(x)
        x = self.output_proj(x)
        x = x.view(B, T, E, -1)
        return x