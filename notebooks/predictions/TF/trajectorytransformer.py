import torch
import torch.nn as nn

class TrajectoryTransformer(nn.Module):
    def __init__(self, n_entities=22, input_dim=2, model_dim=256, num_heads=8, num_layers=4, output_dim=2, seq_out=5):
        super().__init__()
        self.seq_out = seq_out
        self.n_entities = n_entities

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.max_seq_len = 512  # or larger if needed
        self.pos_embedding = nn.Embedding(self.max_seq_len, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            nn.Sigmoid()  # Output normalized to [0, 1]
        )

    def forward(self, x):
        # x: (batch, seq_len, n_entities, 2)
        B, T, E, D = x.shape
        x = self.input_proj(x)  # (B, T, E, model_dim)
        x = x.permute(0, 2, 1, 3)  # (B, E, T, model_dim)
        x = x.reshape(B * E, T, -1)  # Each player is an individual sequence

        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B * E, T)  # (B*E, T)
        x = x + self.pos_embedding(positions)  # Add positional embeddings

        x = self.transformer_encoder(x)  # (B*E, T, model_dim)
        x = self.output_proj(x)  # (B*E, T, output_dim)

        x = x.reshape(B, E, T, -1).permute(0, 2, 1, 3)  # (B, T, E, output_dim)
        return x
