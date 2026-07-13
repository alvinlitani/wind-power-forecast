"""
Seq2seq LSTM model for wind power forecasting.

Architecture:
    - Static MLP: maps site embedding + static features -> initial hidden/cell state
    - Encoder LSTM: processes 48h of historical weather + generation data
    - Decoder LSTM: processes 24h of forecast weather, produces hourly output predictions

Inputs:
    encoder_input: (batch, 48, 5) - output_mwh, available_capacity_mw,
                                     wind_speed_hub, temperature_2m, surface_pressure
    decoder_input: (batch, 24, 3) - wind_speed_hub, temperature_2m, surface_pressure
    static:        (batch, 3)     - capacity_mw, hub_height, site_id

Output:
    predictions:   (batch, 24)    - predicted output_mwh per hour
"""

import torch
import torch.nn as nn


class WindPowerLSTM(nn.Module):
    def __init__(
        self,
        encoder_input_size: int = 5,
        decoder_input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_sites: int = 45,
        site_embedding_dim: int = 8,
        static_numeric_size: int = 2,  # capacity_mw, hub_height
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Site embedding
        self.site_embedding = nn.Embedding(num_sites, site_embedding_dim)

        # Static MLP: maps static features -> initial hidden and cell states
        static_input_size = site_embedding_dim + static_numeric_size
        self.static_mlp = nn.Sequential(
            nn.Linear(static_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2 * num_layers),
        )

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Dropout applied to decoder output before projection
        self.dropout = nn.Dropout(dropout)

        # Output projection: hidden state -> 1 predicted value per timestep
        self.output_layer = nn.Linear(hidden_size, 1)

    def _init_hidden(self, static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute initial hidden and cell states from static features.

        Args:
            static: (batch, 3) - capacity_mw, hub_height, site_id

        Returns:
            (h_0, c_0) each of shape (num_layers, batch, hidden_size)
        """
        # Split static into numeric and site_id
        static_numeric = static[:, :2]  # capacity_mw, hub_height
        site_ids = static[:, 2].long()  # site_id

        # Embed site IDs and concatenate with numeric features
        site_emb = self.site_embedding(site_ids)  # (batch, site_embedding_dim)
        static_combined = torch.cat([static_numeric, site_emb], dim=1)

        # MLP to produce hidden and cell states
        out = self.static_mlp(static_combined)  # (batch, hidden_size * 2 * num_layers)

        # Reshape and split into h_0 and c_0
        out = out.view(-1, self.num_layers, 2, self.hidden_size)
        h_0 = out[:, :, 0, :].permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden_size)
        c_0 = out[:, :, 1, :].permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden_size)

        return h_0, c_0

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            encoder_input: (batch, 48, 5)
            decoder_input: (batch, 24, 3)
            static:        (batch, 3)

        Returns:
            predictions:   (batch, 24)
        """
        # Compute initial hidden state from static features
        h_0, c_0 = self._init_hidden(static)

        # Encode historical sequence
        _, (h_enc, c_enc) = self.encoder(encoder_input, (h_0, c_0))

        # Decode forecast sequence
        decoder_output, _ = self.decoder(decoder_input, (h_enc, c_enc))

        # Project to predictions: (batch, 24, hidden_size) -> (batch, 24, 1) -> (batch, 24)
        predictions = self.output_layer(self.dropout(decoder_output)).squeeze(-1)

        return predictions
