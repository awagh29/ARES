"""Inference architecture matching the single-episode training notebook."""
import copy

import torch

from torch import nn



###################################################################
## PRE-PROCESSING FUNCTION
###################################################################
def encode_sensor_state(raw_states):
    
    if raw_states.ndim < 1 or raw_states.shape[-1] != 9:
        raise ValueError(
            "Expected raw sensor states with last dimension 9."
        )

    raw_states = raw_states.float()
    angles = raw_states[..., 6:9]

    angle_features = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
    angle_features = angle_features.flatten(start_dim=-2)

    return torch.cat((raw_states[..., :6], angle_features), dim=-1)

class PoseToRangeCNN(nn.Module):
    """12 encoded state features -> [batch, 15, 716] scan."""
    def __init__(self, model_config):
        super().__init__()
        self.model_config = copy.deepcopy(model_config)
        c = self.model_config
        activation_types = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU,
                            "identity": nn.Identity, "softplus": nn.Softplus}
        self.latent_channels = c["latent_channels"]
        self.latent_length = c["latent_length"]

        widths = [c["input_features"], *c["encoder_hidden"],
                  self.latent_channels * self.latent_length]
        encoder = []
        for in_features, out_features in zip(widths[:-1], widths[1:]):
            encoder.extend([nn.Linear(in_features, out_features), activation_types[c["hidden_activation"]]()])
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_channels = self.latent_channels
        last = len(c["decoder_channels"]) - 1
        for i, (out_channels, length, kernel) in enumerate(zip(
            c["decoder_channels"], c["decoder_lengths"], c["kernel_sizes"]
        )):
            decoder.extend([
                nn.Upsample(size=length, mode=c["upsample_mode"],
                            align_corners=c["align_corners"]),
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel,
                          padding=kernel // 2),
            ])
            activation = c["hidden_activation"] if i != last else c["output_activation"]
            decoder.append(activation_types[activation]())
            in_channels = out_channels
        self.decoder = nn.Sequential(*decoder)

    def forward(self, sensor_state):
        features = self.encoder(sensor_state)
        features = features.reshape(
            sensor_state.shape[0], self.latent_channels, self.latent_length
        )
        return self.decoder(features)

