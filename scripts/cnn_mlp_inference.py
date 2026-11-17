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

ARCHITECTURE_SCHEMA = "ares-pose-to-range-cnn-architecture-v1"


def describe_model_architecture(model):
    """Save safe, numeric module descriptions rather than pickling notebook code."""
    def describe(layer):
        kind = type(layer).__name__
        if type(layer) is nn.Linear:
            return dict(type=kind, in_features=layer.in_features,
                        out_features=layer.out_features, bias=layer.bias is not None)
        if type(layer) is nn.Conv1d:
            return dict(type=kind, in_channels=layer.in_channels,
                        out_channels=layer.out_channels, kernel_size=list(layer.kernel_size),
                        stride=list(layer.stride), padding=list(layer.padding),
                        dilation=list(layer.dilation), groups=layer.groups,
                        bias=layer.bias is not None, padding_mode=layer.padding_mode)
        if type(layer) is nn.Upsample:
            if layer.size is None or layer.mode != "linear":
                raise ValueError("Export requires linear upsampling with an explicit target size.")
            size = [layer.size] if isinstance(layer.size, int) else list(layer.size)
            return dict(type=kind, size=size, mode=layer.mode, align_corners=layer.align_corners)
        if type(layer) is nn.Dropout:
            return dict(type=kind, p=layer.p, inplace=layer.inplace)
        if type(layer) is nn.GELU:
            if layer.approximate != "none":
                raise ValueError("MATLAB export currently supports exact GELU only.")
            return dict(type=kind)
        if type(layer) in (nn.ReLU, nn.SiLU, nn.Identity):
            return dict(type=kind)
        if type(layer) is nn.Softplus and layer.beta == 1 and layer.threshold == 20:
            return dict(type=kind)
        raise ValueError(f"Unsupported CNN export layer: {kind}")

    return dict(schema_version=ARCHITECTURE_SCHEMA,
                latent_channels=model.latent_channels, latent_length=model.latent_length,
                encoder=[describe(layer) for layer in model.encoder],
                decoder=[describe(layer) for layer in model.decoder])


def _module_from_description(spec):
    options = {key: value for key, value in spec.items() if key != "type"}
    classes = {name: getattr(nn, name) for name in (
        "Linear", "Conv1d", "Upsample", "Dropout", "GELU", "ReLU", "SiLU", "Identity", "Softplus"
    )}
    if spec["type"] not in classes:
        raise ValueError(f"Unsupported saved CNN layer: {spec['type']}")
    return classes[spec["type"]](**options)


def legacy_checkpoint_architecture(checkpoint):
    """Migrate the two historical notebook layouts whose checkpoints lack a module plan.

    0/2/4 is the original all-activated encoder. 0/3/6 is the inspected notebook
    revision with hidden Dropout and a linear latent projection. Unknown layouts
    are rejected: renaming weights alone cannot establish activation semantics.
    """
    import re
    config = checkpoint["config"]["model"]
    widths = [config["input_features"], *config["encoder_hidden"],
              config["latent_channels"] * config["latent_length"]]
    indices = sorted(int(match.group(1)) for name in checkpoint["model_state_dict"]
                     if (match := re.fullmatch(r"encoder\.(\d+)\.weight", name)))
    count = len(widths) - 1
    if indices == list(range(0, 2 * count, 2)):
        # This profile is the original class below, including its final activation.
        model = PoseToRangeCNN(config)
        return describe_model_architecture(model), "legacy-all-activated-encoder"
    if indices == list(range(0, 3 * count, 3)) and count > 1:
        model = PoseToRangeCNN(config)
        activation = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[config["hidden_activation"]]
        layers = []
        for i, (inputs, outputs) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(inputs, outputs))
            if i < count - 1:
                layers.extend([activation(), nn.Dropout(p=0.1)])
        model.encoder = nn.Sequential(*layers)
        return describe_model_architecture(model), "legacy-hidden-dropout-linear-latent"
    raise ValueError(
        f"Unrecognized legacy encoder layout {indices}. Save model_architecture in the "
        "training checkpoint; layer names alone cannot describe an arbitrary network."
    )


def load_checkpoint_network(checkpoint):
    """Rebuild the recorded architecture, then require every trained tensor to match."""
    if "model_architecture" in checkpoint:
        architecture = checkpoint["model_architecture"]
        source = "checkpoint-model-architecture"
    else:
        architecture, source = legacy_checkpoint_architecture(checkpoint)
    model = PoseToRangeCNN(checkpoint["config"]["model"], architecture=architecture).cpu()
    # Do not use strict=False: all weights must load into the correct operations.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval(), architecture, source


class PoseToRangeCNN(nn.Module):
    """12 encoded state features -> [batch, 15, 716] scan."""
    def __init__(self, model_config, *, architecture=None):
        super().__init__()
        self.model_config = copy.deepcopy(model_config)
        c = self.model_config
        activation_types = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU,
                            "identity": nn.Identity, "softplus": nn.Softplus}
        self.latent_channels = c["latent_channels"]
        self.latent_length = c["latent_length"]

        if architecture is not None:
            if architecture["schema_version"] != ARCHITECTURE_SCHEMA:
                raise ValueError("Unsupported saved CNN architecture schema.")
            if (architecture["latent_channels"] != self.latent_channels
                    or architecture["latent_length"] != self.latent_length):
                raise ValueError("Saved architecture differs from the configured latent shape.")
            self.encoder = nn.Sequential(*map(_module_from_description, architecture["encoder"]))
            self.decoder = nn.Sequential(*map(_module_from_description, architecture["decoder"]))
            # Validate module types/properties even when the architecture came from a file.
            describe_model_architecture(self)
            return

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

