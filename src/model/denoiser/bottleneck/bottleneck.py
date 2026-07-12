from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class BottleneckOutput:
    hidden_states: torch.Tensor
    temb: torch.Tensor
    encoder_hidden_states: torch.Tensor
    down_block_res_samples: tuple[torch.Tensor, ...]
    timestep: torch.Tensor
    latent: torch.Tensor

# hidden_states: torch.Tensor при 512×512: [B, 1280, 8, 8]
# temb: torch.Tensor : [B, 1280]
# encoder_hidden_states: torch.Tensor [B, 77, 1024]
# down_block_res_samples: tuple[torch.Tensor, ...] [B, C_i, H_i, W_i]
# timestep: torch.Tensor tensor(999.)
# latent: torch.Tensor [B, 4, 64, 64]


class StableDiffusionBottleneck(nn.Module):
    def __init__(self, unet):
        super().__init__()

        self.mid_block = unet.mid_block

    def forward(self, encoder_output):
        hidden_states = self.mid_block(
            encoder_output.hidden_states,
            encoder_output.temb,
            encoder_hidden_states=encoder_output.encoder_hidden_states,
            attention_mask=None,
            cross_attention_kwargs=None,
            encoder_attention_mask=None,
        )

        return BottleneckOutput(
            hidden_states=hidden_states,
            temb=encoder_output.temb,
            encoder_hidden_states=encoder_output.encoder_hidden_states,
            down_block_res_samples=encoder_output.down_block_res_samples,
            timestep=encoder_output.timestep,
            latent=encoder_output.latent,
        )


def get_stable_diffusion_bottleneck(encoder):
    return StableDiffusionBottleneck(
        unet=encoder.unet,
    )