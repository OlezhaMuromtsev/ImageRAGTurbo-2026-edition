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
    def rag_blending(
        self,
        encoder_output: torch.Tensor,
        rag_output: torch.Tensor,
        encoder_weight: float,
    ):
        import torch.nn.functional as F

        eps = 1e-6

        batch_size = encoder_output.shape[0]
        original_shape = encoder_output.shape

        rag_flat = rag_output.flatten(start_dim=1)
        encoder_flat = encoder_output.flatten(start_dim=1)

        retrieved_unit = F.normalize(rag_flat, dim=1)
        encoder_unit = F.normalize(encoder_flat, dim=1)

        cosine = torch.sum(
            retrieved_unit * encoder_unit,
            dim=1,
            keepdim=True,
        )

        cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)
        omega = torch.acos(cosine)
        sin_omega = torch.sin(omega)

        encoder_weight = torch.as_tensor(
            encoder_weight,
            device=encoder_output.device,
            dtype=encoder_output.dtype,
        )

        encoder_weight = encoder_weight.expand(batch_size)
        encoder_weight = encoder_weight.reshape(batch_size, 1)

        rag_coefficient = torch.sin((1.0 - encoder_weight) * omega) / sin_omega
        encoder_coefficient = torch.sin(encoder_weight * omega) / sin_omega

        hidden_states_blending = rag_coefficient * rag_flat + encoder_coefficient * encoder_flat
        #TODO: вопрос численной стабильности не исследовался на случаях вырождения синуса
        return hidden_states_blending.reshape(original_shape)


def get_stable_diffusion_bottleneck(encoder):
    return StableDiffusionBottleneck(
        unet=encoder.unet,
    )