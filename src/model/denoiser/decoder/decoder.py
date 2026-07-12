from dataclasses import dataclass
import torch
from torch import nn
from PIL import Image


@dataclass
class DecoderOutput:
    model_output: torch.Tensor
    latent: torch.Tensor
    image_tensor: torch.Tensor
    images: list[Image.Image]

# model_output: torch.Tensor  [B, 4, 64, 64]
# latent: torch.Tensor [B, 4, 64, 64]
# image_tensor: torch.Tensor [B, 3, 512, 512]
# images: list[Image.Image]

class StableDiffusionDecoder(nn.Module):
    def __init__(self, pipeline):
        super().__init__()

        self.pipeline = pipeline
        self.unet = pipeline.unet
        self.scheduler = pipeline.scheduler
        self.vae = pipeline.vae
        self.image_processor = pipeline.image_processor

        self.up_blocks = self.unet.up_blocks
        self.conv_norm_out = self.unet.conv_norm_out
        self.conv_act = self.unet.conv_act
        self.conv_out = self.unet.conv_out

    def decode_unet(self, bottleneck_output):
        hidden_states = bottleneck_output.hidden_states

        down_block_res_samples = (
            bottleneck_output.down_block_res_samples
        )

        for up_block in self.up_blocks:
            # Каждый up-block использует столько skip-тензоров,
            # сколько ResNet-слоёв находится внутри него.
            number_of_residuals = len(up_block.resnets)

            residual_samples = down_block_res_samples[
                -number_of_residuals:
            ]

            down_block_res_samples = down_block_res_samples[
                :-number_of_residuals
            ]

            if getattr(
                up_block,
                "has_cross_attention",
                False,
            ):
                hidden_states = up_block(
                    hidden_states=hidden_states,
                    temb=bottleneck_output.temb,
                    res_hidden_states_tuple=residual_samples,
                    encoder_hidden_states=(
                        bottleneck_output.encoder_hidden_states
                    ),
                    cross_attention_kwargs=None,
                    upsample_size=None,
                    attention_mask=None,
                    encoder_attention_mask=None,
                )
            else:
                hidden_states = up_block(
                    hidden_states=hidden_states,
                    temb=bottleneck_output.temb,
                    res_hidden_states_tuple=residual_samples,
                    upsample_size=None,
                )

        # Финальные слои исходного U-Net.
        if self.conv_norm_out is not None:
            hidden_states = self.conv_norm_out(hidden_states)
            hidden_states = self.conv_act(hidden_states)

        model_output = self.conv_out(hidden_states)

        return model_output

    def make_scheduler_step(
        self,
        model_output,
        timestep,
        initial_latent,
    ):
        scheduler_output = self.scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=initial_latent,
            return_dict=True,
        )

        return scheduler_output.prev_sample

    def decode_latent(self, latent):
        latent_for_vae = (
            latent / self.vae.config.scaling_factor
        )

        image_tensor = self.vae.decode(
            latent_for_vae,
            return_dict=True,
        ).sample

        return image_tensor

    def convert_to_pil(self, image_tensor):
        images = self.image_processor.postprocess(
            image_tensor,
            output_type="pil",
        )

        return images

    def forward(self, bottleneck_output):
        model_output = self.decode_unet(
            bottleneck_output
        )

        latent = self.make_scheduler_step(
            model_output=model_output,
            timestep=bottleneck_output.timestep,
            initial_latent=bottleneck_output.latent,
        )

        image_tensor = self.decode_latent(latent)

        images = self.convert_to_pil(image_tensor)

        return DecoderOutput(
            model_output=model_output,
            latent=latent,
            image_tensor=image_tensor,
            images=images,
        )


def get_stable_diffusion_decoder(encoder):
    return StableDiffusionDecoder(
        pipeline=encoder.pipeline,
    )