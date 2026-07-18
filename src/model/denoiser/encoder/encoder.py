from diffusers import StableDiffusionPipeline
from dataclasses import dataclass
import torch
import torch.nn as nn

# Структура Unet
# conv_in
# down_blocks
# mid_block
# up_blocks
# conv_norm_out
# conv_act
# conv_out


MODEL_ID = "stabilityai/sd-turbo"


@dataclass
class EncoderOutput:
    hidden_states: torch.Tensor
    temb: torch.Tensor
    encoder_hidden_states: torch.Tensor
    down_block_res_samples: tuple
    timestep: torch.Tensor
    latent: torch.Tensor

# hidden_states: torch.Size([1, 1280, 8, 8]) - преобразованный шум вместе с промптом
# temb: torch.Size([1, 1280]) - представление timestep
# encoder_hidden_states: torch.Size([1, 77, 1024])  - промт закодированный
# down_block_res_samples: - скип-соеднинения
    # 0 torch.Size([1, 320, 64, 64])
    # 1 torch.Size([1, 320, 64, 64])
    # 2 torch.Size([1, 320, 64, 64])
    # 3 torch.Size([1, 320, 32, 32])
    # 4 torch.Size([1, 640, 32, 32])
    # 5 torch.Size([1, 640, 32, 32])
    # 6 torch.Size([1, 640, 16, 16])
    # 7 torch.Size([1, 1280, 16, 16])
    # 8 torch.Size([1, 1280, 16, 16])
    # 9 torch.Size([1, 1280, 8, 8])
    # 10 torch.Size([1, 1280, 8, 8])
    # 11 torch.Size([1, 1280, 8, 8])
# timestamp: tensor(999., device='cuda:0') - шаг зашумления
# latent: torch.Size([1, 4, 64, 64]) - исходный шум

class StableDiffusionEncoder(nn.Module):
    def __init__(
            self,
            pipeline,
            height=512,
            width=512,
            num_inference_steps=1,
    ):
        super().__init__()

        self.pipeline = pipeline

        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder
        self.scheduler = pipeline.scheduler
        self.unet = pipeline.unet

        self.conv_in = self.unet.conv_in
        self.time_embedding = self.unet.time_embedding
        self.time_embed_act = self.unet.time_embed_act
        self.down_blocks = self.unet.down_blocks

        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps

    def encode_prompt(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(self.unet.device)

        encoder_hidden_states = self.text_encoder(
            input_ids
        ).last_hidden_state

        encoder_hidden_states = encoder_hidden_states.to(
            dtype=self.unet.dtype
        )

        return encoder_hidden_states

    def create_latent(self, batch_size):
        latent_height = self.height // self.pipeline.vae_scale_factor
        latent_width = self.width // self.pipeline.vae_scale_factor

        latent = torch.randn(
            batch_size,
            self.unet.config.in_channels,
            latent_height,
            latent_width,
            device=self.unet.device,
            dtype=self.unet.dtype,
        )

        latent = latent * self.scheduler.init_noise_sigma

        return latent

    def create_timestep(self):
        self.scheduler.set_timesteps(
            self.num_inference_steps,
            device=self.unet.device,
        )

        timestep = self.scheduler.timesteps[0]

        return timestep

    def create_time_embedding(self, sample, timestep):
        t_emb = self.unet.get_time_embed(
            sample=sample,
            timestep=timestep,
        )

        temb = self.time_embedding(t_emb)

        if self.time_embed_act is not None:
            temb = self.time_embed_act(temb)

        return temb

    def forward(
            self,
            text_embedding: torch.Tensor,
            timestep: torch.Tensor = None,
            latent: torch.Tensor = None,
    ):

        batch_size = text_embedding.shape[0]

        encoder_hidden_states = text_embedding


        if latent is None:
            latent = self.create_latent(batch_size)
        else:
            if latent.ndim == 3:
                latent = latent.unsqueeze(0)


        if timestep is None:
            timestep = self.create_timestep()
        else:
            timestep = torch.as_tensor(
                timestep,
                device=text_embedding.device,
            )


        sample = self.scheduler.scale_model_input(
            latent,
            timestep,
        )

        if self.unet.config.center_input_sample:
            sample = 2 * sample - 1.0

        temb = self.create_time_embedding(
            sample,
            timestep,
        )

        encoder_hidden_states = (
            self.unet.process_encoder_hidden_states(
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=None,
            )
        )

        hidden_states = self.conv_in(sample)

        down_block_res_samples = (hidden_states,)

        for down_block in self.down_blocks:
            if getattr(
                    down_block,
                    "has_cross_attention",
                    False,
            ):
                hidden_states, res_samples = down_block(
                    hidden_states=hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    encoder_attention_mask=None,
                )
            else:
                hidden_states, res_samples = down_block(
                    hidden_states=hidden_states,
                    temb=temb,
                )

            down_block_res_samples += tuple(res_samples)

        return EncoderOutput(
            hidden_states=hidden_states,
            temb=temb,
            encoder_hidden_states=encoder_hidden_states,
            down_block_res_samples=down_block_res_samples,
            timestep=timestep,
            latent=latent,
        )

def get_stable_diffusion_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #dtype = torch.float16 if device == "cuda" else torch.float32
    dtype =  torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        safety_checker=None,
        local_files_only=True,
    ).to(device)

    pipeline.unet.eval()
    pipeline.text_encoder.eval()
    pipeline.vae.eval()
    return StableDiffusionEncoder(
            pipeline,
            height=512,
            width=512,
            num_inference_steps=1
    )