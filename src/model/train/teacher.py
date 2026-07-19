from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline

from ..denoiser.encoder import StableDiffusionEncoder
"""
teacher = замороженный SD2.
по z_T строит образцовый латент z_0_real, приближаем student к нему.
"""
MODEL_ID = "sd2-community/stable-diffusion-2-1-base"

GUIDANCE_SCALE = 7.5


class StableDiffusionTeacher(nn.Module):
    def __init__(
            self,
            pipeline,
            guidance_scale=GUIDANCE_SCALE,
    ):
        super().__init__()

        self.pipeline = pipeline
        self.unet = pipeline.unet
        self.scheduler = pipeline.scheduler
        self.guidance_scale = guidance_scale

        # энкодер учителя для дискриминатора, оборачивает свой тем же классом, что у UNet
        # дискриминатор получает привычный интерфейс EncoderOutput
        self.encoder = StableDiffusionEncoder(
            pipeline,
            height=512,
            width=512,
            num_inference_steps=1,
        )

        self.alphas_cumprod = self.scheduler.alphas_cumprod
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        # пустой промпт для classifier-free guidance
        with torch.no_grad():
            uncond_input = pipeline.tokenizer(
                [""],
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,
                return_tensors="pt",
            ).to(pipeline.unet.device)
            self.uncond_embedding = pipeline.text_encoder(
                uncond_input.input_ids
            ).last_hidden_state  # [1, 77, 1024]

        for module in (self.unet, pipeline.vae, pipeline.text_encoder):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def _alpha_sigma(self, t_int):
        alpha_bar = self.alphas_cumprod[t_int].to(self.unet.device)
        alpha = alpha_bar.sqrt()
        sigma = (1.0 - alpha_bar).sqrt()
        return alpha, sigma

    @torch.no_grad()
    def encode_text(self, prompts):
        # обработка "сырых" строк текст-энкодером teacher
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self.pipeline.tokenizer(
            list(prompts),
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.unet.device)
        return self.pipeline.text_encoder(tokens.input_ids).last_hidden_state

    def _predict_eps(self, z, t_int, text_embedding):
        # один вызов UNet с classifier-free guidance
        batch_size = text_embedding.shape[0]
        uncond = self.uncond_embedding.expand(batch_size, -1, -1)

        latent_in = torch.cat([z, z], dim=0)
        cond_in = torch.cat([uncond, text_embedding], dim=0)
        t_in = t_int.expand(latent_in.shape[0])

        eps_uncond, eps_cond = self.unet(
            latent_in,
            t_in,
            encoder_hidden_states=cond_in,
        ).sample.chunk(2)

        return eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)

    @torch.no_grad()
    def forward(
            self,
            text_embedding: torch.Tensor,
            timestep: torch.Tensor = None,
            latent: torch.Tensor = None,
    ):
        """
        многошаговый денойзинг шума в z_0_real [B, 4, 64, 64] в latent-масштабе

        text_embedding: [B, 77, 1024] — закодированный промпт, либо сырые тексты list[str] / str (закодируются сами)
        timestep:       последовательность уровней шума в [0, 1], идём по убыванию шума

        latent:         z_T — стартовый шум [B, 4, 64, 64], такой же у student
        """
        device = self.unet.device

        if isinstance(text_embedding, (str, list, tuple)):
            text_embedding = self.encode_text(text_embedding)
        text_embedding = text_embedding.to(device)

        batch_size = text_embedding.shape[0]

        if latent is None:
            latent = torch.randn(
                batch_size, 4, 64, 64,
                device=device, dtype=self.unet.dtype,
            )
        else:
            latent = latent.to(device, dtype=self.unet.dtype)
            if latent.ndim == 3:
                latent = latent.unsqueeze(0)

        if timestep is None:
            timestep = torch.linspace(0.99, 0.0, 50)

        # [0,1] -> [0, 999]
        t_ints = (
            timestep.flatten() * (self.num_train_timesteps - 1)
        ).long().sort(descending=True).values.to(device)

        z = latent
        for i, t_int in enumerate(t_ints):
            eps_pred = self._predict_eps(z, t_int, text_embedding)

            alpha_t, sigma_t = self._alpha_sigma(t_int)
            z0_pred = (z - sigma_t * eps_pred) / alpha_t

            if i + 1 < len(t_ints):
                # перешумляем z0_pred на следующий уровень предсказанным eps
                alpha_next, sigma_next = self._alpha_sigma(t_ints[i + 1])
                z = alpha_next * z0_pred + sigma_next * eps_pred
            else:
                z = z0_pred

        return z.to(torch.float32)


def get_stable_diffusion_teacher():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    pipeline.unet.eval()
    pipeline.vae.eval()

    return StableDiffusionTeacher(
        pipeline,
        guidance_scale=GUIDANCE_SCALE,
    )
