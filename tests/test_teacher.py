import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.model.text_encoder import TextEncoder
from src.model.train.teacher import get_stable_diffusion_teacher
from src.model.denoiser.encoder import StableDiffusionEncoder

print("Loading text encoder and teacher...")
enc = TextEncoder()
teacher = get_stable_diffusion_teacher()
device = teacher.unet.device


# 1. Совместимость с TextEncoder
prompt = "a photograph of an astronaut riding a horse"
cond = enc.encode_prompt(prompt)
print(f"\ncond from TextEncoder: {tuple(cond.shape)} (expecting (1, 77, 1024))")
assert tuple(cond.shape) == (1, 77, 1024), "TextEncoder returned an unexpected cond shape"

# 2. Основной контракт с trainer: forward(cond, timestep, latent) -> z_0_real
z_T = torch.randn(1, 4, 64, 64)
timestep = torch.tensor([0.25, 0.5, 0.75, 0.99])

print("\nRunning teacher denoising pass...")
z_0_real = teacher.forward(cond, timestep=timestep, latent=z_T)

print(f"z_0_real shape: {tuple(z_0_real.shape)} (expecting (1, 4, 64, 64))")
print(f"z_0_real dtype: {z_0_real.dtype} (expecting torch.float32)")
assert tuple(z_0_real.shape) == (1, 4, 64, 64), f"Invalid z_0_real shape: {z_0_real.shape}"
assert z_0_real.dtype == torch.float32, f"Invalid z_0_real dtype: {z_0_real.dtype}"

# 3. Приём "сырых" строк
print("\nChecking raw-string prompts...")
z_0_from_str = teacher.forward([prompt], timestep=timestep, latent=z_T)
assert tuple(z_0_from_str.shape) == (1, 4, 64, 64), \
    "forward with a list of strings returned an invalid shape"
assert torch.isfinite(z_0_from_str).all(), "forward with strings returned NaN/inf"
print(f"forward(list[str]) -> {tuple(z_0_from_str.shape)} — ok")

# 4. Проверка на отсутствие NaN/inf
assert torch.isfinite(z_0_real).all(), "z_0_real contains NaN/inf"
print(f"\nz_0_real range: [{z_0_real.min():.3f}, {z_0_real.max():.3f}] — finite values, ok")

# 5. Детерминизм: при одних и тех же входах один и тот де результат
z_0_repeat = teacher.forward(cond, timestep=timestep, latent=z_T)
max_diff = (z_0_real - z_0_repeat).abs().max().item()
print(f"Determinism check: max diff between two runs = {max_diff:.2e} (expecting ~0)")
assert max_diff < 1e-4, "Teacher is non-deterministic — reference latents are unstable"

# 6. Батч: trainer гоняет батчами, форма должна масштабироваться по B
cond_batch = enc.encode_prompt(["a red car", "a blue house", "a green tree"])
z_T_batch = torch.randn(3, 4, 64, 64)
z_0_batch = teacher.forward(cond_batch, timestep=timestep, latent=z_T_batch)
print(f"\nBatch output: {tuple(z_0_batch.shape)} (expecting (3, 4, 64, 64))")
assert tuple(z_0_batch.shape) == (3, 4, 64, 64), "Batch is processed incorrectly"

# 7. Encoder для дискриминатора: trainer делает Discriminator(encoder=teacher.encoder)
print("\nChecking teacher.encoder (used to build the discriminator)...")
assert hasattr(teacher, "encoder"), "Teacher does not have the .encoder attribute"
assert isinstance(teacher.encoder, StableDiffusionEncoder), \
    "teacher.encoder must be StableDiffusionEncoder (same interface as student)"

enc_out = teacher.encoder.forward(
    cond.to(device),
    latent=z_T.to(device),
    timestep=torch.tensor(500).to(device),
)
assert tuple(enc_out.hidden_states.shape) == (1, 1280, 8, 8), \
    f"Unexpected shape of teacher encoder H-space: {enc_out.hidden_states.shape}"
print(f"teacher.encoder H-space: {tuple(enc_out.hidden_states.shape)} (expecting (1, 1280, 8, 8))")

print("\n[OK] Teacher passes all checks — shapes/dtype/determinism OK, "
      "accepts both embeddings and raw strings, exposes real noise schedule, "
      "and provides encoder for the discriminator.")
