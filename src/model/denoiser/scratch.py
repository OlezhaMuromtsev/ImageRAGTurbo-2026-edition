from encoder import get_stable_diffusion_encoder
from bottleneck import get_stable_diffusion_bottleneck
from decoder import get_stable_diffusion_decoder
import torch

encoder = get_stable_diffusion_encoder()
bottleneck = get_stable_diffusion_bottleneck(encoder)
decoder = get_stable_diffusion_decoder(encoder)

with torch.inference_mode():
    encoder_output = encoder("zombiehoursewithKhighPhotoRealistic")
    bottleneck_output = bottleneck(encoder_output)
    decoder_output = decoder(bottleneck_output)

print(decoder_output.images[0].show())



