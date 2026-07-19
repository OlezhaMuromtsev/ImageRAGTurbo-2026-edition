from .denoiser import *



class StableDiffusionUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = get_stable_diffusion_encoder()
        self.bottleneck = get_stable_diffusion_bottleneck(self.encoder)
        self.decoderSD = get_stable_diffusion_decoder(self.encoder)

    def forward(self, text_embedding, latent=None, timestep=None):
        encoder_out = self.encoder(text_embedding, latent=latent, timestep=timestep)
        bottleneck_out = self.bottleneck(encoder_out)
        return bottleneck_out

    def blending(
            self,
            rag_hidden_state,
            bottleneck_out,
            encoder_weight: float = 0.8
    ):
        import torch.nn.functional as F

        target_hidden_state = bottleneck_out.hidden_states

        eps = 1e-6

        batch_size = target_hidden_state.shape[0]
        original_shape = target_hidden_state.shape

        rag_flat = rag_hidden_state.flatten(start_dim=1)
        target_flat = target_hidden_state.flatten(start_dim=1)

        retrieved_unit = F.normalize(rag_flat, dim=1)
        target_unit = F.normalize(target_flat, dim=1)

        cosine = torch.sum(
            retrieved_unit * target_unit,
            dim=1,
            keepdim=True,
        )

        cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)
        omega = torch.acos(cosine)
        sin_omega = torch.sin(omega)

        encoder_weight = torch.as_tensor(
            encoder_weight,
            device=target_hidden_state.device,
            dtype=target_hidden_state.dtype,
        )

        encoder_weight = encoder_weight.expand(batch_size)
        encoder_weight = encoder_weight.reshape(batch_size, 1)

        rag_coefficient = torch.sin((1.0 - encoder_weight) * omega) / sin_omega
        encoder_coefficient = torch.sin(encoder_weight * omega) / sin_omega

        hidden_states_blending = rag_coefficient * rag_flat + encoder_coefficient * target_flat
        hidden_states_blending = hidden_states_blending.reshape(original_shape)

        bottleneck_out.hidden_states = hidden_states_blending

        return bottleneck_out


    def decoder(self, bottleneck_out):
        return self.decoderSD.forward(bottleneck_out)