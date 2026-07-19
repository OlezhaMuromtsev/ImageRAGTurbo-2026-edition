import torch
from torch import nn

from model.denoiser.bottleneck import get_stable_diffusion_bottleneck
from model.denoiser.decoder import get_stable_diffusion_decoder
from model.denoiser.encoder import get_stable_diffusion_encoder
from model.RAG.rag import RAG
from model.text_encoder.encoder import TextEncoder

class ImageRagTurbo(nn.Module):
    def __init__(self, db_path):
        super(ImageRagTurbo, self).__init__()
        self.text_encoder = TextEncoder()
        self.rag = RAG(db_path)
        self.encoder = get_stable_diffusion_encoder()
        self.bottleneck = get_stable_diffusion_bottleneck(self.encoder)
        self.decoder = get_stable_diffusion_decoder(self.encoder)
    def forward(self, x):
        query, x = self.text_encoder.forward(x)
        retrieved, latent = self.rag.forward(query)
        retrieved = retrieved.unsqueeze(0)
        x = self.encoder(x)
        retrieved = self.encoder(retrieved)
        if retrieved is not None:
            x = self.bottleneck.rag_blending(x.hidden_states, retrieved.hidden_states, 0.8)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x
    
    def freeze_part(self, part_name):
        part = getattr(self, part_name)
        for param in part.parameters():
            param.requires_grad = False
    
    def unfreeze_part(self, part_name):
        part = getattr(self, part_name)
        for param in part.parameters():
            param.requires_grad = True