import torch
import torch.nn as nn
import torch.optim as optim


class Projection(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.group_norm_2d = nn.GroupNorm(
            num_groups=min(32, in_channels),
            num_channels=in_channels,
            eps=1e-6,
            affine=True
        )
        self.spectral_conv_2d_1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        )
        self.silu = nn.SiLU()
        self.spectral_conv_2d_2 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.group_norm_2d(x)
        x = self.spectral_conv_2d_1(x)
        x = self.silu(x)
        x = self.spectral_conv_2d_2(x)
        x = self.adaptive_pool(x)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]
        x = self.fc(x)  # [B, 1]
        return x

class Discriminator(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder_ = encoder
        self.down_block_res_samples_used_ = [3, 6, 10] # We use only final layers of blocks

        projection_params = {
            0:  320,   # [1, 320, 64, 64]
            1:  320,   # [1, 320, 64, 64]
            2:  320,   # [1, 320, 64, 64]
            3:  320,   # [1, 320, 32, 32]
            4:  640,   # [1, 640, 32, 32]
            5:  640,   # [1, 640, 32, 32]
            6:  640,   # [1, 640, 16, 16]
            7:  1280,  # [1, 1280, 16, 16]
            8:  1280,  # [1, 1280, 16, 16]
            9:  1280,  # [1, 1280, 8, 8]
            10: 1280,  # [1, 1280, 8, 8]
        }
        
        self.projection_params = projection_params
        self.projections = nn.ModuleList([
            Projection(projection_params[k]) 
            for k in range(self.down_block_res_samples_num_)
        ])

    def forward(
            self,
            text_embedding: torch.Tensor,
            timestep: torch.Tensor = None,
            latent: torch.Tensor = None,
    ):
        with torch.no_grad():
            encoder_output = self.encoder_(text_embedding, timestep, latent)
        logits = []
        for k in self.down_block_res_samples_used_:
            logit = self.projections[k](encoder_output.down_block_res_samples[k])
            logits.append(logit)
        concat = torch.cat(logits, dim=1)
        return concat