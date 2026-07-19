import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class TextEncoder(nn.Module):
    MAX_LENGTH = 77  # fixed length for cross-attention

    def __init__(
        self,
        model_id: str = "sd2-community/stable-diffusion-2-1-base",  # from https://huggingface.co/sd2-community/stable-diffusion-2-1-base
        device: str = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        """weights are taken from Stable Diffusion to give UNet 77×1024 embeddings"""
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.model = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", use_safetensors=True
        ).to(self.device, dtype=dtype)

        self.model.eval()  # freezing
        for param in self.model.parameters():
            param.requires_grad = False

        self.embed_dim = self.model.config.hidden_size  # 1024 for SD 2.1

    @torch.no_grad()
    def _encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts,
            padding="max_length",  # all cached conditions in database have same form, could be served in batches
            max_length=self.MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        return self.model(**inputs)

    @torch.no_grad()
    def encode_prompt(self, texts) -> torch.Tensor:
        """sequence of tokens (B, 77, D) - conditioning for UNet"""
        return self._encode(texts).last_hidden_state

    @torch.no_grad()
    def encode_query(self, texts) -> torch.Tensor:
        """normalized pooled vector (B, D) - FAISS search key"""
        pooled = self._encode(texts).pooler_output
        return pooled / pooled.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def forward(self, texts):
        out = self._encode(texts)
        pooled = out.pooler_output
        query = pooled / pooled.norm(dim=-1, keepdim=True)  # normalize because scalar product is equal to the cosine proximity on unit vectors
        """query (1024) - presentation of entire text for searching in FAISS"""
        """out.last_hidden_state (77, 1024) - detailed representation of each word, used in UNet cross-attention"""
        return query, out.last_hidden_state
