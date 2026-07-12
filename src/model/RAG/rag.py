import os
import json
from dataclasses import dataclass
from typing import List, Optional

import faiss
import torch


@dataclass
class RetrievalResult:
    """one base element found"""
    index: int            # element's serial number in database, needed for debugging
    prompt: str           # original text of the found prompt, for logging and debugging
    cond: torch.Tensor    # text condition (77, 1024)
    latent: torch.Tensor  # found image latent (4, 64, 64) 
    score: float          # cosine proximity between query and found element [0.0, 1.0]


class RAG:
    """
    next step: h_retr = unet.encoder(cond=result.cond, t=0, z=result.latent)
    H-space considers the feature frozen UNet encoder
    """

    def __init__(self, db_dir: str, device: Optional[str] = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(db_dir, "meta.json")) as f:
            self.meta = json.load(f)

        self.index = faiss.read_index(os.path.join(db_dir, "index.faiss"))
        self.cond = torch.load(
            os.path.join(db_dir, "text_cond.pt"), map_location="cpu", weights_only=True
        )
        self.latents = torch.load(
            os.path.join(db_dir, "latents.pt"), map_location="cpu", weights_only=True
        )
        with open(os.path.join(db_dir, "prompts.json"), encoding="utf-8") as f:
            self.prompts = json.load(f)

        n = self.index.ntotal
        assert n == len(self.prompts) == len(self.cond) == len(self.latents), (
            "Inconsistent database: the number of elements differs between files."
            "Rebuild the database using data/init_database.py"
        )

    @torch.no_grad()
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 1,
        exclude_prompt: Optional[str] = None,
    ) -> List[RetrievalResult]:
        q = query.detach().cpu().float().numpy()
        if q.ndim == 1:
            q = q[None, :]

        # reserve for k: insurance if excluded prompt is represented in the database by copies
        k = min(top_k + (5 if exclude_prompt is not None else 0), self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if exclude_prompt is not None and self.prompts[idx] == exclude_prompt:
                continue
            results.append(RetrievalResult(
                index=int(idx),
                prompt=self.prompts[idx],
                cond=self.cond[idx].to(self.device, torch.float32),
                latent=self.latents[idx].to(self.device, torch.float32),
                score=float(score),
            ))
            if len(results) == top_k:
                break

        if not results:
            raise ValueError("Database is empty or search failed.")
        return results

    @torch.no_grad()
    def forward(self, text_embedding: torch.Tensor):
        """
        compatibility with current main.py: returns (cond, latent) top-1
        the first element is tau_phi(p_retr), not h_retr
        h_retr is derived from it by the UNet encoder
        """
        top = self.retrieve(text_embedding, top_k=1)[0]
        return top.cond, top.latent
