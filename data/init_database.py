import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # to make work "from src.model..." anywhere

import faiss
import numpy as np
import torch
from datasets import load_dataset
from diffusers import AutoencoderKL
from torchvision import transforms
from src.model.text_encoder.encoder import TextEncoder


MODEL_ID = "sd2-community/stable-diffusion-2-1-base"  # from https://huggingface.co/sd2-community/stable-diffusion-2-1-base
DATASET_ID = "yhshin1020/coco-img-caption-pairs"
IMAGE_SIZE = 512  # 512x512x3 -> latent 4x64x64 for VAE
DB_SCHEMA_VERSION = 1

 
def build_vector_database(
    output_dir: str = "data/vector_db",
    num_pairs: int = 1000,
    batch_size: int = 16,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    print("Initializing text encoder and VAE...")
    text_encoder = TextEncoder(model_id=MODEL_ID, device=device)
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
 
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [0,1] -> [-1,1]
    ])

    print(f"Loading dataset {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train")

    # Deduplication: avoiding prompt duplication
    seen = set()
    records = []
    for ex in dataset:
        cap = ex["caption"].strip() if isinstance(ex["caption"], str) else ""
        if not cap or cap in seen:
            continue
        seen.add(cap)
        records.append((cap, ex["image"]))
        if len(records) >= num_pairs:
            break
    prompts = [c for c, _ in records]
    print(f"Preparing database with {len(records)} unique text-image pairs...")

    query_chunks, cond_chunks, latent_chunks = [], [], []

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        batch_prompts = [c for c, _ in batch]
        batch_images = [img.convert("RGB") for _, img in batch]
        query, cond = text_encoder(batch_prompts)  # 2 text views: (B, D), (B, 77, D)

        """z0_retr = E(x_retr)"""
        pixels = torch.stack([preprocess(img) for img in batch_images]).to(device)  # preprocess for VAE
        # scaling_factor (0.18215) - multiplier for the correct scale for UNet
        with torch.no_grad():
            latent = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor  # .mode() for determined cache instead of random point in different base assemblies
 
        query_chunks.append(query.cpu().float().numpy())
        cond_chunks.append(cond.cpu().half())
        latent_chunks.append(latent.cpu().half())
 
        done = min(i + batch_size, len(records))
        if done % (batch_size * 5) == 0 or done == len(records):
            print(f"Processed {done} / {len(records)}...")
 
    db_queries = np.vstack(query_chunks).astype("float32")
    cond_embeddings = torch.cat(cond_chunks)  # (N, 77, D)
    latents = torch.cat(latent_chunks)  # (N, 4, 64, 64)
 
    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(db_queries.shape[1])  # IndexFlatIP instead of ScaNN from article for precision, can be replaced if we want
    index.add(db_queries)
 
    print(f"Saving database to {output_dir}/ ...")
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))
    torch.save(cond_embeddings, os.path.join(output_dir, "text_cond.pt"))
    torch.save(latents, os.path.join(output_dir, "latents.pt"))
    with open(os.path.join(output_dir, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False)

    # database passport, RAG uses it to verify if database compiled with code:
    meta = {
        "schema_version": DB_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "num_items": len(prompts),
        "embed_dim": int(db_queries.shape[1]),
        "cond_shape": list(cond_embeddings.shape[1:]),
        "latent_shape": list(latents.shape[1:]),
        "image_size": IMAGE_SIZE,
        "vae_scaling_factor": float(vae.config.scaling_factor),
        "dataset": DATASET_ID,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
 
    print(f"Success! Database ready: {output_dir}/")
 
 
if __name__ == "__main__":
    build_vector_database()
