import sys
import argparse
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DB_DIR = PROJECT_ROOT / "data" / "vector_db"


from src.model import StableDiffusionUNet
from src.model.text_encoder import TextEncoder
from src.model.RAG import RAG





def show_image(img):
    img = img.images[0]
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def save_image(img, path):
    img.images[0].save(path)

@torch.inference_mode()
def main():

    encoder_weight = 1.0

    target_prompt = input()
    #Инициализация объектов
    text_encoder = TextEncoder()
    unet = StableDiffusionUNet()
    rag = RAG(
        db_dir = str(DB_DIR),
        device = 'cuda',
    )
    #Получили представление промпта
    query_embedding, text_embedding = text_encoder.forward(target_prompt)
    print(f'вывод текст энкодера {query_embedding.shape}')

    #Вывод РАГ-поиска и латента
    rag_out = rag.retrieve(query_embedding, top_k = 1)
    rag_cond = rag_out[0].cond
    rag_latent = rag_out[0].latent

    print(f'раг латент {rag_latent.shape}')
    rag_h_out = unet.forward(
        text_embedding=rag_cond.unsqueeze(0),
        latent=rag_latent.unsqueeze(0),
        timestep=0,
    )
    rag_h = rag_h_out.hidden_states
    print(f'раг h_space{rag_h.shape}')
    target_h = unet.forward(text_embedding, latent = None, timestep = None)
    #смешивание
    target_h_blended = unet.blending(rag_h, target_h, encoder_weight)
    #Денойз оканчательный
    print(f'после блэндинга {target_h_blended.hidden_states.shape}')
    del rag_h
    del rag_out
    del rag_cond
    del rag_latent

    torch.cuda.empty_cache()

    result_img = unet.decoder(target_h_blended)
    show_image(result_img)
    save_image('/result')
    return 0

if __name__ == "__main__":
    raise SystemExit(main())