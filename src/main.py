import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from model.RAG.rag import RAG
from model.text_encoder.encoder import TextEncoder
from model.denoiser.bottleneck import get_stable_diffusion_bottleneck
from model.denoiser.decoder import get_stable_diffusion_decoder
from model.denoiser.encoder import get_stable_diffusion_encoder

def show_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def save_image(img, path):
    Image.fromarray(img).save(path)

def main():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--db", required=True, help="Vector database for RAG")
    parser.add_argument("--save", action="store_true", help="Save generated image")
    parser.add_argument("--path", type=str, default="generated_image.png", 
                        help="Path to save generated image (default: generated_image.png)")
 
    args = parser.parse_args()
    target_prompt = input()
    text_encoder = TextEncoder()
    text_embedding = text_encoder.forward(target_prompt)
    rag = RAG(args.db)
    retrieve_h = rag.forward(text_embedding)
    encoder = get_stable_diffusion_encoder()
    bottleneck = get_stable_diffusion_bottleneck(encoder)
    decoder = get_stable_diffusion_decoder(encoder)
    with torch.inference_mode():
        encoder_output = encoder(text_embedding)
        bottleneck_output = bottleneck(encoder_output, retrieve=retrieve_h)
        decoder_output = decoder(bottleneck_output)
    result_img = decoder_output.image_tensor.detach().cpu().numpy()
    show_image(result_img)
    if args.save:
        save_image(result_img, args.path)
    return 0

if __name__ == "__main__":
    sys.exit(main())