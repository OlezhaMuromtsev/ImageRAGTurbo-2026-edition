import sys
import argparse
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

def show_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def save_image(img, path):
    Image.fromarray(img).save(path)

def main():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--db", required=True, help="Vector database for RAG")
    parser.add_argument("--unet", required=True, help="Path to trained UNet model")
    parser.add_argument("--discriminator", required=True, help="Path to trained Discriminator model")
    parser.add_argument("--save", action="store_true", help="Save generated image")
    parser.add_argument("--path", type=str, default="generated_image.png", 
                        help="Path to save generated image (default: generated_image.png)")
 
    args = parser.parse_args()
    target_prompt = input()
    text_encoder = TextEncoder()
    unet = UNet(args.unet)
    text_embedding = text_encoder.forward(target_prompt)
    rag = RAG(args.db, unet)
    retrieve_h = rag.forward(text_embedding)
    unet.blend(retrieve_h)
    unet.set_prompt(text_embedding)
    noised_template = torch.rand(32, 32) # size of noised template
    result_img = unet.forward(noised_template).detach().cpu().numpy()
    show_image(result_img)
    if args.save:
        save_image(result_img, args.path)
    return 0

if __name__ == "__main__":
    sys.exit(main())