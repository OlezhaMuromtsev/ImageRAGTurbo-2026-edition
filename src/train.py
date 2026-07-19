import sys
import argparse
import numpy as np
import torch
from utils.config import Config
from model.RAG.rag import RAG

def main():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--config", required=True, help="Path to train config")
    parser.add_argument("--path", type=str, default="model.pth", 
                        help="Path to save trained model")

    args = parser.parse_args()
    config = Config(args.config)
    teacher = config.get_teacher()
    student = config.get_student()
    trainer = config.get_trainer()
    trained = trainer.train(teacher, student)
    checkpoint = {
        'model_state_dict': trained.state_dict()
    }
    torch.save(checkpoint, args.path)
    return 0

if __name__ == "__main__":
    sys.exit(main())