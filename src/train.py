import sys
import argparse
import numpy as np
import torch
from utils.config import Config

def main():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--config", required=True, help="Path to train config")
    parser.add_argument("--path", type=str, default="model.pth", 
                        help="Path to save trained model")
    parser.add_argument("--student_path", type=str, default="S.pth", 
                        help="Path to save trained model")
    parser.add_argument("--discriminator_path", type=str, default="D.pth", 
                        help="Path to save trained model")
    args = parser.parse_args()
    config = Config(args.config)
    teacher = config.get_teacher()
    student = config.get_student()
    trainer = config.get_trainer()
    trained_student_weights, trained_discriminator_weights = trainer.train(teacher, student)
    torch.save(trained_student_weights, args.student_path)
    torch.save(trained_discriminator_weights, args.discriminator_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())