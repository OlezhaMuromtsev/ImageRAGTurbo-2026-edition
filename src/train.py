import sys
import argparse
import numpy as np
import torch

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
    torch.save(trained.state_dict(), args.path)
    return 0

if __name__ == "__main__":
    sys.exit(main())