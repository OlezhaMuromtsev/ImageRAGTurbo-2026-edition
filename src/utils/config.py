import yaml
from pathlib import Path
import importlib
import torch.nn as nn
from model.train.trainer import Trainer
from model import StableDiffusionUNet
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

class Config:
    """
    Parses a YAML configuration file and provides factory methods
    to instantiate the teacher model, student model, and trainer.
    
    Expected YAML structure
    -----------------------
    trainer:
      alpha: float
      beta: float
      epochs: int
      lr: float
      student_steps: list[float]
      alphas: list[float]        # noise schedule alphas
      sigmas: list[float]        # noise schedule sigmas
      teacher_steps: list[float]
      latent_size: int
      db_path: str               # path to the RAG database
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._config: dict = {}
        self.load()

    def load(self) -> None:
        """Load and validate the YAML configuration file."""
        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")

        with open(self._path, 'r', encoding='utf-8') as f:
            if self._path.suffix in ('.yaml', '.yml'):
                self._config = yaml.safe_load(f) or {}
            else:
                raise ValueError(f"Unsupported config format: {self._path.suffix}")

        # Basic sanity checks
        required_sections = ['trainer']
        for section in required_sections:
            if section not in self._config:
                raise KeyError(f"Missing required section '{section}' in config file.")
            if not isinstance(self._config[section], dict):
                raise TypeError(f"Section '{section}' must be a dictionary.")

    def get_teacher(self) -> nn.Module:
        return StableDiffusionUNet() # гы, у нас пока он учит сам себя =)

    def get_student(self) -> nn.Module:
        return StableDiffusionUNet()

    def get_trainer(self) -> Trainer:
        """Create and return the Trainer instance with all hyperparameters."""
        trainer_cfg = self._config['trainer']
        required_keys = [
            'alpha', 'beta', 'epochs', 'lr', 'student_steps',
            'alphas', 'sigmas', 'teacher_steps', 'latent_size', 'db_path'
        ]
        for key in required_keys:
            if key not in trainer_cfg:
                raise KeyError(f"Missing required trainer parameter '{key}' in config.")

        # The Trainer expects these parameters as keyword arguments.
        return Trainer(
            alpha=trainer_cfg['alpha'],
            beta=trainer_cfg['beta'],
            epochs=trainer_cfg['epochs'],
            lr=trainer_cfg['lr'],
            student_steps=torch.tensor(trainer_cfg['student_steps']),
            alphas=trainer_cfg['alphas'],
            sigmas=trainer_cfg['sigmas'],
            teacher_steps=torch.tensor(trainer_cfg['teacher_steps']),
            latent_size=trainer_cfg['latent_size'],
            db_path=(PROJECT_ROOT / trainer_cfg['db_path']),
        )