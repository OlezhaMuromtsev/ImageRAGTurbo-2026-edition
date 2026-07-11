import yaml
from pathlib import Path
import torch
from model.train.trainer import Trainer

class Config:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self.load()
    
    def load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")
        
        with open(self._path, 'r', encoding='utf-8') as f:
            if self._path.suffix in ('.yaml', '.yml'):
                data = yaml.safe_load(f) or {}
            else:
                raise ValueError(f"Unsupported config format: {self._path.suffix}")
        

    def get_teacher(self) -> torch.nn.Module:
        pass
    
    def get_student(self) -> torch.nn.Module:
        pass
    
    def get_trainer(self) -> Trainer:
        pass