from datasets import load_dataset
from typing import List, Iterator, Optional

class LAIONAestheticDataset:
    def __init__(
        self,
        split: str = "train",
        max_prompts: Optional[int] = None,
        seed: int = 42,
        batch_size: int = 32,
        cache_dir: str = "./laion_cache"
    ):
        self.split = split
        self.max_prompts = max_prompts
        self.seed = seed
        self.batch_size = batch_size
        
        print(f"Loading LAION Aesthetic 6.25+ ({split})...")
        self.dataset = load_dataset(
            "xingjianleng/laion_aesthetics_v2_6.25plus",
            split=split,
            streaming=False,
            cache_dir=cache_dir
        )
        
        if max_prompts:
            self.dataset = self.dataset.select(
                range(min(max_prompts, len(self.dataset)))
            )
        
        print(f"Loaded {len(self.dataset)} prompts")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def get_prompts_batch(self, batch_size: Optional[int] = None) -> Iterator[List[str]]:
        if batch_size is None:
            batch_size = self.batch_size
            
        for i in range(0, len(self.dataset), batch_size):
            batch = self.dataset[i:i + batch_size]['TEXT']
            yield batch

