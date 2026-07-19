import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # to make work "from src.model..." anywhere

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.RAG.rag import RAG
from src.model.text_encoder.text_encoder import TextEncoder


DB_DIR = PROJECT_ROOT / "data" / "vector_db"

print("Loading encoder and database...")
enc = TextEncoder()
rag = RAG(str(DB_DIR))
 
print(f"\nIn the database {len(rag.prompts)} entries.")
print("Database metadata:", rag.meta)
 
# database is deduplicated during assembly
n_unique = len(set(rag.prompts))
assert n_unique == len(rag.prompts), (
    f"In the database {len(rag.prompts) - n_unique} duplicate prompts — "
    "rebuild the database with the latest version of data/init_database.py"
)
 
# take a real prompt from the database and check that the search finds itself
real_prompt = rag.prompts[0]
print(f"\nSearching for prompt: {real_prompt[:80]!r}")
 
query = enc.encode_query(real_prompt)
results = rag.retrieve(query, top_k=3)
 
print("\nTop-3 results:")
for r in results:
    print(f"  score={r.score:.4f}  prompt={r.prompt[:70]!r}")
 
top = results[0]
print("\nChecking tensor shapes:")
print("  cond  :", tuple(top.cond.shape), "(expecting (77, 1024))")
print("  latent:", tuple(top.latent.shape), "(expecting (4, 64, 64))")
 
assert results[0].prompt == real_prompt, "Prompt not found — something is wrong with the index"
assert results[0].score > 0.99, f"Score too low: {results[0].score}"
assert tuple(top.cond.shape) == (77, 1024), f"Invalid cond shape: {top.cond.shape}"
assert tuple(top.latent.shape) == (4, 64, 64), f"Invalid latent shape: {top.latent.shape}"

# exclude_prompt: when excluding a target, another prompt must come first
excl = rag.retrieve(query, top_k=1, exclude_prompt=real_prompt)
assert excl[0].prompt != real_prompt, "exclude_prompt not working"
print(f"\nWith exclude_prompt, a similar prompt was found: score={excl[0].score:.4f}  {excl[0].prompt[:60]!r}")

print("\n[OK] All checks passed — the database and encoder are working consistently.")
