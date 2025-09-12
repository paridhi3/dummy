# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from diskcache import Cache
# from functools import lru_cache
# import hashlib
# import time

# # Use the smallest open-access model
# MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

# # Load tokenizer and model (no auth required)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float32,  # SmolLM2 uses float32 for compatibility
#     device_map="auto"
# )

# # Persistent disk cache
# disk_cache = Cache("./llm_cache")

# def hash_prompt(prompt: str) -> str:
#     """Create a unique hash for each prompt."""
#     return hashlib.sha256(prompt.encode()).hexdigest()

# @lru_cache(maxsize=128) # most recent 128 function calls (based on the prompt string) stored in-memory (specifically in Python process’s RAM.)
# def in_memory_cache(prompt: str) -> str:
#     """In-memory cache using LRU."""
#     return generate_response(prompt)

# def generate_response(prompt: str) -> str:
#     """Generate response from LLM."""
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=100,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# def cached_llm_call(prompt: str) -> str:
#     """Check disk cache first, then in-memory, then generate."""
#     key = hash_prompt(prompt)
#     if key in disk_cache:
#         print("[Disk Cache Hit ✅]")
#         return disk_cache[key]
#     else:
#         print("[Cache Miss ❌] Generating...")
#         response = in_memory_cache(prompt)
#         disk_cache[key] = response
#         return response

# if __name__ == "__main__":
#     prompts = [
#         "Explain quantum computing in simple terms.",
#         "Explain quantum computing in simple terms.",
#         "What is the capital of France?",
#         "What is the capital of France?"
#     ]

#     for prompt in prompts:
#         print(f"\nPrompt: {prompt}")
#         start = time.time()
#         output = cached_llm_call(prompt)
#         print(f"Response: {output[:200]}...")
#         print(f"Time taken: {time.time() - start:.2f}s")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diskcache import Cache
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time

# Use the smallest open-access model
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

# Load tokenizer and model (no auth required)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
)

# Load embedding model for semantic caching
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and fast

# FAISS index for semantic search
embedding_dim = 384  # Dimension for MiniLM embeddings - vector of 384 numbers, each representing a feature of the sentence’s meaning.
faiss_index = faiss.IndexFlatL2(embedding_dim)
semantic_cache = []  # Stores (embedding, response) pairs

# Persistent disk cache (optional fallback)
disk_cache = Cache("./llm_cache")

# Exact key hashing and in-memory cache
# def hash_prompt(prompt: str) -> str:
#     return hashlib.sha256(prompt.encode()).hexdigest()

# @lru_cache(maxsize=128)
# def in_memory_cache(prompt: str) -> str:
#     return generate_response(prompt)

def generate_response(prompt: str) -> str:
    """Generate response from LLM."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def cached_llm_call(prompt: str, threshold=0.85) -> str:
    """Check semantic cache first, then generate."""
    query_embedding = embedding_model.encode(prompt)
    if len(semantic_cache) > 0:
        D, I = faiss_index.search(np.array([query_embedding]), k=1)
        if D[0][0] < (1 - threshold):  # Lower distance = higher similarity
            print("[Semantic Cache Hit ✅]")
            return semantic_cache[I[0][0]][1]

    print("[Cache Miss ❌] Generating...")
    response = generate_response(prompt)
    faiss_index.add(np.array([query_embedding]))
    semantic_cache.append((query_embedding, response))
    return response

if __name__ == "__main__":
    prompts = [
        "Explain quantum computing in simple terms.",
        "Explain quantum computing in simple terms.",
        "What is the capital of France?",
        "Which city is France's capital?"  # Semantic variation
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        start = time.time()
        output = cached_llm_call(prompt)
        print(f"Response: {output[:200]}...")
        print(f"Time taken: {time.time() - start:.2f}s")
