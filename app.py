# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import redis
# import time
# import json

# # 🔧 Model setup
# MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float32,
#     device_map="auto"
# )

# # 🧠 Embedding model for semantic similarity
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# embedding_dim = 384
# faiss_index = faiss.IndexFlatL2(embedding_dim)
# semantic_cache = []  # Stores (embedding, redis_key)

# # 🗄️ Redis setup
# REDIS_HOST = "redis-10800.c61.us-east-1-3.ec2.redns.redis-cloud.com"
# REDIS_PORT = 10800
# REDIS_PASSWORD = "4YCGmIOG6QdrHvnHnvWflMyCVwtRw66F"

# redis_client = redis.Redis(
#     host=REDIS_HOST,
#     port=REDIS_PORT,
#     password=REDIS_PASSWORD,
# )

# # 🧾 Response generation
# def generate_response(prompt: str) -> str:
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=100,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # 🔁 Semantic + Redis cache
# def cached_llm_call(prompt: str, threshold=0.85) -> str:
#     query_embedding = np.array(embedding_model.encode(prompt), dtype=np.float32)

#     # Track prompt usage
#     redis_client.hincrby("prompt_usage", prompt, 1)

#     # Semantic search
#     if len(semantic_cache) > 0:
#         D, I = faiss_index.search(np.array([query_embedding]), k=1)
#         if D[0][0] < (1 - threshold):
#             redis_key = semantic_cache[I[0][0]][1]
#             cached_response = redis_client.get(redis_key)
#             if cached_response:
#                 redis_client.incr("cache_hits")  # ✅ Cache hit
#                 print("[Semantic Redis Cache Hit ✅]")
#                 return cached_response.decode()

#     redis_client.incr("cache_misses")  # ✅ Cache miss
#     print("[Cache Miss ❌] Generating...")

#     # Track latency
#     start = time.time()
#     response = generate_response(prompt)
#     duration = time.time() - start
#     redis_client.lpush("response_times", duration)

#     # Store in Redis
#     redis_key = f"llm:{hash(prompt)}"
#     redis_client.set(redis_key, response, ex=3600)

#     # Update semantic cache
#     faiss_index.add(np.array([query_embedding]))
#     semantic_cache.append((query_embedding, redis_key))

#     return response

# # 🧪 Test prompts
# if __name__ == "__main__":
#     prompts = [
#         "Explain quantum computing in simple terms.",
#         "Explain quantum computing in simple terms.",
#         "What is the capital of France?",
#         "Which city is France's capital?"
#     ]

#     for prompt in prompts:
#         print(f"\nPrompt: {prompt}")
#         output = cached_llm_call(prompt)
#         print(f"Response: {output[:200]}...")

#     # Summary metrics
#     hits = int(redis_client.get("cache_hits") or 0)
#     misses = int(redis_client.get("cache_misses") or 0)
#     total = hits + misses
#     if total > 0:
#         print(f"\n🔢 Cache Hit Rate: {hits / total:.2%} ({hits} hits, {misses} misses)")

#     times = redis_client.lrange("response_times", 0, -1)
#     if times:
#         avg_time = sum(map(float, times)) / len(times)
#         print(f"⏱️ Average Response Time: {avg_time:.2f}s")
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import redis
import time

# 🔧 Model setup
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=None
)

# 🧠 Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384
faiss_index = faiss.IndexFlatL2(embedding_dim)
semantic_cache = []

# 🗄️ Redis setup
REDIS_HOST = "redis-10800.c61.us-east-1-3.ec2.redns.redis-cloud.com"
REDIS_PORT = 10800
REDIS_PASSWORD = "4YCGmIOG6QdrHvnHnvWflMyCVwtRw66F"

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
)

# 🧾 Response generation
def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🔁 Semantic + Redis cache
def cached_llm_call(prompt: str, threshold=0.85) -> tuple[str, str]:
    query_embedding = np.array(embedding_model.encode(prompt), dtype=np.float32)
    redis_client.hincrby("prompt_usage", prompt, 1)

    if len(semantic_cache) > 0:
        D, I = faiss_index.search(np.array([query_embedding]), k=1)
        if D[0][0] < (1 - threshold):
            redis_key = semantic_cache[I[0][0]][1]
            cached_response = redis_client.get(redis_key)
            if cached_response:
                redis_client.incr("cache_hits")
                return cached_response.decode(), "✅ Cache Hit"
    
    redis_client.incr("cache_misses")
    start = time.time()
    response = generate_response(prompt)
    duration = time.time() - start
    redis_client.lpush("response_times", duration)

    redis_key = f"llm:{hash(prompt)}"
    redis_client.set(redis_key, response, ex=3600)

    faiss_index.add(np.array([query_embedding]))
    semantic_cache.append((query_embedding, redis_key))

    return response, "❌ Cache Miss"

# 🌐 Streamlit UI
st.set_page_config(page_title="LLM Cache Demo using Redis", layout="centered")
st.title("🧠 LLM Cache & Memory Management")
st.markdown("Enter a prompt below to interact with the model. Cached responses will be reused when possible.")

prompt = st.text_area("💬 Prompt", placeholder="Type your question here...")

if st.button("Generate Response"):
    if prompt.strip():
        with st.spinner("Generating response..."):
            response, cache_status = cached_llm_call(prompt)
            st.success("Response generated!")
            st.markdown(f"**Response:**\n\n{response}")
            st.info(f"**Cache Status:** {cache_status}")
    else:
        st.warning("Please enter a prompt.")

# 📊 Metrics toggle
st.divider()
if st.button("Show Metrics"):
    st.subheader("📈 Cache Metrics")

    hits = int(redis_client.get("cache_hits") or 0)
    misses = int(redis_client.get("cache_misses") or 0)
    total = hits + misses
    hit_rate = (hits / total * 100) if total > 0 else 0

    st.metric("Cache Hit Rate", f"{hit_rate:.2f}%")
    st.metric("Cache Hits", hits)
    st.metric("Cache Misses", misses)

    times = redis_client.lrange("response_times", 0, -1)
    if times:
        avg_time = sum(map(float, times)) / len(times)
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
