import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import redis
import time
import uuid

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
semantic_cache = []  # Stores (embedding, redis_key, original_prompt)

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
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🔁 Semantic + Redis cache
def normalize_prompt(prompt: str) -> str:
    return prompt.strip().lower()

# def cached_llm_call(prompt: str, threshold=0.70) -> tuple[str, str]:
#     normalized = normalize_prompt(prompt)
#     exact_key = f"llm:exact:{hash(normalized)}"
#     redis_client.hincrby("prompt_usage", normalized, 1)

#     # 🔍 Exact match check
#     cached_response = redis_client.get(exact_key)
#     if cached_response:
#         redis_client.incr("cache_hits")
#         return cached_response.decode(), "✅ Exact Match Cache Hit"

#     # 🧠 Semantic match fallback
#     query_embedding = np.array(embedding_model.encode(normalized), dtype=np.float32)
#     if len(semantic_cache) > 0:
#         D, I = faiss_index.search(np.array([query_embedding]), k=1)
#         st.write(f"FAISS distance: {D[0][0]:.4f}")
#         if D[0][0] < (1 - threshold):
#             semantic_key = semantic_cache[I[0][0]][1]
#             matched_prompt = semantic_cache[I[0][0]][2]
#             cached_response = redis_client.get(semantic_key)
#             if cached_response:
#                 redis_client.incr("cache_hits")
#                 st.caption(f"Matched semantic prompt: `{matched_prompt}`")
#                 return cached_response.decode(), "✅ Semantic Cache Hit"

#     # ❌ Cache miss
#     redis_client.incr("cache_misses")
#     start = time.time()
#     response = generate_response(prompt)
#     duration = time.time() - start
#     redis_client.lpush("response_times", duration)

#     # Store exact match
#     redis_client.set(exact_key, response, ex=3600)

#     # Store semantic match
#     semantic_key = f"llm:semantic:{uuid.uuid4().hex}"
#     redis_client.set(semantic_key, response, ex=3600)
#     faiss_index.add(np.array([query_embedding]))
#     semantic_cache.append((query_embedding, semantic_key, normalized))

#     return response, "❌ Cache Miss"

def cached_llm_call(prompt: str, threshold=0.70) -> tuple[str, str, float]:
    start = time.time()
    normalized = normalize_prompt(prompt)
    exact_key = f"llm:exact:{hash(normalized)}"
    redis_client.hincrby("prompt_usage", normalized, 1)

    # 🔍 Exact match check
    cached_response = redis_client.get(exact_key)
    if cached_response:
        redis_client.incr("cache_hits")
        duration = time.time() - start
        return cached_response.decode(), "✅ Exact Match Cache Hit", duration

    # 🧠 Semantic match fallback
    query_embedding = np.array(embedding_model.encode(normalized), dtype=np.float32)
    if len(semantic_cache) > 0:
        D, I = faiss_index.search(np.array([query_embedding]), k=1)
        st.write(f"FAISS distance: {D[0][0]:.4f}")
        if D[0][0] < (1 - threshold):
            semantic_key = semantic_cache[I[0][0]][1]
            matched_prompt = semantic_cache[I[0][0]][2]
            cached_response = redis_client.get(semantic_key)
            if cached_response:
                redis_client.incr("cache_hits")
                duration = time.time() - start
                st.caption(f"Matched semantic prompt: `{matched_prompt}`")
                return cached_response.decode(), "✅ Semantic Cache Hit", duration

    # ❌ Cache miss
    redis_client.incr("cache_misses")
    response = generate_response(prompt)
    duration = time.time() - start
    redis_client.lpush("response_times", duration)

    # Store exact match
    redis_client.set(exact_key, response, ex=3600)

    # Store semantic match
    semantic_key = f"llm:semantic:{uuid.uuid4().hex}"
    redis_client.set(semantic_key, response, ex=3600)
    faiss_index.add(np.array([query_embedding]))
    semantic_cache.append((query_embedding, semantic_key, normalized))

    return response, "❌ Cache Miss", duration

# 🌐 Streamlit UI
st.set_page_config(page_title="LLM Cache Demo using Redis", layout="centered")
st.title("LLM Cache Demo using Redis")
st.markdown("Enter a prompt below to interact with the model. Cached responses will be reused when possible.")

prompt = st.text_area("💬 Prompt", placeholder="Type your question here...")

if st.button("Generate Response"):
    if prompt.strip():
        with st.spinner("Generating response..."):
            response, cache_status, duration = cached_llm_call(prompt)
            st.success("Response generated!")
            st.markdown(f"**Response:**\n\n{response}")
            st.info(f"**Cache Status:** {cache_status} — ⏱️ Time Taken: {duration:.2f}s")
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
