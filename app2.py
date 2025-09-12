import streamlit as st
from openai import OpenAI
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from diskcache import Cache
import time

# ----------------------------
# 🔧 Setup
# ----------------------------
st.set_page_config(page_title="GPT Semantic Cache Demo", layout="wide")

# Directly set API key here
API_KEY = "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"
client = OpenAI(api_key=API_KEY)

# Embedding model for semantic caching
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384
faiss_index = faiss.IndexFlatIP(embedding_dim)  # inner product = cosine similarity
semantic_cache = []  # store responses aligned with FAISS
disk_cache = Cache("./llm_cache")

# ----------------------------
# 🔧 Functions
# ----------------------------
def generate_response(prompt: str) -> str:
    """Call GPT API for response"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-4o
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content

def cached_gpt_call(prompt: str, threshold=0.80) -> tuple[str, str]:
    """Return response and cache status"""
    query_embedding = embedding_model.encode(
        prompt, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32").reshape(1, -1)

    # Semantic cache check
    if len(semantic_cache) > 0:
        D, I = faiss_index.search(query_embedding, k=1)
        if D[0][0] > threshold:
            return semantic_cache[I[0][0]], f"Semantic Cache Hit ✅ (similarity={D[0][0]:.2f})"

    # Disk cache check
    if prompt in disk_cache:
        return disk_cache[prompt], "Disk Cache Hit ✅"

    # Generate new response
    response = generate_response(prompt)
    faiss_index.add(query_embedding)
    semantic_cache.append(response)
    disk_cache[prompt] = response
    return response, "Cache Miss ❌ (Generated)"

# ----------------------------
# 🔧 Streamlit UI
# ----------------------------
st.title("🚀 GPT Semantic Cache Demo")
st.markdown("Enter a prompt below. Similar prompts will reuse cached results.")

prompt = st.text_area("Enter your prompt:", height=120)

col1, col2 = st.columns([1, 3])
with col1:
    generate_btn = st.button("Generate")

if generate_btn and prompt.strip():
    start = time.time()
    response, status = cached_gpt_call(prompt)
    elapsed = time.time() - start

    st.markdown(f"**Cache Status:** {status}")
    st.markdown(f"**Time Taken:** {elapsed:.2f} sec")
    st.divider()
    st.subheader("Response:")
    st.write(response)

# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from diskcache import Cache
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import time

# # ----------------------------
# # 🔧 Setup
# # ----------------------------
# st.set_page_config(page_title="LLM Semantic Cache Demo", layout="wide")

# @st.cache_resource
# def load_models():
#     MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         torch_dtype=torch.float32,
#         device_map=None
#     )
#     embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#     return tokenizer, model, embedding_model

# tokenizer, model, embedding_model = load_models()

# # FAISS index + cache
# embedding_dim = 384
# faiss_index = faiss.IndexFlatIP(embedding_dim)  # cosine similarity via inner product
# semantic_cache = []  # store responses aligned with FAISS
# disk_cache = Cache("./llm_cache")

# # ----------------------------
# # 🔧 Functions
# # ----------------------------
# def generate_response(prompt: str) -> str:
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=150,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# def cached_llm_call(prompt: str, threshold=0.80) -> tuple[str, str]:
#     """Return response and cache status."""
#     # Always get normalized numpy float32 vector
#     query_embedding = embedding_model.encode(
#         prompt, normalize_embeddings=True, convert_to_numpy=True
#     ).astype("float32").reshape(1, -1)

#     # Semantic cache check
#     if len(semantic_cache) > 0:
#         D, I = faiss_index.search(query_embedding, k=1)
#         if D[0][0] > threshold:  # cosine similarity (1 = identical)
#             matched_response = semantic_cache[I[0][0]]
#             return matched_response, f"Semantic Cache Hit ✅ (similarity={D[0][0]:.2f})"

#     # Disk cache check
#     if prompt in disk_cache:
#         return disk_cache[prompt], "Disk Cache Hit ✅"

#     # Generate new response
#     response = generate_response(prompt)

#     # Store in FAISS + semantic cache
#     faiss_index.add(query_embedding)
#     semantic_cache.append(response)

#     # Store in disk cache
#     disk_cache[prompt] = response
#     return response, "Cache Miss ❌ (Generated)"

# # ----------------------------
# # 🔧 Streamlit UI
# # ----------------------------
# st.title("🚀 LLM Semantic Cache Demo")
# st.markdown("Enter a prompt below. Similar prompts will reuse cached results.")

# prompt = st.text_area("Enter your prompt:", height=120)

# col1, col2 = st.columns([1, 3])
# with col1:
#     generate_btn = st.button("Generate")

# if generate_btn and prompt.strip():
#     start = time.time()
#     response, status = cached_llm_call(prompt)
#     elapsed = time.time() - start

#     st.markdown(f"**Cache Status:** {status}")
#     st.markdown(f"**Time Taken:** {elapsed:.2f} sec")
#     st.divider()
#     st.subheader("Response:")
#     st.write(response)
