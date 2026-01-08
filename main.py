from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
import os
import numpy as np
import requests

# AI & Search Libraries
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)

# ---------------------------------------------------------
# 1. LOAD CSV & DETECT COLUMNS
# ---------------------------------------------------------
try:
    df = pd.read_csv("amazon_reviews.csv")
    df.columns = df.columns.str.strip() # Remove spaces from column names
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame()

PRODUCT_COL = None
REVIEW_COL = None

def _detect_columns():
    global PRODUCT_COL, REVIEW_COL
    
    # Priority 1: Exact matches for common names
    product_candidates = ["product_id", "product_link", "asin", "productId"]
    review_candidates = ["review_content", "review_text", "reviewText", "review_body"]

    for col in product_candidates:
        if col in df.columns:
            PRODUCT_COL = col
            break
            
    for col in review_candidates:
        if col in df.columns:
            REVIEW_COL = col
            break

    # Priority 2: Fallback search
    if not PRODUCT_COL:
        for col in df.columns:
            if "product" in col.lower() and "id" in col.lower():
                PRODUCT_COL = col
                break
    
    if not REVIEW_COL:
        for col in df.columns:
            if "content" in col.lower() or "text" in col.lower():
                REVIEW_COL = col
                break

_detect_columns()
print(f"✅ Loaded Columns - Product: {PRODUCT_COL}, Review: {REVIEW_COL}")

# ---------------------------------------------------------
# 2. INITIALIZE AI MODELS (FAISS & EMBEDDINGS)
# ---------------------------------------------------------
print("⏳ Loading AI Models... (This may take a moment)")
try:
    # Load a small, fast model for converting text to numbers
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding Model Loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    embedder = None

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_product_id(link):
    if not link: return None
    # Regex for Amazon ASINs (10 chars, alphanumeric)
    patterns = [r"/dp/([A-Z0-9]{10})", r"/gp/product/([A-Z0-9]{10})", r"[?&]ASIN=([A-Z0-9]{10})"]
    for p in patterns:
        match = re.search(p, link)
        if match: return match.group(1)
    
    # Fallback for short links
    parts = link.rstrip("/\n").split("/")
    last = parts[-1].split("?")[0]
    if re.fullmatch(r"[A-Z0-9]{10}", last): return last
    return None

def get_reviews_by_product_id(product_id):
    if PRODUCT_COL is None or REVIEW_COL is None: return []
    
    # Filter DataFrame for the specific product ID
    mask = df[PRODUCT_COL].astype(str).str.upper() == str(product_id).upper()
    filtered = df[mask]
    
    if filtered.empty: return []
    return filtered[REVIEW_COL].dropna().astype(str).tolist()

def call_gemini_api(prompt: str):
    key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    if not key:
        return "Error: GEMINI_API_KEY not set in environment."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048 
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        if "candidates" in j and j["candidates"]:
             return j["candidates"][0]["content"]["parts"][0]["text"]
        return "No response content from AI."
    except Exception as e:
        return f"AI Connection Error: {e}"

# ---------------------------------------------------------
# 4. ROUTES
# ---------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    link = data.get("link")
    question = (data.get("question") or "").strip()

    # 1. Get Reviews
    product_id = extract_product_id(link)
    if not product_id: return jsonify({"answer": "Could not extract a valid Product ID from that link."})

    reviews = get_reviews_by_product_id(product_id)
    if not reviews: return jsonify({"answer": "No reviews found in the dataset for this product."})

    # 2. RAG SEARCH WITH FAISS
    # We only use FAISS if we have more than a few reviews
    retrieved_reviews = []
    
    if embedder and len(reviews) > 0:
        try:
            # A. Create Embeddings for the reviews found
            review_embeddings = embedder.encode(reviews, convert_to_numpy=True)
            
            # B. Normalize embeddings (crucial for accurate similarity)
            norms = np.linalg.norm(review_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            review_embeddings = review_embeddings / norms

            # C. Create FAISS Index
            dimension = review_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension) # Inner Product (Cosine Similarity)
            index.add(review_embeddings.astype(np.float32))

            # D. Embed the User's Question
            question_embedding = embedder.encode([question], convert_to_numpy=True)
            question_embedding = question_embedding / (np.linalg.norm(question_embedding, axis=1, keepdims=True) + 1e-10)

            # E. Search for top 10 most relevant reviews
            top_k = min(10, len(reviews))
            distances, indices = index.search(question_embedding.astype(np.float32), top_k)
            
            # F. Retrieve the actual text
            for idx in indices[0]:
                if idx < len(reviews):
                    retrieved_reviews.append(reviews[idx])
                    
        except Exception as e:
            print(f"FAISS Search Failed: {e}")
            retrieved_reviews = reviews[:10] # Fallback if FAISS fails
    else:
        retrieved_reviews = reviews[:10]

    # 3. GENERATE ANSWER
    context_text = "\n\n".join(retrieved_reviews)
    
    prompt = (
        f"You are an expert shopping assistant. Answer the user's question based ONLY on the following reviews.\n"
        f"If the reviews don't contain the answer, say 'I couldn't find that information in the reviews'.\n\n"
        f"add a final recommendation whether to buy or not to but'.\n\n"
        f"--- REVIEWS ---\n{context_text}\n---------------\n\n"

        f"Question: {question}\nAnswer:"
    )

    ai_response = call_gemini_api(prompt)

    return jsonify({"answer": ai_response})

if __name__ == "__main__":
    app.run(debug=True)