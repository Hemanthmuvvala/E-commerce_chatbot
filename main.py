from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
import os
import numpy as np
import requests

app = Flask(__name__)

# Load the CSV
try:
    df = pd.read_csv("amazon_reviews.csv")
    
    df.columns = df.columns.str.strip() 
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame()


PRODUCT_COL = None
REVIEW_COL = None

def _detect_columns():
    global PRODUCT_COL, REVIEW_COL
    
    
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
app.logger.info(f"Using Product Column: {PRODUCT_COL}")
app.logger.info(f"Using Review Column: {REVIEW_COL}")


try:
    from sentence_transformers import SentenceTransformer
    import faiss
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    embedder = None
    faiss = None


def extract_product_id(link):
    if not link:
        return None
    
    patterns = [
        r"/dp/([A-Z0-9]{10})", 
        r"/gp/product/([A-Z0-9]{10})", 
        r"[?&]ASIN=([A-Z0-9]{10})"
    ]
    for p in patterns:
        match = re.search(p, link)
        if match:
            return match.group(1)

    # Fallback: try to grab the last segment if it looks like an ID
    parts = link.rstrip("/\n").split("/")
    last = parts[-1].split("?")[0]
    if re.fullmatch(r"[A-Z0-9]{10}", last):
        return last
    return None

def get_reviews_by_product_id(product_id):
    if PRODUCT_COL is None or REVIEW_COL is None:
        return []

    # Case-insensitive filtering
    # Ensure we convert both column and query to string
    mask = df[PRODUCT_COL].astype(str).str.upper() == str(product_id).upper()
    filtered = df[mask]

    if filtered.empty:
        return []

    # Drop NaNs and convert to list
    return filtered[REVIEW_COL].dropna().astype(str).tolist()

# ---------------------------------------------------------
# Gemini API Wrapper (Fixed)
# ---------------------------------------------------------
def call_gemini_api(prompt: str):
    key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    if not key:
        return "[Gemini API key not set. This is a stub response.]\n\nContext used:\n" + prompt[:500]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": key
    }
    
    body = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        
        # Parse standard Gemini response
        if "candidates" in j and j["candidates"]:
            candidate = j["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0]["text"]
        
        return "No content returned by Gemini."

    except Exception as e:
        return f"[Gemini call failed: {e}]"

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    link = data.get("link")
    question = (data.get("question") or "").strip()

    product_id = extract_product_id(link)

    if not product_id:
        return jsonify({"answer": "Could not extract a valid Product ID (ASIN) from the link."})

    reviews = get_reviews_by_product_id(product_id)

    if not reviews:
        return jsonify({"answer": f"No reviews found in CSV for Product ID: {product_id}"})

    # -----------------------------------------------------
    # RAG Logic (Vector Search vs Simple Fallback)
    # -----------------------------------------------------
    retrieved = []
    
    if embedder is not None and faiss is not None and len(reviews) > 0:
        try:
            # Create embeddings for these specific reviews
            review_embeddings = embedder.encode(reviews, convert_to_numpy=True)
            
            # Normalize
            norms = np.linalg.norm(review_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            review_embeddings = review_embeddings / norms

            # Index
            d = review_embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(review_embeddings.astype(np.float32))

            # Query
            q_emb = embedder.encode([question], convert_to_numpy=True)
            q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)

            # Search Top 5
            k = min(5, len(reviews))
            D, I = index.search(q_emb.astype(np.float32), k)
            
            # Fetch text
            for idx in I[0]:
                if idx < len(reviews):
                    retrieved.append(reviews[idx])
        except Exception as e:
            print(f"FAISS error: {e}")
            retrieved = reviews[:5] # Fallback
    else:
        # Simple fallback if no AI libraries installed
        retrieved = reviews[:5]


    context_text = "\n\n".join(retrieved)
    
    # Prompt Engineering
    prompt = (
        f"You are a helpful shopping assistant. "
        f"Answer the user's question strictly based on the review excerpts provided below.Elaborate the answer in detail.And also specify is it good for students,employees and which kind of does it suites.Give in 100 words\n\n"
        f"add a final reccomendation to strongly reccommend to buy or not buy"
        f"--- REVIEWS ---\n{context_text}\n"
        f"---------------\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    gemini_response = call_gemini_api(prompt)

    return jsonify({"answer": gemini_response})

if __name__ == "__main__":
    app.run(debug=True)