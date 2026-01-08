# 🛍️ AI-Powered Amazon Review Assistant

This is a Flask web application that acts as an intelligent shopping assistant. It uses **Retrieval-Augmented Generation (RAG)** to analyze customer reviews and answer questions about Amazon products.

## 🚀 Features

* **Smart Analysis:** Extracts Product IDs (ASIN) directly from Amazon URLs.
* **RAG Architecture:** Uses **FAISS** and **Sentence Transformers** to find the most relevant reviews from a dataset.
* **Generative AI:** Powered by Google's **Gemini 1.5 Flash** to generate human-like, detailed answers.
* **Modern UI:** A clean, glassmorphism-styled interface built with Tailwind CSS.

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **AI/ML:** Google Gemini API, SentenceTransformers (all-MiniLM-L6-v2), FAISS (Vector Search)
* **Data Processing:** Pandas, NumPy
* **Frontend:** HTML5, Tailwind CSS, JavaScript

---

## 📦 Setup & Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone <your-repository-url>
    cd <project-folder>
    ```

2.  **Install Dependencies:**
    Make sure you have Python installed, then run:
    ```bash
    pip install flask pandas sentence-transformers faiss-cpu requests
    ```

3.  **Set your API Key:**
    You need a Google Gemini API key.
    * **Windows (PowerShell):** `$env:GEMINI_API_KEY="your_api_key_here"`
    * **Mac/Linux:** `export GEMINI_API_KEY="your_api_key_here"`

4.  **Prepare Data:**
    Ensure your `amazon_reviews.csv` file is in the root directory.

5.  **Run the App:**
    ```bash
    python app.py
    ```
    Open your browser and go to `http://127.0.0.1:5000`.

---

## 📝 Git Commands Used

Here is a reference of the Git commands used to set up and push this project:

### 1. Initialize & Commit
```bash
git init
git add .
git commit -m "Initial commit of AI Review Assistant"